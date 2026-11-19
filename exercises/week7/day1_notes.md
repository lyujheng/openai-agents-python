# 第7周 Day 1：高级规划算法

## 问题背景

第 2-4 周手搓的 mini_agent，核心循环本质上就是 ReAct：

```
用户输入 → [LLM 思考 → 选工具 → 执行 → 观察结果] × N → 最终回答
```

ReAct 是线性的，每一步只看当前状态决定下一步。很多简单任务够用，但遇到以下场景会出问题。

### 场景一：需要全局规划

任务："分析一个项目的代码质量并生成报告"

ReAct 会走一步看一步：先读了一个文件，分析完了再想下一个读什么。它可能漏掉重要文件，也可能在不重要的文件上花太多时间。需要的是：先扫描目录结构，制定完整的分析计划，再按计划执行。

### 场景二：需要探索多条路径

任务："诊断一个 bug 的根因"

可能的原因有 3 个：配置错误、依赖版本冲突、代码逻辑 bug。ReAct 只能选一条路走下去，走错了就浪费了。需要的是：同时探索多条假设，评估哪条最有希望，再深入。

### 场景三：需要从失败中学习

任务："写一段通过所有测试的代码"

第一版没通过测试，ReAct 只会看到错误信息决定下一步，没有"记住上次为什么错、这次应该避免什么"的机制。需要的是：执行 → 反思 → 把教训存起来 → 带着经验重试。

---

## 应用业务场景

| 算法 | 典型业务场景 | 为什么 ReAct 不够 |
|------|-------------|-------------------|
| Plan-and-Solve | 写调研报告、数据分析流水线、项目迁移方案 | 子任务有依赖顺序，需要先全局规划再执行 |
| Tree of Thoughts | Bug 诊断、方案选型、创意生成、开放式问题 | 需要同时考虑多种可能性，择优深入 |
| Reflexion | 代码生成/修复、文章写作、达标类任务 | 需要多次迭代改进，从失败中学习 |
| LATS | 复杂博弈决策、多步推理证明 | 需要回溯和蒙特卡洛搜索（成本高，实际用得少） |

### 同一任务下四种算法的行为对比

> 任务："对比 Python GIL 和 Rust Ownership，分析它们如何解决并发安全问题"

- **ReAct**：搜索 GIL → 看到结果 → 搜索 Rust ownership → 看到结果 → 写回答。线性的，没有全局视角。
- **Plan-and-Solve**：先规划"第 1 步搜 GIL，第 2 步搜 Ownership，第 3 步搜 Python 并发方案，第 4 步搜 Rust 并发模型，第 5 步综合对比"。有全局结构。
- **Tree of Thoughts**：生成 3 条思路（从内存安全切入 / 从性能切入 / 从开发体验切入），打分，选最好的展开。
- **Reflexion**：写一版回答 → 自评"缺少具体代码示例" → 带着这个教训重来，补上示例。

---

## 技术原理

### 1. Plan-and-Solve（Wang et al., 2023）

核心思想：把规划和执行分成两个独立阶段。

```
                    ┌──────────────────┐
  用户任务  ───────→│  Planner（规划）  │
                    │  生成步骤列表     │
                    └────────┬─────────┘
                             │ [step1, step2, step3, ...]
                             ▼
                    ┌──────────────────┐
                    │  Executor（执行） │ ← 按顺序逐步执行
                    │  调用工具/推理    │
                    └────────┬─────────┘
                             │ 各步骤的结果
                             ▼
                    ┌──────────────────┐
                    │ Synthesizer（综合）│
                    │  汇总最终回答     │
                    └──────────────────┘
```

关键设计决策：

- **Planner 的输出是结构化的**：JSON 格式的步骤列表，每步包含动作类型和参数。
- **Executor 不做决策**：严格按计划执行，不临时改计划。
- **优点**：全局视野、步骤有序、可预测。
- **缺点**：计划一旦定下来就是死的，执行中发现新信息无法调整（除非加 replan 机制）。

和 SDK 的映射关系：

- Planner 就是一个 `Agent(output_type=PlanSchema)`，输出结构化的计划。
- Executor 是 Python 代码循环执行计划中的每一步。
- 这其实就是 examples 里的 **deterministic workflow** —— Python 代码控制流程，LLM 只负责每一步的具体执行。

### 2. Tree of Thoughts / ToT（Yao et al., 2023）

核心思想：把 LLM 的推理过程建模为一棵树，用搜索算法（BFS/DFS）找最优路径。

```
                        [任务]
                       /  |  \
                     /    |    \
                  思路A  思路B  思路C     ← 第1层：生成 + 打分
                  (8分)  (6分)  (9分)
                  / \           / \
               A1   A2       C1   C2     ← 第2层：择优展开
              (7分)(5分)    (8分)(9分)
                                  |
                                 C2.1    ← 最佳路径
```

三个关键操作：

1. **生成（Generate）**：让 LLM 对当前节点提出多条不同的下一步思路。
2. **评估（Evaluate）**：让 LLM 给每条思路打分（0-10），评估其对任务推进的价值。
3. **搜索（Search）**：选得分高的节点继续展开（BFS 保留 top-k，DFS 贪心选最优）。

关键设计决策：

- **breadth（宽度）**：每个节点展开几条子思路。越大探索越充分，但 LLM 调用翻倍。
- **depth（深度）**：展开几层。越深推理越深入，但成本指数增长。
- **剪枝**：只保留 top-k 节点继续展开，砍掉低分支。
- **工具集成**：思路节点可以附带工具调用，执行后把结果加入节点信息。

成本分析（ToT 的核心权衡）：

- 宽度 3、深度 2 的场景：第 1 层生成 3 个节点 + 打分 1 次，第 2 层每个优选节点再生成 3 个 + 打分 1 次。
- LLM 调用次数 ≈ `depth × (生成调用 + 打分调用)`，是 ReAct 的好几倍。
- 所以 ToT 只适合"值得花成本探索"的高价值决策，日常简单任务用它就是浪费。

### 3. Reflexion（Shinn et al., 2023）

核心思想：给 agent 加一个"自我反思"循环和"经验记忆"。

```
                    ┌───────────────┐
  任务 + 记忆 ─────→│  Executor     │──→ 结果
                    │  执行任务      │
                    └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Evaluator    │──→ 通过？──→ 结束
                    │  评估结果质量  │
                    └───────────────┘
                            │ 不通过
                            ▼
                    ┌───────────────┐
                    │  Reflector    │──→ "上次的问题是X，下次应该Y"
                    │  分析失败原因  │
                    └───────┬───────┘
                            │ 经验存入记忆
                            ▼
                    ┌───────────────┐
                    │   Memory      │──→ 回到 Executor，带着经验重试
                    │  经验记忆池    │
                    └───────────────┘
```

三个独立组件：

1. **Executor**：执行任务的 agent，接收任务 + 历史经验作为输入。
2. **Evaluator**：独立评估结果质量，输出分数和具体问题列表。
3. **Reflector**：分析"为什么失败"，提取"下次应该怎么做"的经验教训。

Memory 的作用是关键 —— 它不是简单地把上次的错误传回去，而是把提炼后的经验教训传给下一次尝试：

- 没有 Reflexion：`"上次错了" → 重试`（可能犯同样的错）
- 有 Reflexion：`"上次错了，原因是 X，下次应该 Y" → 带着经验重试`（定向改进）

关键设计决策：

- **评估标准要明确**：分数 + 具体问题列表，不能只说"不好"。
- **反思要可执行**：不是"我应该做得更好"，而是"下次应该先搜索 X 再搜索 Y"。
- **max_attempts 限制**：防止无限反思循环。
- **每次是完整重来**：不是修补上一次的结果，而是从头开始（但带着经验）。

### 4. LATS（Zhou et al., 2023）—— 了解即可

LATS = Language Agent Tree Search，把蒙特卡洛树搜索（MCTS）搬到 LLM agent 上。

它比 ToT 多了一个关键机制：**回溯**。ToT 选了一条路就走下去，LATS 可以走到一半发现不对，回退到之前的节点换一条路。代价是成本更高、实现更复杂，实际工程中用得很少。知道有这个方向就行。

---

## 核心对比

| 维度 | Plan-and-Solve | Tree of Thoughts | Reflexion |
|------|---------------|------------------|-----------|
| 核心思路 | 先规划再执行 | 广度搜索择优 | 试错 + 反思 + 记忆 |
| 决策时机 | 执行前一次性决策 | 每层评估决策 | 每轮结束后决策 |
| 能否回溯 | ❌ 不能 | ⚠️ 层间剪枝 | ✅ 完整重来 |
| LLM 调用量 | 低（规划 + 执行 + 综合） | 高（生成 × 层数 + 打分 × 层数） | 中到高（尝试次数 × 每次完整执行） |
| 适合的任务特征 | 可预先分解的结构化任务 | 有多种可能解的开放式问题 | 有明确质量标准的迭代任务 |

---

## 选型决策树

```
你的任务是什么样的？
│
├─ 可以预先拆成明确的子步骤？
│   └─ YES → Plan-and-Solve
│       （写报告、数据分析流水线、项目迁移）
│
├─ 有多种可能的解法，需要探索？
│   └─ YES → Tree of Thoughts
│       （Bug 诊断、方案选型、创意生成）
│
├─ 有明确的"好/坏"判定标准，需要迭代改进？
│   └─ YES → Reflexion
│       （代码生成、文章写作、测试达标）
│
└─ 简单直接，一步一步走就行？
    └─ YES → ReAct
        （问答、简单工具调用、信息检索）
```

---

## 和 ReAct 的本质区别

ReAct 的隐含假设是"每一步的局部最优选择能导向全局最优结果"。三种进阶算法各自打破了这个假设的一个方面：

- **Plan-and-Solve** 打破"不需要全局视角"：先看全局再做局部。
- **Tree of Thoughts** 打破"每步只走一条路"：每步探索多条路，择优。
- **Reflexion** 打破"不需要跨轮记忆"：失败的经验指导下一轮。

在生产环境中，这三种算法经常组合使用。例如一个代码生成 agent 可能：
1. 用 Plan-and-Solve 规划整体实现步骤
2. 在每步实现中用 Reflexion 迭代到通过测试
3. 遇到设计抉择时用 ToT 探索不同方案

---

## 实现详解

练习代码位置：`exercises/week7/day1_planning_algorithms.py`

三种算法共享同一套基础设施：一个知识库、三个工具函数、一个 LLM 调用封装、一个指标收集器。各算法在此之上实现自己的控制流。

### 共享基础设施

#### 知识库和工具

```python
# 8 条知识，涵盖 Python/Rust/vLLM，用 dict 模拟向量数据库。
KNOWLEDGE_BASE = {
    "python_gc": "Python 使用引用计数为主、分代回收为辅...",
    "python_gil": "GIL 保证同一时刻只有一个线程执行字节码...",
    "rust_ownership": "每个值有且只有一个 owner...",
    ...
}

# 三个工具函数，agent 通过工具名调用。
TOOLS = {
    "search_knowledge": search_knowledge,    # 关键词匹配搜索
    "compare_concepts": compare_concepts,    # 对比两个概念（内部调两次 search）
    "summarize_text": summarize_text,        # 文本截断（模拟摘要）
}
```

工具是纯 Python 函数，不走 function calling —— 这里故意用最朴素的方式实现，让注意力聚焦在规划算法本身的控制流上，而不是 SDK 的工具调度。

#### LLM 调用封装

```python
def call_llm(messages, temperature=0.0) -> str:
    """纯文本回答。"""

def call_llm_json(messages, temperature=0.0) -> dict:
    """要求返回 JSON，解析失败则清洗 markdown 包装后重试一次。"""
```

`call_llm_json` 的清洗逻辑处理了中转站常见的问题：模型把 JSON 包在 ` ```json ... ``` ` 里。这是 week1 练习 4 就遇到过的问题。

#### 指标收集

```python
@dataclass
class PlanningMetrics:
    algorithm: str          # 算法名
    task: str               # 任务描述
    result: str             # 最终回答
    llm_calls: int          # LLM 调用总次数
    tool_calls: int         # 工具调用总次数
    elapsed_seconds: float  # 耗时
    steps_log: list[str]    # 执行日志（每步记一条）
```

每种算法在执行过程中通过 `m.llm_calls += 1`、`m.tool_calls += 1`、`m.log(...)` 记录指标，最后汇总对比。

---

### 算法一：Plan-and-Solve 实现

#### 控制流

```
plan_and_solve(task)
│
├─ 阶段 1：Planner（1 次 LLM 调用）
│   prompt: "把任务拆成 3-6 步，每步包含 action/args/purpose"
│   返回: JSON 数组 [{step_number, action, args, purpose}, ...]
│
├─ 阶段 2：Executor（遍历步骤列表）
│   for step in plan:
│       if action == "think":     → 1 次 LLM 调用（推理）
│       elif action in TOOLS:     → 直接调用 Python 函数（不走 LLM）
│       结果追加到 context_parts
│
└─ 阶段 3：Synthesizer（1 次 LLM 调用）
    prompt: "根据收集到的所有信息，给出最终回答"
    返回: 最终文本
```

#### LLM 交互方式

Plan-and-Solve 和 LLM 的交互是"分段式"的：
- Planner 阶段：LLM 扮演规划者，输出结构化 JSON
- think 步骤：LLM 扮演分析者，基于已有上下文推理
- Synthesizer 阶段：LLM 扮演综合者，汇总所有信息

三次 LLM 调用的 system prompt 各不相同，各自负责一个职责。这和 SDK 的 multi-agent 思路一致 —— 每个 agent 有自己的 instructions。

#### 关键代码段

```python
# Planner 的 prompt 要求 LLM 输出结构化计划。
# 注意 "只返回 JSON 数组" —— 约束输出格式是让 LLM 当规划者的关键。
plan_prompt = [
    {"role": "system", "content": (
        "你是一个规划助手。用户会给你一个研究任务，你需要拆解成 3-6 个执行步骤。\n"
        "每个步骤包含：step_number、action、args、purpose。\n"
        f"可用工具：{list(TOOLS.keys())}\n"
        "只返回 JSON 数组，不要加任何解释。"
    )},
    {"role": "user", "content": task},
]
plan = call_llm_json(plan_prompt)

# Executor 不做任何决策，严格按计划执行。
# 这就是 "deterministic workflow" —— Python 代码控制流程。
for step in plan:
    action = step.get("action", "think")
    args = step.get("args", {})

    if action == "think":
        thought = call_llm(think_prompt)   # LLM 做推理
    elif action in TOOLS:
        result = TOOLS[action](**args)      # 直接调函数
```

#### 成本模型

```
LLM 调用 = 1（规划）+ think 步骤数 + 1（综合）
工具调用 = 计划中工具步骤的数量

典型值：计划 5 步其中 2 步是 think → LLM 调用 = 1+2+1 = 4 次
```

---

### 算法二：Tree of Thoughts 实现

#### 数据结构

```python
@dataclass
class ThoughtNode:
    thought: str                          # 这个节点的思考内容
    score: float = 0.0                    # LLM 评估的分数（0-10）
    parent: ThoughtNode | None = None     # 父节点（用于回溯路径）
    children: list[ThoughtNode]           # 子节点
    depth: int = 0                        # 在树中的深度
    tool_results: list[str]               # 这个思路附带的工具调用结果

    def path(self) -> list[str]:
        """从根走到当前节点的完整思考路径。"""
        # 沿 parent 指针一路向上，再反转。
```

这棵树不是一次性构建的，而是逐层生长：生成一层 → 评估一层 → 剪枝 → 生成下一层。

#### 控制流

```
tree_of_thoughts(task, breadth=3, depth=2)
│
├─ 初始化 root = ThoughtNode(task)
│
├─ 第 1 层循环
│   ├─ 生成（1 次 LLM 调用，temperature=0.3 增加多样性）
│   │   prompt: "给出 3 条不同的下一步思路"
│   │   返回: [{thought, tool}, {thought, tool}, {thought, tool}]
│   │   → 创建 3 个子节点，如果有 tool 则执行
│   │
│   ├─ 评估（1 次 LLM 调用）
│   │   prompt: "给每条思路打分 0-10"
│   │   返回: [{index, score, reason}, ...]
│   │   → 写入各节点的 score 字段
│   │
│   └─ 剪枝
│       按 score 排序，只保留 top-k 进入下一层
│       k = max(1, breadth // 2) = 1（breadth=3 时）
│
├─ 第 2 层循环（同上，但只展开剪枝后保留的节点）
│
└─ 选最佳叶子节点
    沿 parent 回溯收集完整路径和所有工具结果
    → 1 次 LLM 调用生成最终回答
```

#### LLM 交互方式

ToT 和 LLM 有两种不同的交互模式：
- **发散模式**（生成）：temperature=0.3，要求 LLM 提出不同的想法。prompt 明确说"3 条不同的思路"。
- **收敛模式**（评估）：temperature=0.0，要求 LLM 客观打分。prompt 给出明确的评分标准。

这是 ToT 的核心设计：用同一个 LLM 既当"创意提出者"又当"评审官"，通过 prompt 切换角色。

#### 关键代码段

```python
# 生成子思路 —— 注意 temperature=0.3，比默认的 0.0 高，鼓励多样性。
gen_prompt = [
    {"role": "system", "content": (
        "你是一个创造性思考助手。"
        f"请提出 {breadth} 条不同的下一步思路。\n"
        "每条思路包含：thought 和 tool（需要用工具则指定，否则 null）。"
    )},
    {"role": "user", "content": f"任务：{task}\n\n已有路径：\n{path_so_far}"},
]
children_data = call_llm_json(gen_prompt, temperature=0.3)

# 如果思路涉及工具调用，立刻执行，结果附加到节点上。
# 这样评估阶段就能看到"这条路实际查到了什么"来打分。
tool_spec = cd.get("tool")
if tool_spec and isinstance(tool_spec, dict):
    tool_name = tool_spec.get("name", "")
    if tool_name in TOOLS:
        result = TOOLS[tool_name](**tool_args)
        child.tool_results.append(f"[{tool_name}] {result}")

# 打分 —— 把所有思路及其工具结果一起呈现给 LLM 评估。
scoring_items = []
for i, node in enumerate(next_level):
    tool_info = "; ".join(node.tool_results) if node.tool_results else "无工具调用"
    scoring_items.append(f"{i}: {node.thought} [工具结果: {tool_info}]")

# 剪枝 —— 只保留得分最高的进入下一层。
next_level.sort(key=lambda n: n.score, reverse=True)
current_level = next_level[:max(1, breadth // 2)]
```

#### 成本模型

```
每层 LLM 调用 = 节点数 × 1（生成）+ 1（整层打分）
总 LLM 调用 ≈ depth × (展开的节点数 + 1) + 1（最终回答）

breadth=3, depth=2 的典型值：
  第 1 层：1（生成 root 的 3 个子节点）+ 1（打分）= 2
  第 2 层：1（生成保留节点的子节点）+ 1（打分）= 2
  最终：1
  总计 ≈ 5 次 LLM 调用

对比 Plan-and-Solve 的 ~4 次，ToT 并没有贵很多。
但如果 breadth=5, depth=3，成本会快速膨胀。
```

---

### 算法三：Reflexion 实现

#### 控制流

```
reflexion(task, max_attempts=3)
│
├─ memory = []（空的经验池）
│
├─ 第 1 次尝试
│   ├─ Executor（2 次 LLM 调用）
│   │   ├─ call_llm_json: "决定搜索什么" → 返回 {tool_calls, thinking}
│   │   ├─ 执行工具调用
│   │   └─ call_llm: "根据搜索结果回答" → 返回回答文本
│   │
│   ├─ Evaluator（1 次 LLM 调用）
│   │   prompt: "评估回答质量，打分 1-10，列出具体问题"
│   │   返回: {score, pass, issues}
│   │   → pass=true → 结束
│   │   → pass=false → 进入反思
│   │
│   └─ Reflector（1 次 LLM 调用）
│       prompt: "分析失败原因，提取经验教训"
│       返回: {reflection: "下次应该..."}
│       → 存入 memory
│
├─ 第 2 次尝试
│   ├─ Executor（带着 memory 重新执行）
│   │   prompt 中多了："从之前尝试中学到的经验：\n- [尝试1] 下次应该..."
│   │   → agent 能看到上次的教训，避免重复犯错
│   │
│   ├─ Evaluator → pass=true → 结束
│   └─ ...
│
└─ 第 3 次尝试（最后兜底）
    无论评估是否通过，都使用这次的结果
```

#### LLM 交互方式

Reflexion 有三种不同角色的 LLM 调用，各自的 prompt 设计目标不同：

```python
# Executor —— 要求返回结构化的工具调用计划。
# 关键：把 memory 作为上下文传入。
exec_prompt = [
    {"role": "system", "content": "你是一个研究助手，需要决定搜索什么信息..."},
    {"role": "user", "content": (
        f"任务：{task}\n\n"
        f"从之前尝试中学到的经验：\n{memory_text}\n\n"  # ← 这里是关键
        "请先搜索所需信息。"
    )},
]

# Evaluator —— 要求严格打分 + 列出具体问题。
# 关键：评分标准和通过线要写清楚（"7 分以上算通过"）。
eval_prompt = [
    {"role": "system", "content": (
        "你是一个严格的评估者。评估回答的质量。\n"
        "返回 JSON：{score: 1-10, pass: true/false, issues: [...]}\n"
        "评分标准：准确性、完整性、条理性、是否有具体依据。\n"
        "7 分以上算通过。"
    )},
]

# Reflector —— 要求产出可执行的经验教训，不是泛泛的感想。
# 关键：把评估问题和使用的工具都传给它，让它能做针对性分析。
reflect_prompt = [
    {"role": "system", "content": (
        "你是一个反思助手。分析失败原因，提取经验教训。\n"
        "返回 JSON：{reflection: '具体的经验教训，下次应该怎么做'}"
    )},
    {"role": "user", "content": (
        f"任务：{task}\n"
        f"本次回答：\n{answer}\n"
        f"评估问题：{issues}\n"          # ← Evaluator 指出的问题
        f"使用的工具：{tool_names}\n"     # ← 用了哪些工具
        "请反思哪里做得不够好，下次应该怎么改进。"
    )},
]
```

#### Memory 的累积效果

```
第 1 次：memory = []
  → Executor 没有历史参考，可能漏搜一些信息
  → Evaluator: "缺少 Rust 方面的对比数据"
  → Reflector: "下次应该同时搜索 Python 和 Rust 的信息"
  → memory = ["[尝试1] 下次应该同时搜索 Python 和 Rust 的信息"]

第 2 次：memory = ["[尝试1] 下次应该同时搜索 Python 和 Rust 的信息"]
  → Executor 看到经验，这次同时搜了两方面
  → Evaluator: "对比分析不够深入，缺少具体场景"
  → Reflector: "应该用 compare_concepts 工具直接对比，而不是分开搜"
  → memory 追加 "[尝试2] 应该用 compare_concepts 工具..."

第 3 次：memory 包含两条经验
  → Executor 同时参考两条教训，调用 compare_concepts + 补充搜索
  → Evaluator: pass = true
```

每次反思都是在**累积**经验，不是替换。第 3 次尝试能同时利用第 1 次和第 2 次的教训。

#### 成本模型

```
每次尝试 = 2（Executor）+ 1（Evaluator）+ 1（Reflector，最后一次没有）
         = 3-4 次 LLM 调用

最好情况（第 1 次就通过）：3 次 LLM
最差情况（跑满 3 次）：3 + 4 + 3 = 10 次 LLM

对比：Plan-and-Solve ~4 次，ToT ~5 次
Reflexion 的成本方差最大，取决于任务难度和评估标准的严格程度。
```

---

### 对比实验设计

代码在同一个任务上运行三种算法，收集四个指标：

```
任务: "对比 Python GIL 和 Rust Ownership，分析各自如何解决并发安全问题"

         算法          LLM调用  工具调用  耗时   步骤数
         Plan-and-Solve    ~4      ~3      低     ~8
         Tree-of-Thoughts  ~5      ~2      中     ~10
         Reflexion         3~10    ~2      高     ~6-15
```

观察维度：
- **LLM 调用次数**：直接反映 token 成本
- **工具调用次数**：反映信息收集的充分程度
- **耗时**：LLM 调用都是串行的，调用越多越慢
- **最终回答质量**：主观评估，看是否完整、有条理、有依据

运行方式：

```bash
export DEEPSEEK_API_KEY="你的key"
uv run python exercises/week7/day1_planning_algorithms.py
```
