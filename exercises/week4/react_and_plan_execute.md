# Agent 规划策略：ReAct 与 Plan-then-Execute

## 背景：为什么需要规划策略

最基本的 agent 循环（也就是 mini_agent 的做法）是这样的：

```
用户输入 → LLM 决定调工具 → 执行工具 → 结果喂回 LLM → LLM 继续或回答
```

LLM 在每一步直接输出行动（工具调用），中间没有显式的思考过程，也没有全局规划。这在简单任务上够用，但复杂任务会出问题：

- LLM 走一步看一步，容易偏离目标
- 出错后没有反思，可能重复犯同样的错
- 多步骤任务没有全局视角，可能遗漏关键步骤或做了不必要的步骤

ReAct 和 Plan-then-Execute 是两种解决这个问题的策略，分别从不同角度切入。

---

## ReAct（Reasoning + Acting）

### 论文

Yao et al., 2022. "ReAct: Synergizing Reasoning and Acting in Language Models"

### 核心思想

在每次行动之前，强制 LLM 先写出一段思考过程。把"想"和"做"交替进行，而不是直接跳到行动。

### 格式

ReAct 通过 prompt 模板约定一个固定格式，LLM 每一步按这个格式输出：

```
Thought: 我需要先查一下东京的天气，因为用户问的是今天该穿什么。
Action: get_weather(city="东京")
Observation: 晴，28°C，湿度 65%

Thought: 28°C 比较热，湿度也不低，应该建议穿轻薄透气的衣服。不需要再调工具了。
Action: 直接回答用户
Answer: 东京今天 28°C，晴天，建议穿短袖或轻薄衬衫，可以带把折叠伞防晒。
```

### 和普通 agent 循环的对比

```
普通循环：  Action → Observation → Action → Observation → Answer
ReAct：    Thought → Action → Observation → Thought → Action → Observation → Answer
```

区别就在于多了 Thought 步骤。看起来只是多了几句话，但效果显著：

1. **思考过程可审计** — 你能看到 LLM 为什么做这个决定，方便调试
2. **减少冲动行动** — LLM 先想再做，不会一上来就乱调工具
3. **自我纠错** — Thought 阶段可以反思上一步的结果，发现问题后换方向

### 实现方式

不需要改代码架构，只需要改 system prompt：

```python
REACT_PROMPT = """你是一个助手。每次行动前，先写出你的思考过程。

严格按照以下格式：

Thought: 分析当前情况，决定下一步做什么
Action: 调用工具（如果需要）
Observation: 工具返回的结果（这个由系统填充，你不要写）

重复以上步骤，直到你能给出最终回答。

Thought: 我已经有足够信息回答用户了
Answer: 最终回答
"""
```

### 局限

- 每步都要写 Thought，增加了 token 消耗
- 是局部思考，没有全局规划——第一步时 LLM 不知道整个任务一共需要几步
- 遇到需要探索多条路径的任务（"可能是 A 的问题，也可能是 B"），线性的 Thought-Action 链不够用

---

## Plan-then-Execute

### 论文

Wang et al., 2023. "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models"

### 核心思想

把任务分成两个阶段：先生成一个完整的执行计划，再按计划逐步执行。

```
阶段 1 — Plan:   用户任务 → LLM 生成步骤列表
阶段 2 — Execute: 按步骤逐个执行，每步可以调工具
阶段 3 — Replan（可选）: 某步失败或发现新信息时，修改剩余计划
```

### 具体流程

```
用户输入："帮我分析当前目录下所有 Python 文件的代码行数，按从多到少排序，写入 report.txt"

--- Plan 阶段 ---
LLM 输出计划：
  Step 1: 用 list_directory 列出当前目录的文件
  Step 2: 从结果中筛选出 .py 文件
  Step 3: 对每个 .py 文件调用 count_lines 统计行数
  Step 4: 按行数从多到少排序
  Step 5: 用 write_file 把结果写入 report.txt
  Step 6: 确认文件写入成功，回复用户

--- Execute 阶段 ---
  执行 Step 1: list_directory(".") → 得到文件列表
  执行 Step 2: 筛选出 3 个 .py 文件
  执行 Step 3: count_lines 逐个统计
  执行 Step 4: 排序（LLM 自己做，不需要工具）
  执行 Step 5: write_file("report.txt", ...) → 成功
  执行 Step 6: 回复用户"已完成"

--- Replan（假设 Step 3 某个文件出错了）---
LLM 修改计划：
  Step 3b: 跳过出错的文件，在报告中标注"读取失败"
  其余步骤不变
```

### 和 ReAct 的对比

| 维度 | ReAct | Plan-then-Execute |
|------|-------|-------------------|
| 规划时机 | 每步局部思考 | 开始前全局规划 |
| 全局视角 | 没有，走一步看一步 | 有，知道一共几步 |
| 灵活性 | 高，随时调整方向 | 中，需要显式 replan |
| Token 消耗 | 每步多一段 Thought | Plan 阶段集中消耗 |
| 适合的任务 | 探索性、不确定步骤数的 | 步骤可预知、流程明确的 |
| 可审计性 | 看每步的 Thought | 看计划 + 执行日志 |
| 出错时 | 下一个 Thought 里反思 | 触发 Replan 修改计划 |

### 实现方式

需要两个阶段的 prompt：

```python
# 阶段 1：生成计划
PLAN_PROMPT = """分析用户的任务，生成一个分步执行计划。
每步包含：步骤编号、要做什么、需要什么工具。
输出 JSON 格式：
[
  {"step": 1, "description": "列出目录文件", "tool": "list_directory"},
  {"step": 2, "description": "筛选 .py 文件", "tool": null},
  ...
]
"""

# 阶段 2：逐步执行
EXECUTE_PROMPT = """你正在按计划执行任务。
当前计划：{plan}
已完成：{completed_steps}
当前步骤：{current_step}
请执行当前步骤。
"""
```

### 局限

- Plan 阶段的计划质量依赖 LLM 对任务的理解，任务描述模糊时计划可能不靠谱
- 计划是静态的，执行过程中遇到意外需要显式 replan，逻辑更复杂
- 简单任务（一两步就完成）不需要计划，Plan 阶段反而是浪费

---

## 什么时候用哪个

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 简单任务（1-3 步） | 普通循环就够 | 不需要规划，加了反而浪费 token |
| 中等任务（3-8 步），步骤基本可预知 | Plan-then-Execute | 全局规划避免遗漏步骤 |
| 探索性任务，不确定需要几步 | ReAct | 灵活应对，边想边做 |
| 高可靠性要求 | ReAct + Plan | 先做计划，执行时每步加 Thought |
| 需要调试/审计 | ReAct | Thought 链是天然的调试日志 |

## 组合使用

两种策略不互斥，可以组合：

```
1. 先用 Plan-then-Execute 生成全局计划
2. 执行每一步时用 ReAct 格式（Thought → Action → Observation）
3. 某步失败时，Thought 阶段反思原因，决定是重试还是触发 Replan
```

这就是很多生产级 agent 框架的实际做法。

---

## 更高级的规划算法（Week 8 会学）

| 算法 | 在 ReAct/Plan-Execute 基础上的改进 |
|------|-----------------------------------|
| Tree of Thoughts (ToT) | 不是线性链，而是在多条思路之间搜索最优路径 |
| Reflexion | 执行失败后做深度反思，把经验存入记忆，下次避免同样的错误 |
| LATS | 蒙特卡洛树搜索 + LLM，在复杂决策空间中寻找最优策略 |

这些都是更复杂的规划策略，建立在 ReAct 和 Plan-then-Execute 的基础概念之上。
