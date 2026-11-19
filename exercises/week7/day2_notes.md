# Week 7 Day 2：Multi-Agent 协作模式

## 问题背景：为什么需要多个 Agent？

单 Agent 有四个瓶颈：

| 瓶颈 | 说明 | 多 Agent 怎么解决 |
|------|------|-----------------|
| 能力上限 | 一个 prompt 塞不下"既写代码又做审计又写文档"，每个能力都打折 | 每个 Agent 专精一件事，prompt 短且精准 |
| 上下文窗口 | 工具调用 20 次后，早期结果被注意力衰减淹没 | 每个 Agent 只看自己需要的上下文 |
| 延迟 | 所有步骤串行执行，独立子任务也得排队 | 独立子任务并行执行 |
| 可靠性 | 单点故障，第 8 步出错前 7 步白做 | 各 Agent 独立失败、独立重试 |

本质就一个字：**分**。把复杂任务拆给多个专精 Agent，每个有自己的 prompt、工具集、职责范围。

类比：微服务架构——每个服务做好一件事，通过明确接口协作。

---

## 5 种协作模式

### 1. Serial Handoff（串行交接）

```
Agent A ──→ Agent B ──→ Agent C ──→ 最终结果
       handoff     handoff
```

像工厂流水线。Agent A 做完后把完整对话历史交给 Agent B，控制权**转移**——A 退出，B 接管。

- **场景**：客服系统（前台分类 → 技术支持 → 满意度回访）、文档流水线（起草 → 审核 → 排版）
- **SDK 机制**：`handoffs=[agent_b]`，Agent A 的 handoff 列表声明它可以交给谁
- **特征**：A 自己决定何时交出控制权，交出后不再参与

### 2. Parallel Fan-out（并行分发）

```
         ┌──→ Agent A ──┐
任务 ────┼──→ Agent B ──┼──→ 汇总 Agent → 最终结果
         └──→ Agent C ──┘
```

同一个问题同时扔给多个 Agent，各做各的，最后汇总。

- **场景**：调研（同时搜论文、搜代码、搜新闻，最后综合）、多维度分析
- **SDK 机制**：`asyncio.gather(Runner.run(a), Runner.run(b), ...)`
- **特征**：子任务独立无依赖，纯粹靠并行省时间

### 3. Debate（辩论）

```
Agent 正方 ←──── 多轮交锋 ────→ Agent 反方
                    ↓
              Judge Agent → 最终结论
```

两个 Agent 持相反立场论证，第三个裁判 Agent 综合判断。

- **场景**：技术选型、风险评估、需要多角度分析的决策
- **SDK 机制**：多轮对话传递，各 Agent 能看到对方论点
- **特征**：通过对抗暴露单一视角的盲区

### 4. Voting（投票）

```
Agent 1 → 答案 X
Agent 2 → 答案 Y  ──→ 多数一致 → 最终答案
Agent 3 → 答案 X
```

多个 Agent **独立**回答（互相看不到），取多数一致的结果。

- **场景**：高可靠判断（安全漏洞检测、合同合规审查）
- **SDK 机制**：并行 `Runner.run()` + 结果聚合逻辑
- **特征**：用冗余换可靠性，类似分布式系统的 quorum 机制

### 5. Hierarchical（层级管理）

```
        Manager Agent（自主决策）
       ┌────┼────────┐
  Worker A  Worker B  Worker C
  （搜索）  （分析）   （写作）
```

Manager 拿到任务后自主决定调哪个 Worker、调几次、以什么顺序。Worker 返回结果给 Manager 继续决策。

- **场景**：项目管理、复杂多步任务（Manager 拆任务 → 分配 → 检查 → 汇总）
- **SDK 机制**：`agent.as_tool()` 把 Worker Agent 注册为 Manager 的工具
- **特征**：最灵活也最不可预测——Manager 的执行计划完全由 LLM 决定

---

## 两个关键设计选择

### 选择 1：Handoff 还是 Agent-as-Tool？

| 维度 | Handoff | Agent-as-Tool |
|------|---------|---------------|
| 控制权 | **转移**——当前 Agent 退出 | **保持**——当前 Agent 继续 |
| 对话历史 | 全部传递给下一个 Agent | Worker 只看到传入的参数 |
| 类比 | 转接电话（你挂了，同事接） | 问同事一个问题（你等答案，继续工作） |
| 适合 | 职责阶段切换（不同阶段不同专家） | 子任务辅助调用（需要某个能力帮忙） |

判断标准：**当前 Agent 做完自己的部分后还需要继续吗？**
- 需要 → Agent-as-Tool（调用子 Agent 拿结果，自己继续）
- 不需要 → Handoff（把整个任务交出去）

### 选择 2：何时该用 Multi-Agent？

核心公式：

```
协调成本 < 分工收益 → 用 Multi-Agent
协调成本 > 分工收益 → 用 Single Agent
```

协调成本包括：
- 每个 Agent 的 system prompt 占 token
- Agent 之间通信的额外 LLM 调用
- 调试复杂度指数增长
- 错误在 Agent 之间传播

**原则：能用单 Agent 就不用多 Agent，能用确定性 Workflow 就不用自主 Agent。**

来自 Anthropic 的复杂度阶梯：

```
简单 prompt → prompt chain → routing → parallel → agent → multi-agent
     ↑                                                          ↑
  大多数任务在这里就够了                               只有真正复杂的任务才到这里
```

---

## 模式选择决策树

```
任务来了
  │
  ├─ 子任务之间有依赖？
  │     ├─ 是 → 上下游阶段分明？
  │     │         ├─ 是 → Serial Handoff
  │     │         └─ 否 → Hierarchical（Manager 动态调度）
  │     └─ 否 → 需要高可靠性？
  │               ├─ 是 → Voting（多数一致）
  │               └─ 否 → 需要对抗性观点？
  │                         ├─ 是 → Debate（正反方 + 裁判）
  │                         └─ 否 → Parallel Fan-out（并行 + 汇总）
```

---

## 与 Day 1 规划算法的关系

Day 1 的算法和 Day 2 的模式不是互斥的，它们工作在不同层面：

| Day 1 算法 | Day 2 模式中的体现 |
|---|---|
| Plan-and-Solve | Hierarchical 中 Manager 用 Plan-and-Solve 做任务拆解 |
| Reflexion | 天然多 Agent：Executor + Evaluator + Reflector |
| ToT | Parallel Fan-out 可以看作简化版——并行探索多条路径 |

生产中的典型组合：
- 全局用 **Plan-and-Solve** 拆任务
- 独立子任务用 **Parallel Fan-out** 并行执行
- 每个子 Agent 内部用 **ReAct** 做工具调用
- 关键步骤用 **Voting** 确保可靠性
- 整体用 **Reflexion** 做质量检查和迭代

---

## SDK 实现要点

### Handoff 在 SDK 里是什么

```python
# 声明 Agent A 可以交接给 Agent B
agent_a = Agent(
    name="前台客服",
    handoffs=[agent_b],  # 允许交接的目标
    instructions="如果用户问技术问题，交给技术支持。"
)
```

SDK 内部：handoff 被转换成一个特殊的 tool call。当模型决定 handoff 时，Runner 循环把当前 Agent 替换为目标 Agent，对话历史原样传递。

### Agent-as-Tool 在 SDK 里是什么

```python
# 把 Worker 注册为 Manager 的工具
searcher = Agent(name="搜索员", ...)
analyst = Agent(name="分析员", ...)

manager = Agent(
    name="项目经理",
    tools=[searcher.as_tool(...), analyst.as_tool(...)],
    instructions="根据任务需要调用搜索员或分析员。"
)
```

SDK 内部：`as_tool()` 把 Agent 包装成一个 function_tool。Manager 调用时，SDK 启动一个子 Runner 执行 Worker，拿到结果作为 tool result 返回给 Manager。

### 并行执行

```python
# asyncio.gather 并行跑多个 Agent
results = await asyncio.gather(
    Runner.run(agent_a, "子任务 1"),
    Runner.run(agent_b, "子任务 2"),
    Runner.run(agent_c, "子任务 3"),
)
# 汇总
combined = "\n".join(r.final_output for r in results)
```

这不是 SDK 的专属功能，就是 Python 的异步并发。SDK 的 `Runner.run()` 本身是 async 的，天然支持并行。
