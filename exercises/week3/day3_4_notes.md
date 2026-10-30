# Day 3-4 笔记：运行循环与结果处理

## 核心收获

Day 3-4 的核心问题：SDK 的 agent 运行循环和我的 mini_agent 有什么本质区别？答案是**骨架一样，但 SDK 在循环周围加了大量生产级机制**——并发安全、可观测性、断点恢复、多 agent 协作。

---

## 主循环骨架

所有 agent 框架的核心都是同一个 while 循环：

```python
# SDK 主循环（简化）—— run.py:1026
while True:
    result = await run_single_turn(agent, input, tools, ...)
    if result 是最终输出:
        break
    elif result 是 handoff:
        agent = 新 agent
        continue
    elif result 是工具调用后再跑:
        input = 加上工具结果
        continue
    elif result 是中断:
        break

# 我的 mini_agent
while True:
    response = client.chat.completions.create(messages=messages)
    if 没有工具调用:
        break
    else:
        执行工具，结果放回 messages
        continue
```

SDK 多处理了 handoff 和 interruption 两种情况，其余骨架完全一样。

---

## 调用链

```
Runner.run()                  ← 用户调的公开入口（run.py:275）
  ↓
AgentRunner._run_impl()       ← 内部实现，参数准备 + 状态恢复（run.py:623）
  ↓
while True:                   ← 主循环（run.py:1026）
  ↓
  run_single_turn()           ← 单轮执行（run_loop.py:2665）
    ↓
    get_single_step_result_from_response()  ← 响应解析（turn_resolution.py:3784）
```

Runner.run() 是对外入口，_run_impl() 是内部实现，前面带下划线不暴露给用户。这种拆法让公开接口保持稳定，内部可以自由调整。

---

## run_single_turn 完整流程

```
1. guard 并发安全检查（agent_tool_configuration.py）
   ↓
2. 触发 on_agent_start hooks（可观测性）
   ↓
3. 收集工具列表（get_all_tools + initialize_computer_tools）
   ↓
4. 获取 system_prompt 和 prompt_config（并发执行，gather_with_cancel）
   ↓
5. 解决工具同名冲突（resolve_tool_name_collisions）
   ↓
6. 调用 LLM（get_new_response → model.get_response）
   ↓
7. 处理响应、执行工具、返回结果（get_single_step_result_from_response）
```

---

## 并发安全机制

### 问题场景

生产环境多个请求共享同一个 agent 实例：

```
用户 A ──→ ┐
用户 B ──→ ├→ FastAPI 服务 → 共享同一个 agent 实例
用户 C ──→ ┘
```

如果某个 hook 里错误地修改了 agent.tools，其他正在运行的请求会受影响。

### guard_agent_tool_configuration 做了什么

在 hooks 执行前后做快照对比：

```
snapshot_before = agent.tools 的快照
  ↓
执行 hooks（可能修改 agent.tools）
  ↓
snapshot_after = agent.tools 的当前状态
  ↓
对比：如果不一致 → 抛出明确错误
```

### 正确做法：用 is_enabled 而不是修改 agent.tools

```python
def admin_only_tool(ctx, agent):
    return ctx.context.user_role == "admin"

tool = FunctionTool(name="delete_user", is_enabled=admin_only_tool, ...)
```

is_enabled 在每轮 get_all_tools() 时动态过滤，不修改 agent.tools 本身，并发安全。

---

## 可观测性：Hooks 机制

### 两个级别

- **RunHooks（全局）**：挂在 Runner.run() 上，监控所有 agent
- **AgentHooks（agent 级别）**：挂在单个 Agent 上，只监控该 agent

```python
# 全局监控
Runner.run(agent, run_config=RunConfig(hooks=logging_hooks))

# agent 专属监控
agent = Agent(name="客服", hooks=satisfaction_hooks)
```

### 回调时机

SDK 在关键节点调用回调：on_agent_start、on_llm_start、on_llm_end、on_tool_start、on_tool_end、on_handoff。

### AgentHookContext

hooks 是外部代码，不能直接访问 SDK 内部变量。SDK 把运行信息打包成 AgentHookContext 传递：
- context — 业务数据
- usage — token 用量
- turn_input — 当前轮输入

_share_tool_state_with 用 Python 引用赋值（不是拷贝）让 hook 和主流程共享同一份工具状态。

### gather_with_cancel

并发执行两个级别的 hooks，一个失败立刻取消另一个，避免半完成状态。

---

## _run_impl 主循环之前的准备工作

两条路径：

- **全新运行**：包装上下文、解析工具审批、准备输入数据
- **恢复运行**（断点续跑）：从 RunState 恢复中断状态、验证合法性、恢复 session

mini_agent 只支持从头跑，SDK 还支持断点续跑，所以准备工作多很多。

---

## Items：运行过程的数据记录

### RunItemBase 基类

两个核心字段：
- `agent` — 哪个 agent 产生的这条数据
- `raw_item` — 原始数据内容

### 关键子类

| 子类 | 含义 | mini_agent 对应 |
|------|------|----------------|
| MessageOutputItem | LLM 文本回复 | response.choices[0].message.content |
| ToolCallItem | LLM 要求调工具 | response.choices[0].message.tool_calls |
| ToolCallOutputItem | 工具执行结果 | {"role": "tool", "content": result} |
| HandoffCallItem | LLM 要求转交 | 无 |
| HandoffOutputItem | 转交记录（source_agent → target_agent） | 无 |

所有 Item 子类都是**纯数据容器，不含执行逻辑**。执行逻辑在 run.py 和 run_internal/ 里。

### call_id

LLM 服务端为每次工具调用生成的唯一 ID，用于配对：工具结果带上同一个 call_id 返回，LLM 才知道结果对应哪次调用。一次返回多个工具调用时尤其重要。

---

## Result：运行结果包装

### RunResultBase 核心字段

| 字段 | 作用 |
|------|------|
| input | 原始输入 |
| new_items | 运行过程每一步的记录（Item 列表） |
| raw_responses | LLM 原始响应列表 |
| final_output | 最终输出 |
| input/output_guardrail_results | 四种护栏的检查结果 |
| context_wrapper | 共享上下文（含 token 用量等） |

new_items 是完整的运行日志，final_output 只是最终结果。

---

## Turn Resolution：响应解析

### process_model_response

遍历 LLM 响应的 output 列表，按类型分拣：

```python
for output in response.output:
    if 是文本消息 → 放进 message 结果
    elif 是工具调用 → 放进 tool_calls 结果
    elif 是 handoff → 放进 handoff 结果
    elif 是推理过程 → 放进 reasoning 结果
```

和 mini_agent 的 `if tool_calls:` 本质一样，只是分类更细。

### 工具查找表

提前建三个 map，用名字 O(1) 查找工具对象：

```python
handoff_map = {handoff.tool_name: handoff for handoff in handoffs}
function_map = build_function_tool_lookup_map([...])
custom_tool_map = {tool.name: tool for tool in all_tools if isinstance(tool, CustomTool)}
```

特殊工具（ComputerTool、ShellTool 等）单独取出来，因为执行方式不同。

### 工具同名冲突处理

resolve_tool_name_collisions：同名时 handoff 优先，都同类型则最后一个赢，其余丢弃。策略可选 warn（静默去重）或 error（直接报错）。

---

## 多 Agent 架构本质

```
单 agent = while 循环（调 LLM + 执行工具）
多 agent = 调度层 + 多个 while 循环
```

这个 SDK 用 handoff 实现串行接力式多 agent，调度逻辑嵌在主循环里。其他框架有并行协作（AutoGen）、层级调度（CrewAI）、辩论投票等模式。

Handoff 不是硬编码的 if-else，而是 LLM 自行决定的——handoff 在 LLM 眼里就是一个工具，LLM 根据 instructions 和对话内容推理是否调用。

---

## 和 mini_agent 的关键差异总结

| 维度 | mini_agent | SDK |
|------|-----------|-----|
| 主循环 | 相同骨架 | 相同骨架 |
| 并发安全 | 无（单用户） | guard 快照检查 + is_enabled 过滤 |
| 可观测性 | print | Hooks 回调（两级） + Tracing |
| 断点恢复 | 无 | RunState 恢复 |
| 多 agent | 无 | Handoff 串行接力 |
| 数据记录 | 原始字典 | 类型化 Item 体系 |
| 结果包装 | 直接返回字符串 | RunResult（含 items、usage、guardrail 结果） |
| 错误处理 | 基础 try-except | gather_with_cancel + 异常传播 |
| 工具冲突 | 无 | 同名检测 + 自动去重/报错 |

---

## 疑问记录

1. get_single_step_result_from_response 里工具执行的并发控制细节？
2. 流式 run_streamed 和非流式 run 在主循环层面有何差异？
3. Tracing 系统的具体实现（Day 5 内容）
