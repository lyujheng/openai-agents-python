# Day 1 笔记：入口和 Agent 基本概念

## 示例观察

### 最小 agent 代码

```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You only respond in haikus.",
)
result = await Runner.run(agent, "Tell me about recursion in programming.")
print(result.final_output)
```

三个要素：Agent 定义 + Runner.run() 驱动 + result.final_output 取结果。

### @tool / @function_tool 做了什么

`@tool` 是 `@function_tool` 的别名（`decorators.py` 只有一行 `tool = function_tool`）。

它把一个普通 Python 函数包装成 SDK 能识别的 `FunctionTool` 对象，自动从函数签名生成 JSON Schema：
- 参数类型注解 → JSON Schema type
- `Annotated[str, "描述"]` → JSON Schema description
- docstring 第一行 → tool description
- Pydantic Model 返回值 → 自动序列化

对比我 week2 手写的 `func_to_schema()`：做的是同一件事，但 SDK 支持 Pydantic Model、Annotated、Field 等复杂类型，我的版本只支持 6 种基础类型。

### run() vs run_streamed() 的区别

| | `Runner.run()` | `Runner.run_streamed()` |
|--|---------------|------------------------|
| 调用方式 | `await` 等全部完成 | 不 await，立即返回流对象 |
| 拿结果 | `result.final_output` 一次性拿完整字符串 | `async for event in result.stream_events()` 逐 token 消费 |
| 用户体验 | 等 LLM 想完了才显示 | 边生成边显示，像 ChatGPT 一样逐字输出 |

流式消费的核心模式：
```python
result = Runner.run_streamed(agent, input="...")
async for event in result.stream_events():
    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
        print(event.data.delta, end="", flush=True)
```

---

## Agent 类字段分析

### 按功能分组

| 分组 | 字段 | 我的 mini_agent 有吗？ |
|------|------|----------------------|
| 身份 | `name`, `instructions` | ✅ 有 system_prompt，没有 name |
| 工具 | `tools`, `mcp_servers` | ✅ 有 tools dict |
| 输出控制 | `output_type` | ❌ |
| 护栏 | `input_guardrails`, `output_guardrails` | ❌ |
| 交接 | `handoffs`, `handoff_description` | ❌ |
| 生命周期 | `hooks` | ❌ |
| 模型配置 | `model`, `model_settings` | ❌ 硬编码 client 和 model 名 |

### instructions 的多种形式

类型签名：`str | Callable[[RunContextWrapper, Agent], MaybeAwaitable[str]] | None`

1. **静态字符串**：`instructions="You only respond in haikus."` — 行为固定
2. **同步函数**：`instructions=custom_instructions` — 根据运行时 context 动态生成
3. **async 函数**：动态生成时需要异步操作（如从数据库读取 prompt）
4. **None**：不传，使用默认行为

动态 instructions 解决的问题：同一个 agent 定义，面对不同用户/场景需要不同的 system prompt（如根据用户语言、权限级别调整行为）。

---

## 和 mini_agent 的关键差异

| # | 差异 | mini_agent | SDK Agent | 解决的问题 |
|---|------|-----------|-----------|-----------|
| 1 | 定义与执行分离 | 一个类揉在一起 | Agent（声明）+ Runner（执行）分开 | 同一个 agent 可被不同 Runner 以不同配置运行 |
| 2 | instructions 形式 | 纯字符串 | 字符串 / 函数 / async 函数 | 同一个 agent 面对不同场景动态调整行为 |
| 3 | 工具类型 | 只有自定义函数 | FunctionTool / ComputerTool / FileSearchTool / WebSearchTool / MCP 等 | 覆盖更多工具形态 |
| 4 | 输出控制 | 返回 str，格式不可控 | output_type 约束结构化输出 | LLM 输出格式可预测，可直接作为程序数据使用 |
| 5 | 护栏 | 硬编码在工具里（_safe_path） | input/output guardrails，可插拔 | 安全检查与业务逻辑解耦，可复用 |
| 6 | 多 agent 协作 | 不支持 | handoffs 机制 | agent 间任务交接，关注点分离 |
| 7 | 生命周期观测 | 硬编码 print | hooks 回调系统 | 不改核心代码就能插入监控、日志、审计 |
| 8 | 模型切换 | 硬编码 client | model 字段 + Provider 模式 | 不改 agent 定义就能切换 LLM 后端 |
| 9 | 流式输出 | 不支持 | run_streamed() + stream_events | 用户体验提升，边生成边显示 |
| 10 | name 字段 | 没有 | 有，用于 handoff 路由和 tracing | 多 agent 系统的身份标识 |

---

## 重要概念理解

### name 的实际用途
不只是标签。在 handoff 时自动生成工具名 `transfer_to_{name}`，在 tracing 中标识是哪个 agent 在执行。

### output_type 的依赖
SDK 把 output_type 转成 JSON Schema 发给 LLM，依赖 LLM 的 Structured Output 能力。不是所有后端都支持——我 week1 的中转站（Claude API）就不支持 json_schema strict mode。vLLM 的约束解码可以在解码层面保证输出符合 schema。

### handoff 的本质
handoff = 工具调用。SDK 把目标 agent 包装成一个工具（`transfer_to_XXX`），LLM 调用这个工具就触发交接。传输的是对话历史（HandoffInputData：input_history + pre_handoff_items + new_items），让下一个 agent 知道"之前发生了什么"。input_filter 可以在转交前过滤敏感或无关内容。

### hooks 的执行时机
```
on_start()        ← agent 开始
├→ on_llm_start() ← 调 LLM 前
├→ on_llm_end()   ← LLM 返回后
├→ on_tool_start() ← 工具执行前
├→ on_tool_end()   ← 工具执行后
├→ on_handoff()    ← 交接时
└→ on_end()        ← agent 结束
```

两套 hooks：AgentHooks（挂在单个 agent 上）和 RunHooks（挂在整个 run 上，通过 RunConfig 传入）。

### model 的两种传法
- 传字符串 `model="gpt-4o-mini"`：Runner 通过 ModelProvider 查找对应实现
- 传 Model 对象：跳过查找，直接使用（可以给不同 agent 指定不同后端）

### Pydantic Model 和 Field
- Pydantic Model = 带类型验证 + JSON Schema 生成 + 序列化的数据结构（类似 dataclass 的加强版）
- Field = 给字段加元信息（description、default 等），描述会进入 JSON Schema，LLM 能看到

---

## 疑问

1. guardrails 具体怎么中断运行？抛异常？返回特殊值？（Day 5 读 guardrail.py 时解答）
2. Runner 循环内部，guardrails / hooks / tracing 分别在什么精确位置被调用？（Day 3-4 读 run.py 时解答）
3. 流式和非流式路径共享多少核心逻辑？（Day 3-4 解答）
4. handoff 时如果目标 agent 的 model 和当前 agent 不同，怎么处理？（Day 5 解答）
