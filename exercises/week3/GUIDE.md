# 第3周：OpenAI Agents SDK 源码精读

> 目标：精读本仓库核心源码，理解生产级 agent 框架在最小循环之上做了哪些工程决策。
> 核心问题：从你的 150 行 mini_agent 到一个生产级框架，中间差了哪些工程决策？每个决策解决了什么真实问题？

---

## SDK 整体架构

从 `src/agents/__init__.py` 的导出结构，可以看出 SDK 由以下核心子系统组成：

```
┌─────────────────────────────────────────────────────┐
│                    用户代码                           │
│         Agent(...) + Runner.run(agent, input)        │
└──────────────┬──────────────────────┬────────────────┘
               │                      │
    ┌──────────▼──────────┐  ┌───────▼────────┐
    │   Agent 定义层       │  │   Runner 运行层  │
    │ agent.py             │  │ run.py          │
    │ - instructions       │  │ - 主循环编排     │
    │ - tools              │  │ - turn 计数     │
    │ - handoffs           │  │ - 终止判断      │
    │ - guardrails         │  │ run_config.py   │
    │ - output_type        │  │ run_context.py  │
    └──────────┬──────────┘  └───┬────┬───┬────┘
               │                 │    │   │
   ┌───────────▼───┐   ┌────────▼┐ ┌▼───▼──────────┐
   │  Tool 系统     │   │ Model   │ │  Items/Result  │
   │ tool.py        │   │ 抽象层  │ │ items.py       │
   │ function_schema│   │interface│ │ result.py      │
   │ decorators.py  │   │ openai_ │ │ stream_events  │
   └───────┬───────┘   │ *.py    │ └────────────────┘
           │            └────┬────┘
   ┌───────▼───────┐        │
   │  MCP 子系统    │   ┌────▼──────────────┐
   │  mcp/          │   │  支撑子系统         │
   └───────────────┘   │ tracing/  guardrail │
                        │ handoffs/ memory/   │
                        │ lifecycle  usage    │
                        └────────────────────┘
```

**核心关系：Agent 是声明（"我是谁、我有什么能力"），Runner 是执行（"怎么跑循环"），Model 是推理（"调谁的 LLM"），Tool 是行动（"能做什么"）。**

对比你第2周手搓的 `MiniAgent`——它把这四件事全揉在一个类里。SDK 的核心设计决策就是把它们拆开。

### 子系统对照你的 mini_agent

| SDK 子系统 | 对应你的 mini_agent | SDK 多了什么 |
|-----------|-------------------|------------|
| Agent 定义层 | `self.system_prompt` + `self.tools` | handoffs、guardrails、output_type、动态 instructions、hooks |
| Runner 运行层 | `async def run()` 里的 while 循环 | 独立的 Runner 类、RunConfig、流式/非流式双路径、session 持久化 |
| Tool 系统 | `_build_tool_schemas()` + `_execute_tool()` | 丰富的工具类型（Function/Computer/Shell/MCP/FileSearch/CodeInterpreter...） |
| Model 抽象层 | 硬编码 `self.client = OpenAI(...)` | Model 接口 + Provider 模式，可插拔不同后端 |
| Items/Result | 直接返回 `message.content` | 结构化的 RunItem 类型（ToolCallItem、HandoffCallItem、ReasoningItem...） |
| Tracing | 无 | 完整的 span 层级追踪系统 |
| Guardrails | 无 | 输入/输出护栏，可中断运行 |
| Handoffs | 无 | agent 间任务交接机制 |

---

## 阅读路线（按依赖顺序，5 天）

### 第一层：入口和基本概念（Day 1）

#### 先读示例，建立直觉

| 文件 | 行数 | 看什么 |
|------|------|--------|
| `examples/basic/hello_world.py` | 20 | 最简用法：Agent + Runner.run() |
| `examples/basic/tools.py` | 40 | `@function_tool` 装饰器定义工具 |
| `examples/basic/stream_text.py` | 21 | 流式输出：Runner.run_streamed() |
| `examples/basic/dynamic_system_prompt.py` | 70 | instructions 的动态函数形式 |

带着这些问题读：

- [ ] **最小可运行的 agent 需要几行代码？需要哪些 API 调用？**
- [ ] **`@function_tool` 做了什么？和你手写的 `_build_tool_schemas()` 对应哪个环节？**
- [ ] **`Runner.run()` vs `Runner.run_streamed()` 调用方式有什么区别？返回值有什么区别？**
- [ ] **dynamic_system_prompt 示例中，instructions 是怎么变成函数的？这解决了什么问题？**

#### 再读 Agent 类定义

| 文件 | 行数 | 看什么 |
|------|------|--------|
| `src/agents/agent.py` | 1170 | Agent 类的所有字段和方法 |

这个文件比较长，不需要逐行读。聚焦 `Agent` 类的字段定义部分（找 `class Agent` 开头的 dataclass），带着这个核心问题：

> **Agent 类的每个字段，对应 agent 循环（感知→推理→行动→观察）的哪个环节？哪些字段是你的 mini_agent 没有的？这些"多出来的"字段各自解决了什么问题？**

按功能分组关注：

| 分组 | 要找的字段 | 你的 mini_agent 有吗？ |
|------|----------|---------------------|
| 身份 | `name`, `instructions` | ✅ 有 system_prompt |
| 工具 | `tools`, `mcp_servers` | ✅ 有 tools dict |
| 输出控制 | `output_type`, `stop_at_tools` | ❌ 没有 |
| 护栏 | `input_guardrails`, `output_guardrails` | ❌ 没有 |
| 交接 | `handoffs`, `handoff_description` | ❌ 没有 |
| 生命周期 | `hooks` | ❌ 没有 |
| 模型配置 | `model`, `model_settings` | ❌ 硬编码 |

特别留意：
1. `instructions` 字段支持哪几种形式？（静态字符串？动态函数？还有别的吗？）
2. `output_type` 是怎么约束 agent 的最终输出的？
3. `handoffs` 字段的类型是什么？暗示了 handoff 的本质是什么？

#### Day 1 检查点

- [ ] 能说出 SDK 的最小使用模式：`Agent(name=..., instructions=...) + Runner.run(agent, input)`
- [ ] 能列出 Agent 类的主要字段分组，理解每个分组对应 agent 循环的哪个环节
- [ ] 理解 instructions 的多种形式（字符串 / 函数 / ...），知道动态 instructions 解决了什么问题
- [ ] 发现了至少 3 个你的 mini_agent 没有、但 SDK 有的字段，能说出它们各自解决的问题

#### Day 1 笔记模板

在 `exercises/week3/` 下创建 `day1_notes.md`，记录：

```markdown
## Day 1 笔记

### 示例观察
- 最小 agent 代码：（写出关键几行）
- @function_tool 做了什么：
- run() vs run_streamed() 的区别：

### Agent 类字段分析
（列出你发现的字段，按分组整理，标注哪些是 mini_agent 没有的）

### instructions 的多种形式
1. ...
2. ...

### 和 mini_agent 的关键差异
| 差异点 | mini_agent | SDK Agent | 解决了什么问题 |
|--------|-----------|-----------|--------------|
| ... | ... | ... | ... |

### 疑问
（记录读代码时产生的疑问，后续几天可能会解答）
```

---

### 第二层：工具系统（Day 2）

#### 阅读文件

| 文件 | 行数 | 看什么 |
|------|------|--------|
| `src/agents/tool.py` | 3015 | Tool 基类、FunctionTool、各种工具类型 |
| `src/agents/function_schema.py` | 557 | 从 Python 函数签名自动生成 JSON Schema |
| `src/agents/decorators.py` | 19 | `@tool` 装饰器（注意：只有 19 行，它是入口） |
| `src/agents/tool_context.py` | - | 工具执行时的上下文 |

#### 阅读策略

`tool.py` 有 3015 行，不要从头读到尾。建议：

1. **先读 `Tool` 基类**：找 `class Tool`，看它定义了什么接口（哪些方法是子类必须实现的）
2. **再读 `FunctionTool`**：这是最常用的工具类型，对应 `@function_tool` 装饰器创建的工具
3. **浏览其他工具类型**：`ComputerTool`、`FileSearchTool`、`CodeInterpreterTool`、`WebSearchTool` 等，了解 SDK 支持哪些工具形态，不需要细读
4. **读 `function_schema.py`**：这是自动 schema 生成的核心，对比你手写的版本

#### 带着这些问题读

- [ ] **`Tool` 基类定义了什么接口？子类需要实现哪些方法？**
- [ ] **`FunctionTool` 怎么把一个 Python 函数包装成工具？schema 是什么时候生成的？**
- [ ] **和你第2周的 `_build_tool_schemas()` 对比，SDK 怎么处理复杂类型？**
  - 你的版本：用 `inspect` 模块读参数名和类型注解
  - SDK 的版本：看 `function_schema.py` 怎么处理 Pydantic model、`Annotated`、`Field` 等
- [ ] **工具执行失败时 SDK 怎么处理？（看 `tool_context.py` 和 `tool.py` 中的错误处理逻辑）**
- [ ] **`decorators.py` 只有 19 行，它做了什么？实际的装饰器逻辑在哪里？**

#### 对比实验

打开你的 `exercises/week2/mini_agent.py`，找到 `_build_tool_schemas()` 方法，和 `function_schema.py` 放在一起对比：

| 维度 | 你的实现 | SDK 实现 |
|------|---------|---------|
| 类型映射 | `str→string, int→integer, ...` | ？（看 SDK 怎么做的） |
| 复杂类型 | 不支持 | ？ |
| 参数描述 | 从 docstring 解析？ | ？ |
| 必填/可选 | ？ | ？ |
| 默认值处理 | ？ | ？ |

#### Day 2 检查点

- [ ] 能画出 Tool 类型继承树：`Tool` → `FunctionTool` / `ComputerTool` / `FileSearchTool` / ...
- [ ] 理解 `function_schema.py` 的 schema 生成逻辑，能说出它比你的版本多处理了哪些情况
- [ ] 知道 `@function_tool` / `@tool` 装饰器的完整调用链
- [ ] 理解工具执行失败时的错误处理流程

---

### 第三层：运行循环（Day 3-4）—— 最核心

这是整个源码精读最重要的部分。

#### 阅读文件

| 文件 | 行数 | 看什么 |
|------|------|--------|
| `src/agents/run.py` | 2827 | Runner 类：`run()` 和 `run_streamed()` 的主循环 |
| `src/agents/run_config.py` | 655 | RunConfig：运行时配置 |
| `src/agents/items.py` | 1034 | RunItem：每一步产生的数据项 |
| `src/agents/result.py` | 1324 | RunResult / RunResultStreaming：最终结果 |
| `src/agents/run_context.py` | 1717 | RunContextWrapper：跨轮共享的上下文 |

同时注意 `src/agents/run_internal/` 目录——Runner 的内部实现被拆分到了这里：

| 文件 | 职责 |
|------|------|
| `run_internal/run_loop.py` | 主循环实现 |
| `run_internal/turn_resolution.py` | 一轮结束时的判断逻辑 |
| `run_internal/tool_execution.py` | 工具执行编排 |
| `run_internal/turn_preparation.py` | 每轮开始前的准备 |
| `run_internal/streaming.py` | 流式处理路径 |
| `run_internal/session_persistence.py` | 会话持久化 |
| `run_internal/guardrails.py` | 护栏执行 |

#### 阅读策略

`run.py` 有 2827 行，建议这样读：

1. **Day 3 上午**：先读 `Runner` 类的公共方法签名（`run()`, `run_streamed()`），理解入口
2. **Day 3 下午**：跟踪 `run()` 的执行流程，它会调用 `run_internal/` 的哪些模块？按调用顺序走一遍
3. **Day 4 上午**：精读 `run_internal/run_loop.py` 和 `turn_resolution.py`——这是循环的心脏
4. **Day 4 下午**：读 `items.py` 和 `result.py`，理解循环每一步产生什么数据、最终结果长什么样

#### 带着这些问题读

- [ ] **Runner 的主循环和你手搓的 while 循环在结构上有什么区别？**
  - 你的循环：`for turn in range(max_turns): response = call_llm(); if tool_calls: execute(); else: return`
  - SDK 的循环：找到对应的结构，画出流程
- [ ] **一轮（turn）是怎么定义的？什么算一轮结束？**
  - 提示：看 `turn_resolution.py`，关注 handoff 和 final_output 的判断逻辑
- [ ] **max_turns 是怎么计的？和你的 max_turns 有什么区别？**
- [ ] **流式和非流式路径的行为一致性怎么保证？**
  - 提示：看 `run()` 和 `run_streamed()` 是否共享核心逻辑
- [ ] **RunConfig 有哪些配置项？哪些是你的 mini_agent 用硬编码解决的？**
- [ ] **RunItem 有哪些类型？对应 agent 循环的哪些事件？**
  - 你的 mini_agent 只区分"工具调用"和"最终回答"
  - SDK 区分了 ToolCallItem、HandoffCallItem、MessageOutputItem、ReasoningItem 等

#### 对比你的 run() 方法

打开 `exercises/week2/mini_agent.py`，找到你的 `run()` 方法，逐行和 SDK 的循环对比：

```
你的 run():                          SDK Runner.run():
─────────────                        ──────────────────
1. append user message               1. ?
2. for turn in range(max_turns):     2. ?
3.   call_llm()                      3. ?
4.   if tool_calls:                  4. ?
5.     execute_tool()                5. ?
6.     append result                 6. ?
7.   else:                           7. ?
8.     return content                8. ?
9. return "达到最大轮次"              9. ?
```

填写 SDK 每一步对应做了什么，以及 SDK 在你的步骤之间额外插入了哪些逻辑（guardrails? hooks? tracing? handoff 检测?）。

#### Day 3-4 检查点

- [ ] 能画出 Runner.run() 的完整流程图（从用户输入到最终返回）
- [ ] 能指出 SDK 循环和你的 while 循环的至少 5 个结构性区别
- [ ] 理解 turn 的定义和终止条件
- [ ] 知道 RunItem 的主要类型及其含义
- [ ] 理解 RunConfig 提供了哪些可配置项

---

### 第四层：Handoff、护栏和 Tracing（Day 5 上午）

#### 阅读文件

| 文件 | 行数 | 看什么 |
|------|------|--------|
| `src/agents/handoffs/__init__.py` | 395 | handoff 的定义和创建 |
| `examples/handoffs/message_filter.py` | 194 | handoff 时怎么过滤传递的消息 |
| `src/agents/guardrail.py` | 343 | 输入/输出护栏 |
| `src/agents/lifecycle.py` | 233 | AgentHooks / RunHooks |

Tracing 子系统（概览级别）：

| 文件 | 行数 | 看什么 |
|------|------|--------|
| `src/agents/tracing/traces.py` | 575 | Trace 的创建和管理 |
| `src/agents/tracing/spans.py` | 423 | Span 类型 |
| `src/agents/tracing/span_data.py` | 452 | 各种 Span 数据：agent / generation / function / handoff / guardrail |
| `src/agents/tracing/processors.py` | 922 | TraceProcessor 接口和实现 |

#### 带着这些问题读

**Handoff：**
- [ ] **handoff 的本质是什么？**（提示：它和工具调用有什么关系？看 handoff 在 Runner 循环中是怎么被检测和执行的）
- [ ] **handoff 时消息历史怎么处理？为什么需要 message_filter？**
- [ ] **多 agent 协作的"交接"在代码层面是怎么实现的？是创建新 Runner 还是切换 Agent？**

**Guardrails：**
- [ ] **输入护栏（InputGuardrail）在循环的哪个位置执行？能中断运行吗？怎么中断？**
- [ ] **输出护栏（OutputGuardrail）在什么时候执行？**
- [ ] **护栏和你第2周的安全边界设计（`_safe_path`、危险命令拦截）有什么本质区别？**

**Tracing：**
- [ ] **Trace 和 Span 是什么关系？Span 的层级结构是怎样的？**
- [ ] **有哪些类型的 Span？（看 `span_data.py`）每种 Span 对应 agent 循环的哪个事件？**
- [ ] **TraceProcessor 是什么？SDK 默认把 trace 数据发到哪里？**

**Lifecycle Hooks：**
- [ ] **`AgentHooks` vs `RunHooks` 的区别是什么？**
- [ ] **hooks 在循环的哪些点被调用？**

#### Day 5 上午检查点

- [ ] 能用一句话解释 handoff 的本质
- [ ] 能说出护栏在 Runner 循环中的执行位置
- [ ] 能画出 tracing 的 span 层级结构（一个典型的多轮工具调用产生哪些 span）

---

### 第五层：Model Provider 抽象（Day 5 下午）

这一层直接关系到第7周的 vLLM 接入。

#### 阅读文件

| 文件 | 行数 | 看什么 |
|------|------|--------|
| `src/agents/models/interface.py` | 161 | `Model` 抽象基类和 `ModelProvider` 接口 |
| `src/agents/models/openai_chatcompletions.py` | 789 | Chat Completions API 的具体实现 |
| `examples/model_providers/custom_example_provider.py` | 77 | 自定义 provider 的示例 |
| `src/agents/model_settings.py` | - | ModelSettings 的定义 |

#### 带着这些问题读

- [ ] **`Model` 接口需要实现哪些方法？**（提示：`get_response()` 和 `stream_response()`）
- [ ] **`ModelProvider.get_model()` 怎么把 model name 映射到具体实现？**
- [ ] **`OpenAIChatCompletionsModel` 怎么把 SDK 内部格式转成 API 请求？**
  - 输入转换：SDK 的内部消息格式 → Chat Completions API 的 messages 格式
  - 输出转换：API 响应 → SDK 的 ModelResponse 格式
- [ ] **如果要接入 vLLM（兼容 OpenAI API），需要改什么？不需要改什么？**
  - 提示：看 `custom_example_provider.py` 示例，它是怎么指向自定义 API 的

#### 对比你的硬编码

你的 mini_agent 中的 LLM 调用：
```python
# 你的代码（大概是这样）
self.client = OpenAI(base_url="...", api_key="...")
response = self.client.chat.completions.create(model="...", messages=..., tools=...)
```

SDK 的做法：
```
Runner → RunConfig.model_provider → ModelProvider.get_model() → Model.get_response()
```

思考：**这层抽象带来了什么好处？有什么代价？**

#### Day 5 下午检查点

- [ ] 能说出 `Model` 接口的两个核心方法
- [ ] 理解 Provider 模式：为什么不直接在 Agent 里硬编码 LLM 客户端
- [ ] 知道接入 vLLM 需要做什么（为第7周做准备）
- [ ] 看完 `custom_example_provider.py`，知道自定义 provider 的最小实现

---

## 每日时间分配建议

| 时间段 | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 |
|--------|-------|-------|-------|-------|-------|
| 上午 2h | 读示例 + Agent 字段 | Tool 基类 + FunctionTool | Runner 入口 + 调用链 | run_loop + turn_resolution | Handoff + Guardrail + Tracing |
| 下午 2h | 填笔记 + 疑问 | function_schema 对比 | 跟踪 run_internal/ | items + result | Model Provider + 周总结 |

---

## 第3周总体检查点

完成整周后，你应该能回答以下问题：

- [ ] 能画出 SDK 的核心类关系图：`Agent → Runner → Model → Tool → RunItem → RunResult`
- [ ] 能用自己的话解释 `run.py` 主循环的每个阶段
- [ ] 能指出 SDK 和你的 mini_agent 之间至少 10 个结构性差异，并说出每个差异解决了什么问题
- [ ] 理解 handoff 机制和单 agent 循环的区别
- [ ] 清楚 `Model` 接口需要实现什么，为第7周 vLLM 接入做好准备
- [ ] 理解 tracing 子系统的 span 层级和 processor 模型

### 周总结模板

在 `exercises/week3/` 下创建 `week3_summary.md`：

```markdown
## 第3周总结

### SDK 核心类关系图
（画出你理解的类关系图）

### Runner 主循环流程
（用你自己的话描述）

### 和 mini_agent 的 10 个结构性差异
| # | 差异 | mini_agent | SDK | 解决的问题 |
|---|------|-----------|-----|-----------|
| 1 | ... | ... | ... | ... |

### Handoff 的本质
（一段话解释）

### Model 接口设计
（为第7周 vLLM 接入准备的笔记）

### Tracing Span 层级
（画出典型场景的 span 树）

### 遗留疑问
（列出还没搞明白的问题，后续几周可能解答）
```

---

## 源码文件快速索引

方便你在阅读过程中快速定位文件：

### 核心文件

| 文件路径 | 行数 | 职责 |
|---------|------|------|
| `src/agents/agent.py` | 1170 | Agent 类定义 |
| `src/agents/run.py` | 2827 | Runner 入口 |
| `src/agents/tool.py` | 3015 | 工具类型体系 |
| `src/agents/function_schema.py` | 557 | 函数签名 → JSON Schema |
| `src/agents/items.py` | 1034 | RunItem 类型 |
| `src/agents/result.py` | 1324 | RunResult |
| `src/agents/run_config.py` | 655 | RunConfig |
| `src/agents/run_context.py` | 1717 | RunContextWrapper |

### Runner 内部实现

| 文件路径 | 职责 |
|---------|------|
| `src/agents/run_internal/run_loop.py` | 主循环 |
| `src/agents/run_internal/turn_resolution.py` | 轮次终止判断 |
| `src/agents/run_internal/tool_execution.py` | 工具执行编排 |
| `src/agents/run_internal/turn_preparation.py` | 轮前准备 |
| `src/agents/run_internal/streaming.py` | 流式路径 |
| `src/agents/run_internal/guardrails.py` | 护栏执行 |
| `src/agents/run_internal/session_persistence.py` | 会话持久化 |

### 支撑子系统

| 文件路径 | 行数 | 职责 |
|---------|------|------|
| `src/agents/guardrail.py` | 343 | 输入/输出护栏定义 |
| `src/agents/handoffs/__init__.py` | 395 | Handoff 定义 |
| `src/agents/lifecycle.py` | 233 | AgentHooks / RunHooks |
| `src/agents/models/interface.py` | 161 | Model / ModelProvider 接口 |
| `src/agents/models/openai_chatcompletions.py` | 789 | Chat Completions 实现 |
| `src/agents/tracing/traces.py` | 575 | Trace 管理 |
| `src/agents/tracing/spans.py` | 423 | Span 实现 |
| `src/agents/tracing/span_data.py` | 452 | Span 数据类型 |
| `src/agents/tracing/processors.py` | 922 | Trace 导出 |

### 参考示例

| 文件路径 | 行数 | 用途 |
|---------|------|------|
| `examples/basic/hello_world.py` | 20 | 最简 agent |
| `examples/basic/tools.py` | 40 | 工具定义 |
| `examples/basic/stream_text.py` | 21 | 流式输出 |
| `examples/basic/dynamic_system_prompt.py` | 70 | 动态 instructions |
| `examples/handoffs/message_filter.py` | 194 | Handoff 消息过滤 |
| `examples/model_providers/custom_example_provider.py` | 77 | 自定义 Model Provider |
