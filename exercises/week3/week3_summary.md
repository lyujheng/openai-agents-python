# Week 3 总结：OpenAI Agents SDK 源码精读

## 学习成果

用 5 天时间，从示例代码到核心运行循环，完整读通了 SDK 的架构。

---

## 五层架构全景

```
第五层：Model Provider（后端可替换）
  └── Model 接口 + ModelProvider 接口 → 接入任意 LLM 后端

第四层：Handoff + Guardrails + Tracing（生产级辅助能力）
  ├── Handoff：多 agent 串行接力，通过工具调用实现
  ├── Guardrails：四个检查点（input/output × agent/tool）
  └── Tracing：三层架构（SpanData → Span → Processor）

第三层：运行循环（核心骨架）
  ├── Runner.run() → _run_impl() → while True 主循环
  ├── run_single_turn()：guard → hooks → 收集工具 → 调 LLM → 处理响应
  └── turn_resolution：解析响应，判断下一步（结束/handoff/继续/中断）

第二层：工具系统（LLM 的能力扩展）
  ├── Tool Union（13 种工具类型）
  ├── FunctionTool + function_schema（Pydantic 驱动的 schema 生成）
  └── ToolContext（运行时上下文注入）

第一层：Agent 定义（声明式配置）
  └── Agent 类：name, instructions, tools, handoffs, guardrails, hooks, model, output_type
```

---

## 核心认知

### 1. 所有 agent 框架的本质是同一个 while 循环

```python
while True:
    response = call_llm(messages, tools)
    if 最终输出:
        break
    elif 工具调用:
        执行工具，结果放回
        continue
```

不管 SDK 有多少代码，骨架和 mini_agent 一样。差异全在循环周围。

### 2. SDK 比 mini_agent 多的是生产级能力

| 维度 | mini_agent（150 行） | SDK（数千行） |
|------|---------------------|-------------|
| 主循环 | 相同骨架 | 相同骨架 |
| 并发安全 | 无（单用户） | guard 快照检查 + is_enabled 过滤 |
| 可观测性 | print | Hooks（两级回调）+ Tracing（自动链路记录） |
| 断点恢复 | 无 | RunState 序列化/恢复 |
| 多 agent | 无 | Handoff 串行接力 |
| 安全防护 | 无 | Guardrails（4 个检查点，可并行/串行） |
| 工具类型 | 1 种（函数） | 13 种（函数/搜索/代码执行/电脑操作...） |
| Schema 生成 | 手写类型映射 | Pydantic 驱动，支持复杂类型 |
| 数据记录 | 原始字典 | 类型化 Item 体系 |
| 结果包装 | 返回字符串 | RunResult（items + usage + guardrail 结果） |
| 后端切换 | 硬编码 | Model/ModelProvider 抽象接口 |

### 3. 多 agent 的本质是调度层 + 多个 while 循环

```
单 agent = while 循环
多 agent = 调度层 + 多个 while 循环
```

这个 SDK 用 handoff（串行接力），其他框架有并行协作（AutoGen）、层级调度（CrewAI）等模式。调度方式不同，但每个 agent 内部都是同一个循环。

### 4. Handoff 在 LLM 眼里就是工具调用

SDK 把可转交的 agent 包装成工具（`transfer_to_{name}`），LLM 根据 instructions 和对话内容自行决定是否调用。流程控制交给了 LLM，不是硬编码的 if-else。

### 5. Hooks 和 Tracing 是两套独立机制

- Hooks：你写的回调，你决定做什么（报警、计费、审计）
- Tracing：SDK 自动记录，固定的（耗时、token、调用链路）

### 6. Model Provider 抽象让后端可替换

实现 `get_response()` 和 `stream_response()` 两个方法即可接入任意 LLM 后端。vLLM 兼容 OpenAI 格式，改 base_url 就行，不需要重写。

---

## 关键代码文件索引

| 文件 | 核心内容 |
|------|---------|
| src/agents/agent.py | Agent 类定义，所有字段 |
| src/agents/tool.py | Tool Union、FunctionTool、@function_tool 装饰器 |
| src/agents/function_schema.py | Pydantic 驱动的 JSON Schema 生成 |
| src/agents/run.py | Runner 入口、主循环 |
| src/agents/run_internal/run_loop.py | run_single_turn 完整流程 |
| src/agents/run_internal/turn_resolution.py | 响应解析、NextStep 判断 |
| src/agents/run_internal/agent_tool_configuration.py | 并发安全 guard |
| src/agents/items.py | RunItem 数据类型体系 |
| src/agents/result.py | RunResult 结果包装 |
| src/agents/handoffs/__init__.py | Handoff 定义、HandoffInputData |
| src/agents/guardrail.py | InputGuardrail、OutputGuardrail |
| src/agents/lifecycle.py | RunHooks、AgentHooks |
| src/agents/tracing/span_data.py | Span 数据类型 |
| src/agents/models/interface.py | Model、ModelProvider 抽象接口 |
| src/agents/models/openai_responses.py | OpenAI Responses API 实现 |

---

## 为后续学习建立的基础

| 后续周次 | 需要的 Week 3 知识 |
|---------|-------------------|
| Week 4（Prompt Engineering） | instructions 的多种形式、output_type 结构化输出 |
| Week 5（RAG） | Tool 系统、FileSearchTool |
| Week 6（多 Agent） | Handoff 机制、HandoffInputData、input_filter |
| Week 7（vLLM 接入） | Model/ModelProvider 接口、base_url 替换 |
| Week 8（生产部署） | Hooks、Tracing、Guardrails、并发安全 |

---

## 遗留疑问（后续学习解答）

1. 流式 run_streamed 和非流式 run 在主循环层面的具体差异
2. Tracing Processor 如何对接第三方监控平台
3. 接入 vLLM 时除了 base_url 还需要注意什么
4. Session 持久化的具体实现细节
5. Sandbox 沙箱执行的应用场景和实现
