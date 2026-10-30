# Day 5 笔记：Handoff、Guardrails、Tracing、Model Provider

## 核心收获

Day 5 覆盖了 SDK 的四个辅助子系统。它们不是主循环的骨架，而是围绕骨架的**生产级能力**：多 agent 协作、安全防护、可观测性、后端可替换。

---

## Handoff：多 Agent 协作

### HandoffInputData——交接数据包

handoff 发生时，SDK 把 Agent A 的历史打包传给 Agent B：

```
用户输入 → Agent A 第1轮（调工具）→ Agent A 第2轮（调工具）→ Agent A 第3轮（决定 handoff）
           ↑                                                    ↑                    ↑
       input_history                                   pre_handoff_items         new_items
```

| 字段 | 内容 |
|------|------|
| input_history | 最原始的用户输入 |
| pre_handoff_items | Agent A 在 handoff 之前的所有轮次产生的 items |
| new_items | 触发 handoff 那一轮产生的 items |
| input_items | 过滤后的子集，实际喂给 Agent B（None 时默认用 new_items） |

pre_handoff_items + new_items = 完整历史（存档用），input_items = 过滤后（输入用）。

### input_filter——过滤交接内容

SDK 不替你决定传什么给下一个 agent，通过 input_filter 回调让你自己控制：

```python
def my_filter(handoff_data: HandoffInputData) -> HandoffInputData:
    # 只保留文本消息，去掉工具调用细节
    filtered = tuple(item for item in handoff_data.new_items if isinstance(item, MessageOutputItem))
    return HandoffInputData(..., input_items=filtered)

handoff(agent=翻译agent, input_filter=my_filter)
```

### handoff() 函数的三种使用方式

```python
# 1. 最简单——直接转交
handoff(agent=翻译agent)

# 2. 带回调 + 带输入类型——LLM 转交时需要提供额外参数
handoff(agent=翻译agent, on_handoff=my_callback, input_type=TransferInfo)

# 3. 带回调、不带输入——转交时做副作用（记日志等），不需要 LLM 传参
handoff(agent=翻译agent, on_handoff=log_handoff)
```

三个 @overload 不是真正的重载，是给 IDE 的类型提示，运行时只有一个函数。

### Handoff 触发机制

Handoff 不是硬编码的 if-else，是 LLM 自己决定的。SDK 把可转交的 agent 包装成工具发给 LLM（如 `transfer_to_翻译`），LLM 根据 instructions 和对话内容推理是否调用。

---

## Guardrails：安全防护

### 四个检查点

```
用户输入 → [InputGuardrail] → LLM → [OutputGuardrail] → 最终输出
                                ↓
                           工具调用
                   [ToolInputGuardrail] → 执行工具 → [ToolOutputGuardrail]
```

### InputGuardrail 核心字段

| 字段 | 作用 |
|------|------|
| guardrail_function | 你写的检查函数，接收 (ctx, agent, input)，返回是否触发 |
| name | 护栏名称，用于 tracing 和日志 |
| run_in_parallel | True=和 LLM 并行检查（快），False=先检查再调 LLM（严格拦截） |

### 使用示例

```python
async def check_no_id_number(ctx, agent, output):
    if re.search(r'\d{18}', str(output)):
        return GuardrailFunctionOutput(tripwire_triggered=True, message="输出包含身份证号")
    return GuardrailFunctionOutput(tripwire_triggered=False)

agent = Agent(name="客服", output_guardrails=[OutputGuardrail(guardrail_function=check_no_id_number)])
```

### 为什么区分 Input/Output Guardrail

因为检查的数据类型不同：输入是字符串或消息列表，输出可能是 Pydantic 结构化对象。分开后类型清晰，IDE 能给正确的类型提示。

---

## Tracing：可观测性

### 与 Hooks 的区别

| | Hooks | Tracing |
|--|-------|---------|
| 谁控制 | 你写回调代码 | SDK 自动记录 |
| 记什么 | 你决定（业务逻辑） | 固定的（耗时、token、输入输出） |
| 用途 | 报警、计费、审计 | 调用链路可视化、性能分析 |

两套独立机制，SDK 在同一个节点既触发 hook 回调，也记录 span。

### 三层架构

**第一层：SpanData（数据）** —— span_data.py

每种 Span 定义要记录什么字段：

| SpanData 类型 | 记录什么 |
|--------------|---------|
| AgentSpanData | agent 名字、工具列表、handoff 列表 |
| GenerationSpanData | LLM 模型名、token 用量、输入输出 |
| FunctionSpanData | 工具名、输入参数、返回结果 |
| HandoffSpanData | 从谁转给谁 |
| GuardrailSpanData | 护栏检查结果 |

区分多种类型是因为不同事件要记录的字段不同，统一用一个类会导致大量 Optional 字段。

**第二层：Span（时间线）** —— spans.py

给数据加上开始时间、结束时间、父子关系。

**第三层：Processor（导出）** —— processors.py

Span 结束后发到监控平台（OpenAI dashboard、Jaeger、Datadog 等）。

### Span 层级结构

```
Trace（一次完整的 Runner.run）
├── AgentSpan（Agent A 执行）
│   ├── GenerationSpan（LLM 调用，耗时、tokens）
│   ├── FunctionSpan（工具调用）
│   ├── GenerationSpan（第二次 LLM 调用）
│   └── HandoffSpan（转交）
└── AgentSpan（Agent B 执行）
    ├── GenerationSpan（LLM 调用）
    └── GuardrailSpan（护栏检查）
```

### Tracing 数据来源

全部在 SDK 本地记录，不是推理端传回的。SDK 在发请求前记开始时间，收到响应后记结束时间，token 用量从响应里读取。

### 自己实现 Tracing 的核心

```python
# 1. 定义数据结构
class MySpan:
    name, type, start_time, end_time, data, children

# 2. 在关键节点埋点（写在 run 方法内）
span.start()
response = await call_llm(...)
span.data["tokens"] = response.usage.total_tokens
span.finish()

# 3. 导出
export(trace)  # 发到监控平台
```

---

## Model Provider：后端可替换

### 两个核心接口（interface.py）

**Model** — LLM 调用的抽象：
- `get_response()` — 非流式：组装请求 → 发 HTTP → 解析响应
- `stream_response()` — 流式版本

**ModelProvider** — 模型查找：
- `get_model(name)` — 根据模型名返回 Model 实例

### 调用链

```
主循环 → get_new_response() → model.get_response(tools=all_tools, ...)
```

对应 mini_agent 里的：
```python
response = client.chat.completions.create(messages=messages, tools=tools)
```

### get_response 内部做了什么

1. 准备输入——把 SDK 参数组装成 HTTP 请求的 JSON 格式（_fetch_response）
2. 发请求——调推理服务
3. 解析响应——提取 token 用量等信息
4. tracing 记录——把输入输出记到 span
5. 错误处理——超时、异常

### 接入 vLLM 的方式

vLLM 兼容 OpenAI API 格式，**不需要重写 Model**，改 base_url 就行：

```python
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="none")
agent = Agent(name="客服", model=OpenAIResponsesModel(model="模型名", openai_client=client))
```

只有推理框架的 API 格式不兼容 OpenAI 时才需要重写 Model 接口。

### 所谓的 API 格式

就是推理服务端的 HTTP 请求/响应 JSON 结构。不同框架字段名、嵌套结构可能不同。OpenAI 格式已成行业标准，大部分框架（vLLM、Ollama、LiteLLM）都兼容。

---

## 疑问记录

1. stream_response 和 get_response 在实现层面具体差异多大？
2. Tracing 的 Processor 如何对接第三方监控平台？
3. 第 7 周接入 vLLM 时，除了改 base_url，还有哪些需要注意的？
