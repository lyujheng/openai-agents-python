# LLM Agent 学习计划

> 目标：从零掌握 LLM Agent 的原理与工程实践，成为 Agent 领域专家，能根据不同任务从零搭建 agent，掌握评估、安全、多模态、MCP 等主流方向，并用 vLLM 本地部署。
> 背景：熟悉高性能算子开发、vLLM 推理框架（源码级）、用过 chat completion，未用过 function calling。
> 节奏：每天 3-4 小时，共 10 周。

## 学习进度

| 周 | 主题 | 状态 | 完成日期 |
|----|------|------|---------|
| 第1周 | 原理认知 + Function Calling 基础 | ✅ 已完成 | 2026-10-22 |
| 第2周 | 手搓最小 Agent（不用框架） | ✅ 已完成 | 2026-10-23 |
| 第3周 | OpenAI Agents SDK 源码精读 | ✅ 已完成 | 2025-07-25 |
| 第4周 | 工程问题攻克（错误处理、context 管理、安全对抗） | 🔜 下一步 | - |
| 第5周 | MCP 协议与工具生态 | ⬜ | - |
| 第6周 | 评估体系与可观测性 | ⬜ | - |
| 第7周 | vLLM 本地推理后端接入 | ⬜ | - |
| 第8周 | 高级规划与 Agentic RAG | ⬜ | - |
| 第9周 | 多模态与 Computer Use Agent | ⬜ | - |
| 第10周 | 综合实战 + 生产部署 | ⬜ | - |

### 第1周学习调整记录

- 原计划前 2 天纯阅读，实际调整为：读完 Anthropic + Lilian Weng 两篇文章后发现缺乏体感，提前进入动手练习
- 阅读理论后直接动手写代码，再回头对照理论，效果更好
- 练习代码位置：`exercises/week1/`

---

## 总览

第1周  原理认知 + Function Calling 基础
第2周  手搓最小 Agent（不用框架）
第3周  OpenAI Agents SDK 源码精读
第4周  工程问题攻克（错误处理、context 管理、安全对抗）
第5周  MCP 协议与工具生态
第6周  评估体系与可观测性
第7周  vLLM 本地推理后端接入
第8周  高级规划与 Agentic RAG
第9周  多模态与 Computer Use Agent
第10周  综合实战 + 生产部署

---

## 第1周：原理认知 + Function Calling 基础

### 目标

理解 agent 的核心循环（感知→推理→行动→观察），掌握 function calling 的调用方式，建立对 agent 架构从简到繁的全景认知。

### 阅读材料（2 天）

| 材料 | 重点 | 时间 |
|------|------|------|
| [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) (Anthropic) | 什么时候该用 agent、Workflow vs Agent 的边界、架构从简到繁的演进 | 2h |
| [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) (Lilian Weng) | Planning / Memory / Tool Use 三大模块的全景图 | 2h |
| [OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling) | JSON Schema 定义工具、tool_choice 参数、多工具调用 | 1h |
| [A Survey on LLM-based Agents](https://arxiv.org/abs/2308.11432) | agent 架构分类的学术全景（选读摘要和架构章节） | 1h |

### 动手练习（3 天）

#### 练习 1：单轮 function calling

用纯 `openai` Python SDK 完成一次 function calling 调用。理解请求和响应的完整数据结构。

```python
# 目标输出：理解 tool_calls 字段、function.name、function.arguments 的结构
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "东京天气怎么样？"}],
    tools=tools,
    tool_choice="auto"
)

# 打印完整 response，逐字段理解
print(response.choices[0].message)
```

#### 练习 2：多轮 function calling

让 LLM 调用工具后，把工具结果喂回去，让它生成最终回答。完整走通一次"用户提问→LLM 决定调工具→执行工具→LLM 总结回答"。

```python
# 关键点：
# 1. response.choices[0].message.tool_calls 不为空时，说明 LLM 要调工具
# 2. 你需要执行工具，然后把结果以 role="tool" 的消息追加到 messages
# 3. 再次调用 LLM，这次它会用工具结果生成最终回答
```

#### 练习 3：并行工具调用

定义 3 个工具（天气、汇率、新闻），让 LLM 在一次回复中同时调用多个工具。理解 `tool_calls` 是一个列表。

#### 练习 4：Structured Output 初体验

用 `response_format={"type": "json_object"}` 和 `response_format={"type": "json_schema", ...}` 分别约束 LLM 输出格式。理解 Structured Output 和普通 JSON Mode 的区别。

### 检查点

- [x] 能画出 function calling 的完整数据流（request → response with tool_calls → tool execution → follow-up request → final response）  ✅ 练习1-2 完成
- [x] 理解 JSON Schema 怎么描述工具参数  ✅ 练习3 的 tools 定义
- [x] 知道 `tool_choice` 的 `auto` / `required` / `none` 区别  ✅ 练习中使用了 auto
- [x] 理解 Workflow vs Agent 的适用边界（来自 Anthropic 文章）  ✅ 结合动手练习回顾了理解

额外收获：
- [x] 理解 Structured Output 的三种层次（无约束 / JSON Mode / JSON Schema）及实际限制  ✅ 练习4
- [x] 了解到 Structured Output 不是所有模型/中转站都支持，需要后处理清洗 + 重试 + 约束解码等组合方案
- [x] 理解 Chat Completions API 是推理层之上的协议层，tool_calls 是服务端解析模型原始输出后的结构化封装
- [x] 理解 base_url 中 /v1 的作用（API 版本路径）

学习笔记：
- 使用的 API：中转站 zyapi.xmsxb.com/v1，底层模型为 Claude，模型名 mco-4
- 中转站的 JSON Schema strict mode 不支持透传（Claude API 不支持 OpenAI 的 response_format json_schema）
- JSON Mode 输出可能带 markdown 代码块包装，需要清洗

---

## 第2周：手搓最小 Agent（不用框架）

### 目标

用纯 Python + OpenAI API，不依赖任何 agent 框架，实现一个能自主循环调用工具完成任务的 agent。

### Day 1-2：核心循环

从零实现 agent 的主循环，代码量控制在 150 行以内：

```python
# mini_agent.py 核心结构

class MiniAgent:
    def __init__(self, system_prompt: str, tools: dict):
        self.system_prompt = system_prompt
        self.tools = tools  # {"tool_name": callable}
        self.tool_schemas = [...]  # JSON Schema 列表
        self.messages = []
        self.max_turns = 10

    async def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for turn in range(self.max_turns):
            response = await self._call_llm()
            message = response.choices[0].message

            if message.tool_calls:
                # 执行工具，追加结果
                for tool_call in message.tool_calls:
                    result = self._execute_tool(tool_call)
                    self.messages.append(tool_result_message(tool_call, result))
            else:
                # LLM 没有调工具，说明任务完成
                return message.content

        return "达到最大轮次限制"
```

实现的模块：
- `_call_llm()`: 封装 API 调用
- `_execute_tool()`: 解析 function name 和 arguments，路由到对应函数
- `_build_tool_schemas()`: 从 Python 函数自动生成 JSON Schema（用 `inspect` 模块）
- 终止条件判断

### Day 3-4：加入实用工具

给 agent 接入 3-5 个真实工具，让它能完成一个端到端任务：

| 工具 | 功能 | 学习点 |
|------|------|--------|
| `read_file(path)` | 读取本地文件 | 工具的输入输出处理 |
| `write_file(path, content)` | 写入文件 | 有副作用的工具怎么处理 |
| `run_command(cmd)` | 执行 shell 命令 | 安全边界、超时控制 |
| `search_web(query)` | 搜索信息 | 工具返回大量文本时的截断策略 |
| `python_exec(code)` | 执行 Python 代码 | 沙箱隔离的必要性 |

测试任务："帮我分析当前目录下所有 Python 文件的代码行数，按从多到少排序，结果写入 report.txt"

### Day 5：错误处理和重试

- 工具执行出错时，把错误信息喂回 LLM，让它自己修正
- LLM 返回无法解析的 JSON 时的重试逻辑
- 设置超时和最大重试次数

### 检查点

- [x] 不依赖任何框架，agent 能自主循环完成包含多步工具调用的任务  ✅ mini_agent.py 测试2 多步汇率换算
- [x] 工具出错时 agent 能自我纠正（至少重试一次）  ✅ test_error_handling.py 三个错误场景
- [x] 理解为什么需要 max_turns 限制  ✅ 防止 agent 无限循环消耗 token

额外收获：
- [x] 从函数签名 + docstring 自动生成 JSON Schema，理解了工具描述质量对 LLM 填参的影响
- [x] 安全边界设计：路径限制（_safe_path）、危险命令拦截、输出截断；认识到应用层路径检查管不住子进程
- [x] agent 的自主策略选择：list_directory 被拦后自动切换到 run_command；选 wc -l 而非逐个 count_lines
- [x] 错误信息是写给 LLM 看的：措辞直接影响 agent 下一步是重试、换方案还是放弃
- [x] API 调用指数退避重试、JSON 解析容错、参数类型错误的友好提示

学习笔记：
- 练习代码位置：`exercises/week2/`
- mini_agent.py：可复用的 agent 核心（~150 行），包含 schema 自动生成、循环、错误处理
- real_tools_agent.py：接入真实文件系统工具，完成了端到端任务（分析 Python 文件行数并写入 report.txt）
- test_error_handling.py：验证了三类错误场景（间歇性失败、逻辑错误、数据不存在）

---

## 第3周：OpenAI Agents SDK 源码精读

### 目标

精读本仓库核心源码，理解生产级 agent 框架在最小循环之上做了哪些工程决策。同时了解 tracing 和 MCP 子系统。

### 阅读路线（按依赖顺序）

#### 第一层：入口和基本概念（Day 1）

| 文件 | 行数 | 重点 |
|------|------|------|
| `examples/basic/hello_world.py` | 20 | 最简用法，Agent + Runner.run() |
| `examples/basic/tools.py` | 41 | `@tool` 装饰器定义工具，Pydantic 返回值 |
| `examples/basic/stream_text.py` | 22 | 流式输出 |
| `src/agents/agent.py` | ~1170 | Agent 类的所有字段：instructions, tools, handoffs, output_type, guardrails |

带着问题读：
- Agent 类有哪些字段？每个字段对应 agent 循环的哪个环节？
- instructions 支持哪几种形式？（静态字符串 / 动态函数）

#### 第二层：工具系统（Day 2）

| 文件 | 行数 | 重点 |
|------|------|------|
| `src/agents/tool.py` | ~3015 | Tool 基类、FunctionTool、工具 schema 生成 |
| `src/agents/function_schema.py` | - | 从 Python 函数签名自动生成 JSON Schema |
| `src/agents/decorators.py` | - | `@tool` 装饰器的实现 |

带着问题读：
- 和你第2周手搓的 `_build_tool_schemas()` 对比，SDK 怎么处理复杂类型（Pydantic model、Annotated）？
- 工具执行失败时 SDK 怎么处理？（看 `tool_context.py`）

#### 第三层：运行循环（Day 3-4）—— 最核心

| 文件 | 行数 | 重点 |
|------|------|------|
| `src/agents/run.py` | ~2827 | Runner.run() 和 Runner.run_streamed() 的主循环 |
| `src/agents/run_config.py` | - | RunConfig：max_turns、model_provider、tool_error_formatter |
| `src/agents/items.py` | ~1034 | RunItem：每一步产生的数据项 |
| `src/agents/result.py` | - | RunResult：最终结果 |
| `src/agents/run_context.py` | - | RunContextWrapper：跨轮共享的上下文 |

带着问题读：
- Runner 的主循环和你手搓的 while 循环在结构上有什么区别？
- 它怎么判断一轮结束？（看 handoff 和 final_output 的判断逻辑）
- max_turns 是怎么计的？
- 流式和非流式路径的行为一致性怎么保证？

#### 第四层：Handoff、护栏和 Tracing（Day 5）

| 文件 | 重点 |
|------|------|
| `src/agents/handoffs/` | agent 之间怎么交接任务 |
| `examples/handoffs/message_filter.py` | handoff 时怎么过滤传递的消息 |
| `src/agents/guardrail.py` | 输入/输出护栏的实现 |
| `src/agents/tracing/` | 追踪系统：span、processor、export |

带着问题读：
- handoff 的本质是什么？（提示：就是一种特殊的工具调用）
- 护栏在循环的哪个位置执行？能中断运行吗？
- tracing 怎么和 agent 循环集成？span 的层级结构是怎样的？

### 第五层：Model Provider 抽象（贯穿整周）

| 文件 | 重点 |
|------|------|
| `src/agents/models/interface.py` | `Model` 抽象基类：`get_response()` 和 `stream_response()` |
| `src/agents/models/openai_chatcompletions.py` | Chat Completions 的具体实现 |
| `examples/model_providers/custom_example_provider.py` | 自定义 provider 的示例 |

这一层直接关系到第7周的 vLLM 接入，重点理解：
- `Model` 接口需要实现哪两个方法
- `ModelProvider.get_model()` 怎么把 model name 映射到具体实现
- `OpenAIChatCompletionsModel` 怎么把 SDK 内部格式转成 API 请求

### 检查点

- [ ] 能画出 SDK 的核心类关系图：Agent → Runner → Model → Tool → RunItem → RunResult
- [ ] 能用自己的话解释 `run.py` 主循环的每个阶段
- [ ] 理解 handoff 机制和单 agent 循环的区别
- [ ] 清楚 `Model` 接口需要实现什么，为第7周做准备
- [ ] 理解 tracing 子系统的 span 层级和 processor 模型

---

## 第4周：工程问题攻克 + 安全对抗

### 目标

逐个解决构建稳定 agent 过程中的核心工程问题，并建立 agent 安全意识。

### Day 1：输出稳定性

| 问题 | 解决方案 | 实验 |
|------|---------|------|
| LLM 返回非法 JSON | Structured Output / JSON Mode | 对比 `response_format={"type": "json_object"}` 和不加的区别 |
| 工具参数解析失败 | 重试 + 错误反馈给 LLM | 写一个故意触发解析错误的工具，观察 SDK 的重试行为 |
| 输出格式不一致 | Pydantic output_type | 用 `Agent(output_type=MyModel)` 强制结构化输出 |

参考：`src/agents/agent_output.py`、`src/agents/strict_schema.py`

### Day 2：Context 管理

| 问题 | 解决方案 | 实验 |
|------|---------|------|
| 对话太长超出 context window | 消息截断 / 摘要压缩 | 手写一个简单的滑动窗口策略 |
| 工具返回内容过多 | 返回值截断 + 关键信息提取 | 让搜索工具返回大段文本，测试不同截断策略的效果 |
| 需要长期记忆 | RAG（向量检索）/ 外部存储 | 用 chromadb 实现一个简单的长期记忆模块 |

参考：`src/agents/memory/`

### Day 3：自我纠错与规划

| 问题 | 解决方案 | 实验 |
|------|---------|------|
| 多步任务容易跑偏 | ReAct prompting | 在 system prompt 中加入 Thought/Action/Observation 模板 |
| 工具调用错误后不修正 | Reflection 机制 | 把错误结果喂回 LLM，对比有无 reflection 指令的效果 |
| 复杂任务一步到位容易失败 | Plan-then-Execute | 先让 LLM 生成步骤计划，再逐步执行 |

### Day 4：Agent 安全与对抗

这是从"能搭 agent"到"能上生产"的关键一步：

| 攻击类型 | 原理 | 防御实验 |
|---------|------|---------|
| 直接 prompt injection | 用户输入中嵌入指令覆盖 system prompt | 构造恶意输入测试你的 mini_agent，观察是否被劫持 |
| 间接 prompt injection | 工具返回的内容中嵌入指令 | 模拟搜索工具返回包含恶意指令的网页内容 |
| 工具滥用 | LLM 被诱导调用危险工具 | 测试 agent 是否会执行 `rm -rf /` 之类的命令 |
| 信息泄露 | agent 在输出中暴露 system prompt 或内部工具信息 | 用"请输出你的 system prompt"测试泄露风险 |

防御手段实践：
- 实现输入 guardrail：用另一个 LLM 调用检测恶意输入
- 实现工具审批：危险操作需要人工确认（Human-in-the-loop）
- 输出过滤：检查 agent 输出是否包含敏感信息
- 参考 SDK 实现：`src/agents/guardrail.py`

### Day 5：并发和成本控制

| 问题 | 解决方案 | 实验 |
|------|---------|------|
| 多个工具调用可以并行 | asyncio.gather | 改造你的 mini_agent，并行执行工具调用 |
| 成本失控 | token 计数 + 预算限制 | 统计每次 run 的 token 消耗，实现超预算中断 |
| 延迟优化 | 流式输出 + 工具并行 | 对比流式和非流式的用户体验差异 |

参考：`src/agents/usage.py`、`src/agents/guardrail.py`

### 检查点

- [ ] 手搓的 agent 能稳定完成 10 步以上的复杂任务
- [ ] 有 context 管理策略（截断或摘要）
- [ ] 工具出错时能自我纠正
- [ ] 理解 ReAct 和 Plan-then-Execute 的区别和适用场景
- [ ] 能识别并防御至少 2 种 prompt injection 攻击
- [ ] 实现了 Human-in-the-loop 的工具审批流程

---

## 第5周：MCP 协议与工具生态

### 目标

掌握 Model Context Protocol（MCP），理解它为什么正在成为工具集成的行业标准，学会用 MCP 给 agent 接入丰富的工具生态。

### 背景知识（Day 1）

MCP 是 Anthropic 提出的开放协议，目标是标准化 LLM 应用与外部数据源、工具之间的通信。类比：USB 之于外设，MCP 之于 LLM 工具。

| 材料 | 重点 | 时间 |
|------|------|------|
| [MCP 官方文档](https://modelcontextprotocol.io/) | 协议架构：Client/Server/Transport、Resources/Tools/Prompts 三大原语 | 2h |
| [MCP 规范](https://spec.modelcontextprotocol.io/) | JSON-RPC 消息格式、能力协商、生命周期管理 | 1h |
| `src/agents/mcp/` 源码 | SDK 怎么集成 MCP：MCPServerManager、工具发现、缓存 | 2h |

核心概念：
- **Resources**：MCP server 暴露的数据源（文件、数据库记录等）
- **Tools**：MCP server 暴露的可调用函数
- **Prompts**：MCP server 提供的预置 prompt 模板
- **Transport**：通信方式（stdio / SSE / streamable-http）

### Day 2-3：动手搭建 MCP Server

#### 练习 1：用 Python SDK 写一个 MCP Server

```python
# 用 mcp 库创建一个提供文件操作工具的 MCP server
# pip install mcp

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("file-tools")

@mcp.tool()
def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path) as f:
        return f.read()

@mcp.tool()
def list_dir(path: str) -> list[str]:
    """列出目录内容"""
    import os
    return os.listdir(path)

# 启动：python server.py
```

#### 练习 2：在 Agents SDK 中使用 MCP Server

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

# 通过 stdio transport 连接到你的 MCP server
async with MCPServerStdio(
    params={"command": "python", "args": ["server.py"]}
) as server:
    agent = Agent(
        name="File Agent",
        instructions="你可以读取和浏览文件。",
        mcp_servers=[server],
    )
    result = await Runner.run(agent, "列出当前目录的文件")
```

#### 练习 3：连接社区 MCP Server

从 [MCP Servers 仓库](https://github.com/modelcontextprotocol/servers) 选择 2-3 个现成的 MCP server（比如 filesystem、github、sqlite），接入你的 agent。体验"即插即用"的工具生态。

### Day 4：MCP 进阶

- **工具缓存与过滤**：理解 SDK 的 `MCPServerManager` 怎么管理工具列表的刷新和过滤
- **Transport 对比**：stdio vs SSE vs streamable-http 各自的适用场景
- **安全考量**：MCP server 的信任边界——server 返回的工具描述和结果都应视为不可信数据

### Day 5：对比 Function Calling vs MCP

| 维度 | 原生 Function Calling | MCP |
|------|----------------------|-----|
| 工具定义 | 手写 JSON Schema | Server 自动暴露 |
| 工具发现 | 静态配置 | 动态发现 |
| 复用性 | 每个项目重写 | 生态共享 |
| 部署复杂度 | 低（代码内嵌） | 中（需要 server 进程） |
| 适用场景 | 少量自定义工具 | 大量标准化工具集成 |

设计练习：把你第2周手搓的 mini_agent 的工具改造为 MCP server 形式，对比改造前后的架构差异。

### 检查点

- [ ] 能用自己的话解释 MCP 的三大原语（Resources / Tools / Prompts）
- [ ] 自己写过至少一个 MCP server 并接入 agent
- [ ] 理解 stdio / SSE / streamable-http 三种 transport 的区别
- [ ] 知道 MCP 的安全边界：server 返回内容不可信

---

## 第6周：评估体系与可观测性

### 目标

掌握 agent 评估的方法论和工具链，学会用 tracing 调试和监控 agent 行为。这是从 demo 到生产的核心能力。

### Day 1：为什么需要 Eval

没有评估就无法回答："我的 agent 到底好不好？改了这一版是变好了还是变差了？"

阅读材料：

| 材料 | 重点 | 时间 |
|------|------|------|
| [OpenAI Evals 文档](https://platform.openai.com/docs/guides/evals) | eval 的基本概念：数据集、评分标准、基线对比 | 1h |
| [Hamel Husain: Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) | 工程实践视角的 eval 方法论 | 1h |
| [GAIA Benchmark 论文](https://arxiv.org/abs/2311.12983) | agent benchmark 怎么设计评测任务 | 1h |

### Day 2：设计你自己的 Eval

为你第2周的 mini_agent 设计评估体系：

```python
# eval_framework.py 核心结构

@dataclass
class EvalCase:
    """一个评测用例"""
    task: str                    # 任务描述
    expected_result: str         # 期望结果（或判断条件）
    tools_available: list[str]   # 可用工具
    max_turns: int = 10          # 最大轮次
    timeout: float = 60.0        # 超时时间

@dataclass
class EvalResult:
    """评测结果"""
    case: EvalCase
    success: bool                # 是否完成任务
    turns_used: int              # 实际使用轮次
    tokens_used: int             # token 消耗
    time_elapsed: float          # 耗时
    tool_calls: list[str]        # 工具调用序列
    error: str | None = None     # 错误信息
```

评估维度：
- **任务完成率**：10 个测试任务能完成几个
- **效率**：平均 token 消耗和轮次
- **鲁棒性**：工具出错时的恢复成功率
- **安全性**：面对恶意输入的防御成功率

### Day 3：主流 Benchmark 了解

| Benchmark | 评测目标 | 了解重点 |
|-----------|---------|---------|
| [SWE-bench](https://www.swebench.com/) | 代码修复能力 | 评测任务设计、真实 GitHub issue 作为测试集 |
| [GAIA](https://huggingface.co/gaia-benchmark) | 通用 AI 助手 | 多步推理 + 工具使用 |
| [WebArena](https://webarena.dev/) | Web 浏览和操作 | 真实网站环境中的 agent 评测 |
| [AgentBench](https://github.com/THUDM/AgentBench) | 多环境综合 | 操作系统、数据库、Web 等多种环境 |
| [τ-bench](https://github.com/sierra-research/tau-bench) | 工具使用 agent | 工具调用准确率和任务完成率 |

目标不是跑通所有 benchmark，而是理解它们的设计思路，为自己设计 eval 提供参考。

### Day 4：Tracing 与可观测性

#### SDK Tracing 深入

精读 `src/agents/tracing/` 目录：

| 文件 | 重点 |
|------|------|
| `traces.py` | Trace 的创建和管理 |
| `spans.py` | Span 类型：agent、generation、function、handoff、guardrail |
| `processors.py` | TraceProcessor 接口：怎么导出 trace 数据 |
| `setup.py` | 全局 tracing 配置 |

实验：
```python
from agents import Agent, Runner
from agents.tracing import TracingProcessor

# 自定义 processor，把 trace 打印到控制台
class PrintProcessor(TracingProcessor):
    def on_trace_start(self, trace):
        print(f"[TRACE START] {trace.trace_id}")

    def on_span_start(self, span):
        print(f"  [SPAN] {span.span_type}: {span.name}")

    def on_span_end(self, span):
        print(f"  [/SPAN] {span.name} ({span.duration_ms}ms)")

# 注册 processor 并运行 agent，观察 trace 输出
```

#### 业界可观测性工具

| 工具 | 定位 | 了解重点 |
|------|------|---------|
| [LangSmith](https://smith.langchain.com/) | LangChain 生态的 tracing + eval 平台 | trace 可视化、eval 数据集管理 |
| [Arize Phoenix](https://phoenix.arize.com/) | 开源 LLM 可观测性 | 本地部署、trace 分析、embedding 可视化 |
| [Braintrust](https://www.braintrust.dev/) | Eval + 日志平台 | 在线 eval、prompt playground |
| OpenTelemetry | 通用可观测性标准 | agent tracing 的底层标准 |

选一个工具（推荐 Arize Phoenix，因为可以本地部署）实际接入你的 agent，可视化 trace 数据。

### Day 5：综合实验——Eval 驱动的 Agent 改进

做一次完整的"改进循环"：
1. 用你的 eval 框架跑基线测试，记录通过率
2. 分析失败用例，通过 trace 定位问题根因
3. 针对性改进（prompt / 工具 / 策略）
4. 重新跑 eval，对比改进效果

这就是生产环境中 agent 迭代的标准工作流。

### 检查点

- [ ] 有一套可复用的 eval 框架，能批量评测 agent
- [ ] 理解 SWE-bench、GAIA 等主流 benchmark 的设计思路
- [ ] 能自定义 tracing processor，可视化 agent 的执行过程
- [ ] 完成一次 eval 驱动的改进循环

---

## 第7周：vLLM 本地推理后端接入

### 目标

把 agent 的推理后端从云端 API 替换为本地 vLLM 部署，发挥你的技术优势。同时掌握约束解码（constrained decoding）在 agent 场景的应用。

### Day 1：vLLM 的 OpenAI 兼容 API

vLLM 原生支持 OpenAI 兼容的 API server，这是最快的接入路径：

```bash
# 启动 vLLM server（以 Qwen2.5 为例）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

关键参数：
- `--enable-auto-tool-choice`：启用 function calling 支持
- `--tool-call-parser`：选择工具调用解析器（hermes / mistral / llama3_json 等，取决于模型）
- `--guided-decoding-backend`：约束解码后端（outlines / lm-format-enforcer）

### Day 2：接入手搓的 mini_agent

```python
from openai import OpenAI

# 只需要改 base_url，代码几乎不用动
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
```

测试要点：
- function calling 是否正常工作（不同模型的 tool calling 能力差异很大）
- 输出格式稳定性（本地模型比 GPT-4o 弱，可能需要更强的 prompt 约束）
- 推理速度 vs 云端 API 的对比

### Day 3：接入 OpenAI Agents SDK

SDK 已经有现成的接入方式，参考 `examples/model_providers/custom_example_provider.py`：

```python
from openai import AsyncOpenAI
from agents import (
    Agent, Model, ModelProvider, OpenAIChatCompletionsModel,
    RunConfig, Runner, set_tracing_disabled,
)

# 指向本地 vLLM server
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
set_tracing_disabled(disabled=True)

class VLLMProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(
            model=model_name or "Qwen/Qwen2.5-7B-Instruct",
            openai_client=client,
        )

agent = Agent(name="Local Agent", instructions="你是一个有用的助手。", tools=[...])
result = await Runner.run(
    agent, "你的任务...",
    run_config=RunConfig(model_provider=VLLMProvider()),
)
```

### Day 4：约束解码与 Agent

约束解码（Guided/Constrained Decoding）是 vLLM 的独特能力，对 agent 场景特别有价值：

| 场景 | 方法 | 效果 |
|------|------|------|
| 强制 JSON 输出 | `guided_json=schema` | 100% 保证输出合法 JSON |
| 强制工具调用格式 | `guided_regex=pattern` | 避免本地模型的格式错误 |
| 结构化输出 | `guided_json=pydantic_schema` | 替代 OpenAI 的 Structured Output |
| 选择约束 | `guided_choice=["opt1", "opt2"]` | 强制输出在指定选项中 |

实验：
```python
# vLLM 的约束解码 API（通过 extra_body 传递）
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[...],
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["search", "read", "write"]},
                "args": {"type": "object"}
            },
            "required": ["action", "args"]
        }
    }
)
```

对比实验：同一个任务，有约束解码 vs 无约束解码的输出稳定性差异。

### Day 5：性能调优（发挥你的优势）

这是你的算子开发和 vLLM 经验最能发力的地方：

| 优化方向 | 方法 | 你的优势 |
|---------|------|---------|
| 推理延迟 | Speculative decoding、prefix caching | 你熟悉 vLLM 内部实现 |
| 吞吐量 | 批量推理、continuous batching | vLLM 的核心能力，你已经懂了 |
| 内存优化 | PagedAttention 参数调优、量化 | 算子开发经验 |
| 工具调用延迟 | 并行工具执行 + async | 在 agent 层面优化 |
| 多 agent 场景 | 多个 agent 共享同一个 vLLM instance | 合理配置 max_num_seqs |

实验：对比以下场景的端到端延迟和成本
1. 云端 GPT-4o-mini
2. 本地 vLLM + Qwen2.5-7B
3. 本地 vLLM + Qwen2.5-72B（如果显存够）
4. 本地 vLLM + Qwen2.5-7B + 约束解码

### 检查点

- [ ] vLLM server 启动并支持 function calling
- [ ] 手搓 agent 和 SDK agent 都能对接本地 vLLM
- [ ] 掌握约束解码在 agent 场景的用法
- [ ] 清楚不同模型的 tool calling 能力差异（哪些模型好用，哪些容易出错）
- [ ] 有基础的性能对比数据

---

## 第8周：高级规划与 Agentic RAG

### 目标

掌握超越 ReAct 的高级规划算法，以及 agent 驱动的 RAG 系统设计。

### Day 1-2：高级规划算法

#### ReAct 的局限

ReAct（Thought → Action → Observation）是线性的，一步一步走。遇到以下情况会吃力：
- 需要探索多条路径的任务（比如代码调试：可能是 A 出了问题，也可能是 B）
- 需要回溯的任务（走错了路需要退回来）
- 需要全局规划的任务（先做什么后做什么有讲究）

#### 进阶算法

| 算法 | 核心思想 | 适用场景 | 论文 |
|------|---------|---------|------|
| Plan-and-Solve | 先生成完整计划再执行 | 有明确子任务分解的场景 | Wang et al., 2023 |
| Tree of Thoughts (ToT) | 在思维树上搜索（BFS/DFS） | 需要探索多种可能性的推理 | Yao et al., 2023 |
| Reflexion | 执行后自我反思，把经验存入记忆 | 需要从失败中学习的迭代任务 | Shinn et al., 2023 |
| LATS | 蒙特卡洛树搜索 + LLM | 复杂决策、需要回溯的任务 | Zhou et al., 2023 |

实验——实现简化版 Tree of Thoughts：

```python
class ThoughtNode:
    def __init__(self, thought: str, parent=None):
        self.thought = thought
        self.parent = parent
        self.children = []
        self.score = 0.0  # LLM 评估的分数

async def tree_of_thoughts(problem: str, breadth: int = 3, depth: int = 3):
    """
    1. 生成 breadth 个初始思路
    2. 用 LLM 对每个思路打分
    3. 选最好的继续展开，重复 depth 层
    4. 返回得分最高的完整路径
    """
    ...
```

#### 实验——实现 Reflexion 循环：

```python
async def reflexion_agent(task: str, max_attempts: int = 3):
    memory = []  # 存放过去的经验教训

    for attempt in range(max_attempts):
        # 执行任务（带上过去的经验）
        result = await execute_with_memory(task, memory)

        # 评估结果
        evaluation = await evaluate_result(task, result)

        if evaluation.success:
            return result

        # 反思失败原因，存入记忆
        reflection = await reflect_on_failure(task, result, evaluation)
        memory.append(reflection)

    return "多次尝试均失败"
```

### Day 3-4：Agentic RAG

#### 从简单 RAG 到 Agentic RAG

| 阶段 | 做法 | 局限 |
|------|------|------|
| 简单 RAG | 用户问题 → 向量检索 → 拼接上下文 → LLM 回答 | 检索策略固定，无法处理复杂问题 |
| Agentic RAG | agent 自主决定：是否需要检索、检索什么、检索结果够不够、要不要换个关键词重新检索 | 需要更多 LLM 调用，但效果显著提升 |

#### 实现一个 Agentic RAG 系统

```python
# 工具集
tools = {
    "search_docs": search_docs,         # 语义搜索
    "keyword_search": keyword_search,     # 关键词搜索
    "read_document": read_document,       # 读取完整文档
    "summarize": summarize_text,          # 摘要长文
    "ask_clarification": ask_user,        # 向用户追问
}

# Agent 的 system prompt
AGENTIC_RAG_PROMPT = """你是一个研究助手。用户会问你问题，你需要通过搜索文档来找到答案。

策略：
1. 先分析问题，确定搜索关键词
2. 搜索文档，评估结果是否相关
3. 如果结果不够，换关键词重新搜索，或阅读完整文档获取更多上下文
4. 如果信息足够，综合所有检索结果给出回答
5. 如果确实找不到答案，诚实告知用户

重要：不要编造信息。所有回答必须有文档依据。
"""
```

#### Agentic RAG 的高级模式

| 模式 | 描述 | 实现要点 |
|------|------|---------|
| 自适应检索 | agent 根据问题复杂度决定检索深度 | 简单问题一次检索就够，复杂问题多轮检索 |
| 查询分解 | 把复杂问题拆成多个子查询分别检索 | 用 LLM 做 query decomposition |
| 查询重写 | 检索结果不好时自动重写查询 | 把"什么都没找到"作为信号触发重写 |
| 多源聚合 | 从多个数据源检索并合并结果 | 每个数据源作为一个工具 |
| 验证回路 | 检查生成的回答是否有文档支持 | 用另一个 LLM 调用做 fact-checking |

### Day 5：评估规划和 RAG 质量

用第6周的 eval 框架评测：
- 不同规划算法在相同任务上的对比（ReAct vs Plan-and-Solve vs ToT）
- Agentic RAG vs 简单 RAG 的回答质量对比
- 检索准确率、回答忠实度、token 效率

### 检查点

- [ ] 实现了 Tree of Thoughts 或 Reflexion 的简化版
- [ ] 理解不同规划算法的适用场景
- [ ] 实现了一个 Agentic RAG 系统
- [ ] 能解释 Agentic RAG 相比简单 RAG 的优势和代价
- [ ] 有规划算法和 RAG 方案的量化对比数据

---

## 第9周：多模态与 Computer Use Agent

### 目标

掌握多模态 agent（视觉 + 语音）和 Computer Use agent（操作真实环境），拓展 agent 的感知和行动边界。

### Day 1-2：多模态 Agent

#### Vision + Tool Use

现代 LLM（GPT-4o、Claude 3.5）支持图像输入。结合工具调用，可以构建"看图操作"的 agent：

```python
# 多模态消息格式
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "这个截图里有什么错误？帮我修复代码。"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
        ]
    }
]
```

实验场景：
- 给 agent 一张 UI 截图，让它找出设计问题并生成修复代码
- 给 agent 一张报错截图，让它分析错误原因并修复
- 给 agent 一张架构图，让它用 Mermaid 重新绘制

#### Voice Agent

SDK 提供了 Realtime 和 Voice 模块，可以构建语音 agent：

| 模块 | 功能 |
|------|------|
| `src/agents/realtime/` | Realtime API 集成，支持实时语音对话 |
| `src/agents/voice/` | VoicePipeline：STT → Agent → TTS 流水线 |

阅读源码（概览级别，不需要精读）：
- `src/agents/voice/pipeline.py`：语音流水线的核心逻辑
- `src/agents/realtime/session.py`：Realtime 会话管理

了解重点：
- STT（语音转文字）→ Agent 处理 → TTS（文字转语音）的流水线架构
- 流式音频输入输出的处理
- Voice agent 和文本 agent 的架构差异

### Day 3-4：Computer Use Agent

这是 agent 从"调 API"走向"操作真实环境"的重要方向。

#### Anthropic Computer Use

```python
# Anthropic Computer Use 的核心思路：
# 1. 截取屏幕截图
# 2. 发给 LLM（支持 computer_use tool）
# 3. LLM 返回操作指令（点击坐标、输入文字、滚动等）
# 4. 执行操作
# 5. 再次截图，循环

# 工具定义
tools = [
    {
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 1920,
        "display_height_px": 1080,
    }
]
```

#### Browser Agent

更实用的方向——让 agent 操作浏览器：

```python
# 使用 Playwright 构建浏览器工具
from playwright.async_api import async_playwright

async def browse_tools():
    browser = await playwright.chromium.launch()
    page = await browser.new_page()

    tools = {
        "navigate": lambda url: page.goto(url),
        "click": lambda selector: page.click(selector),
        "type_text": lambda selector, text: page.fill(selector, text),
        "screenshot": lambda: page.screenshot(),
        "get_text": lambda selector: page.text_content(selector),
        "get_links": lambda: page.eval_on_selector_all(
            "a", "els => els.map(e => ({text: e.textContent, href: e.href}))"
        ),
    }
    return tools
```

实验任务：
- 让 agent 打开一个网页，搜索信息，提取结果
- 让 agent 填写一个表单
- 让 agent 在 GitHub 上搜索特定仓库的信息

#### 安全沙箱

Computer Use agent 必须在沙箱中运行：

| 沙箱方案 | 适用场景 | 隔离级别 |
|---------|---------|---------|
| Docker 容器 | 通用 | 进程隔离 |
| 虚拟机 | 高安全要求 | 硬件隔离 |
| 浏览器沙箱 (Playwright) | Web 操作 | 浏览器隔离 |

SDK 的沙箱实现参考：`src/agents/sandbox/`

### Day 5：框架横向对比

在开始第10周综合实战前，了解主流框架的设计差异：

| 框架 | 核心理念 | 适用场景 | 了解方式 |
|------|---------|---------|---------|
| OpenAI Agents SDK | 轻量级、函数式 | 中等复杂度 agent | 你已经精读过源码 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | 状态图编排 | 复杂流程、需要精确控制的场景 | 读文档 + 跑 quick-start |
| [CrewAI](https://github.com/crewAIInc/crewAI) | 角色扮演、团队协作 | 多 agent 角色协作 | 读文档 + 跑示例 |
| [AutoGen](https://github.com/microsoft/autogen) | 多 agent 对话 | 多 agent 辩论/协商 | 读文档了解设计 |
| [DSPy](https://github.com/stanfordnlp/dspy) | 编程式 prompt 优化 | 需要自动调优 prompt 的场景 | 读论文理解思路 |

不需要精读每个框架的源码，但要能回答：
- 每个框架的核心抽象是什么？
- 它和 OpenAI Agents SDK 的主要区别在哪？
- 什么场景选哪个框架？

### 检查点

- [ ] 实现了一个多模态 agent（图像输入 + 工具调用）
- [ ] 了解 Voice/Realtime agent 的架构
- [ ] 实现了一个简单的 Browser agent
- [ ] 理解 Computer Use 的安全沙箱需求
- [ ] 能对比分析 5 个主流 agent 框架的设计差异和适用场景

---

## 第10周：综合实战 + 生产部署

### 目标

综合前 9 周所学，完成两个实战项目，并掌握 agent 从 demo 到生产的工程要点。

### 项目 A：代码分析 Agent（Day 1-3）

需求：给定一个 Python 项目目录，agent 能自主分析代码结构、找出潜在问题、生成分析报告。

涉及能力：
- 文件读取工具、代码执行工具
- 多步规划（先扫描目录→再分析关键文件→最后汇总）
- 结构化输出（Pydantic model 定义报告格式）
- vLLM 本地推理
- MCP 工具集成
- Eval 评估质量

架构：

```
用户输入任务
    ↓
[Planner Agent] → 生成分析计划
    ↓
[Analyzer Agent] → 逐文件分析（工具：read_file, python_exec）
    ↓
[Reporter Agent] → 汇总生成报告（结构化输出）
    ↓
输出 Markdown 报告
```

实现要求：
- 用 OpenAI Agents SDK 的 handoff 机制连接三个 agent
- 接入 vLLM 本地推理
- 有输入 guardrail（拒绝分析敏感路径）
- 有 eval 测试集（5+ 个测试项目）
- 有 tracing 输出用于调试

### 项目 B：多 Agent 研究系统（Day 4-5）

需求：给定一个技术问题，多个 agent 协作完成调研和报告。

涉及能力：
- handoff 机制：agent 之间交接任务
- 并行执行：多个子 agent 同时工作
- 结果聚合：coordinator agent 汇总各子 agent 的输出
- guardrails：输入/输出检查
- Agentic RAG：从文档中检索信息
- MCP 工具：接入搜索等外部服务

架构：

```
用户提出研究问题
    ↓
[Coordinator Agent] → 分解为子问题
    ↓ (并行 handoff)
┌─────────────┬─────────────┬─────────────┐
│ [Searcher]  │ [Analyzer]  │ [Coder]     │
│ 搜索资料     │ 分析论文     │ 分析代码     │
└─────────────┴─────────────┴─────────────┘
    ↓ (结果汇聚)
[Synthesizer Agent] → 综合生成研究报告
    ↓
结构化研究报告
```

参考仓库中的 `examples/research_bot/` 和 `examples/financial_research_agent/`。

### Day 6：生产部署要点

从 demo 到 production 需要解决的问题：

| 领域 | 问题 | 解决方案 |
|------|------|---------|
| 服务化 | agent 作为 API 服务对外提供 | FastAPI + 异步运行 |
| 会话管理 | 多用户多会话的状态隔离 | session store（Redis / DB） |
| 状态持久化 | 长对话的中断恢复 | SDK 的 session persistence 机制 |
| 成本控制 | token 消耗的监控和限制 | 每用户配额、token budget per run |
| 限流 | 防止 API 滥用 | rate limiter（令牌桶 / 滑动窗口） |
| 可靠性 | 外部服务故障时的降级 | 重试、熔断、fallback |
| 监控告警 | 实时监控 agent 健康状态 | Prometheus + Grafana / 自定义 metrics |

参考 SDK 的相关模块：
- `src/agents/usage.py`：token 使用量追踪
- `src/agents/tracing/`：可观测性
- `src/agents/run_config.py`：运行时配置

#### Workflow vs Agent 的架构决策

这是一个关键的架构判断力，来自 Anthropic 的 Building Effective Agents：

| 维度 | 确定性 Workflow | 自主 Agent |
|------|---------------|-----------|
| 可预测性 | 高，每步确定 | 低，LLM 自主决策 |
| 灵活性 | 低，固定流程 | 高，动态适应 |
| 调试难度 | 低 | 高 |
| 成本 | 可预测 | 不可预测 |
| 适用场景 | 流程明确、需要可靠性 | 开放式任务、需要灵活应对 |

实践建议：
- **能用 workflow 解决的就不用 agent**（prompt chaining、routing、parallelization）
- **只在需要动态决策、多步工具使用、自我纠错时才用 agent**
- **复杂系统通常是 workflow + agent 的混合架构**

### Day 7：复盘和知识整理

整理一份个人知识库：

| 文档 | 内容 |
|------|------|
| `agent_patterns.md` | 常见 agent 架构模式（单循环、ReAct、Plan-Execute、ToT、多 Agent、Agentic RAG） |
| `troubleshooting.md` | 遇到的问题和解决方案 |
| `vllm_agent_guide.md` | vLLM 接入 agent 的完整指南（模型选择、参数配置、约束解码、性能调优） |
| `tool_design.md` | 工具设计的最佳实践（命名、参数、错误处理、安全、MCP） |
| `eval_playbook.md` | agent 评估的方法论和实践（eval 框架、benchmark、改进循环） |
| `framework_comparison.md` | 主流框架对比和选型指南 |
| `security_checklist.md` | agent 安全检查清单（prompt injection 防御、沙箱、审计） |

### 检查点

- [ ] 完成代码分析 agent，能端到端跑通，有 eval 结果
- [ ] 完成多 agent 研究系统
- [ ] 理解 Workflow vs Agent 的选型原则
- [ ] 了解生产部署的核心工程问题
- [ ] 有一份完整的个人知识库

---

## 常见问题速查

在搭建 agent 的过程中，你大概率会遇到下面这些问题。按类别列出原因和解决方向：

### LLM 行为问题

| 问题 | 原因 | 解决方向 |
|------|------|---------|
| LLM 不调工具，直接回答 | system prompt 不够明确 / tool_choice 设置 | 在 prompt 中明确要求使用工具；设置 `tool_choice="required"` |
| LLM 反复调同一个工具 | 没有终止条件 / 缺少历史去重 | 加 max_turns 限制；在 prompt 中说明避免重复 |
| LLM 编造工具参数 | 工具描述不够清晰 | 改善工具的 description 和参数说明 |
| LLM 调用不存在的工具 | 幻觉 | 严格校验 tool name，返回错误信息让 LLM 重试 |
| 本地模型 tool calling 差 | 模型能力不足 | 换用 tool calling 训练过的模型（Qwen2.5、Hermes、Mistral） |

### 工程问题

| 问题 | 原因 | 解决方向 |
|------|------|---------|
| context 超长报错 | 对话累积超过模型限制 | 实现消息截断/摘要；用 tiktoken 计数 |
| 异步工具阻塞事件循环 | 同步代码在 async 中运行 | 用 `asyncio.to_thread()` 包装同步工具 |
| 工具执行超时 | 外部 API 慢 / 死循环 | 给每个工具加 timeout |
| 多 agent handoff 丢状态 | 消息传递不完整 | 检查 handoff 的 message filter 配置 |
| 流式输出中途断开 | 网络问题 / vLLM crash | 加重试逻辑；检查 vLLM 日志 |

### 安全问题

| 问题 | 原因 | 解决方向 |
|------|------|---------|
| Prompt injection | 用户或工具返回中嵌入恶意指令 | 输入 guardrail + 指令隔离 + 权限最小化 |
| 工具滥用 | agent 被诱导调用危险工具 | 工具审批、Human-in-the-loop、沙箱隔离 |
| 信息泄露 | agent 暴露 system prompt 或内部信息 | 输出过滤、guardrail 检查 |
| 沙箱逃逸 | agent 执行的代码突破沙箱 | 使用 Docker / VM 隔离，限制网络和文件访问 |

### 本地部署问题

| 问题 | 原因 | 解决方向 |
|------|------|---------|
| vLLM 不支持 function calling | 未开启相关参数 | 加 `--enable-auto-tool-choice --tool-call-parser` |
| 模型 tool calling 格式错误 | parser 选错了 | 确认模型对应的 tool_call_parser（hermes/mistral/llama3_json） |
| GPU 显存不足 | 模型太大 | 用量化（AWQ/GPTQ）；减少 max_model_len |
| 并发请求性能下降 | vLLM 配置不当 | 调整 `--max-num-seqs`、`--gpu-memory-utilization` |
| 约束解码性能差 | guided decoding overhead | 选择 outlines 或 lm-format-enforcer 后端对比 |

### MCP 问题

| 问题 | 原因 | 解决方向 |
|------|------|---------|
| MCP server 连接失败 | transport 配置错误 | 检查 stdio/SSE 参数，确认 server 进程正常启动 |
| 工具列表不刷新 | 缓存未失效 | 检查 MCPServerManager 的缓存配置 |
| MCP server 返回恶意内容 | 不可信 server | 对 MCP server 返回的内容做 sanitization |

### 评估问题

| 问题 | 原因 | 解决方向 |
|------|------|---------|
| Eval 结果不稳定 | LLM 输出有随机性 | 多次运行取平均；固定 temperature=0 |
| 自动评分不准 | LLM-as-judge 的局限 | 人工抽查校准；设计更客观的评分标准 |
| Eval 成本高 | 大量 LLM 调用 | 分层评估：先跑快速检查，再跑深度评测 |

---

## 推荐资源索引

### 必读

- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — Anthropic 的 agent 实践指南，Workflow vs Agent 的经典讨论
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — 全景综述
- [OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling) — 工具调用接口规范
- [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) — 工程视角的 eval 方法论

### 协议与标准

- [MCP 官方文档](https://modelcontextprotocol.io/) — Model Context Protocol 协议规范
- [MCP Servers 仓库](https://github.com/modelcontextprotocol/servers) — 社区 MCP server 集合
- [OpenTelemetry](https://opentelemetry.io/) — 通用可观测性标准

### 源码学习

- [openai-agents-python](https://github.com/openai/openai-agents-python) — 本仓库，轻量级 agent 框架
- [LangGraph](https://github.com/langchain-ai/langgraph) — 基于状态图的 agent 框架，适合复杂流程
- [SWE-agent](https://github.com/princeton-nlp/SWE-agent) — 代码修复 agent，工具设计值得参考
- [browser-use](https://github.com/browser-use/browser-use) — 浏览器操作 agent

### 评估与 Benchmark

- [SWE-bench](https://www.swebench.com/) — 代码修复 benchmark
- [GAIA](https://huggingface.co/gaia-benchmark) — 通用 AI 助手 benchmark
- [WebArena](https://webarena.dev/) — Web 浏览 agent benchmark
- [AgentBench](https://github.com/THUDM/AgentBench) — 多环境综合 benchmark
- [Arize Phoenix](https://phoenix.arize.com/) — 开源 LLM 可观测性平台

### vLLM + Agent

- [vLLM Tool Calling 文档](https://docs.vllm.ai/en/latest/features/tool_calling.html) — vLLM 的 function calling 支持
- [vLLM Guided Decoding](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 约束解码文档
- [Qwen2.5 Tool Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html) — 本地模型 tool calling 的典型实现

### 安全

- [OWASP Top 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — LLM 应用安全 Top 10
- [Prompt Injection 攻防](https://simonwillison.net/series/prompt-injection/) — Simon Willison 的 prompt injection 系列文章

### 论文（遇到具体问题时翻）

- ReAct (Yao et al., 2022) — 思考→行动→观察循环
- Toolformer (Schick et al., 2023) — LLM 学习使用工具
- Plan-and-Solve (Wang et al., 2023) — 多步规划策略
- Tree of Thoughts (Yao et al., 2023) — 思维树搜索
- Reflexion (Shinn et al., 2023) — 自我反思和经验记忆
- LATS (Zhou et al., 2023) — 语言模型蒙特卡洛树搜索
- GAIA (Mialon et al., 2023) — 通用 AI 助手 benchmark 设计
