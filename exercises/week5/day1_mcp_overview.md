# Day 1：MCP 协议全景

> 本文档梳理 Model Context Protocol（MCP）的背景、核心原理、协议细节和应用场景，作为 Week 5 学习的基础参考。

## 1. MCP 是什么

MCP（Model Context Protocol）是 Anthropic 于 2024 年 11 月提出的一个**开放标准协议**，定义了 LLM 应用与外部数据源、工具之间的通信方式。

一句话总结：**MCP 是 LLM 工具调用的 USB 接口标准。**

在 MCP 出现之前，每个 LLM 应用要接入外部工具，都需要自己定义工具的 JSON Schema、写调用逻辑、处理返回值——就像每个外设都用自己独有的接口。MCP 把这些全部标准化了：工具怎么描述自己、怎么被发现、怎么被调用、怎么返回结果，全部走统一协议。

## 2. 为什么需要 MCP

### 2.1 没有 MCP 时的痛点

回顾你 Week 1-2 的经验，接入一个工具需要：

```
1. 手写 JSON Schema 描述工具（参数名、类型、描述）
2. 在 agent 代码里注册工具函数
3. 解析 LLM 返回的 tool_calls，路由到对应函数
4. 执行函数，把结果格式化后喂回 LLM
```

这套流程的问题：

| 痛点 | 具体表现 |
|------|---------|
| **重复劳动** | 每个项目都要重新定义工具的 schema 和调用逻辑 |
| **耦合严重** | 工具定义和 agent 代码绑死在一起，换个 agent 框架就要重写 |
| **无法复用** | 你写的 `read_file` 工具，别人没法直接用，得自己再写一遍 |
| **发现困难** | 工具列表是静态硬编码的，不能动态增减 |
| **生态碎片化** | LangChain、CrewAI、AutoGen 各有一套工具接口，互不兼容 |

### 2.2 MCP 的解决思路

MCP 在 LLM 应用和工具之间加了一层标准协议：

```
┌──────────────────┐         MCP 协议          ┌──────────────────┐
│                  │ ◄──── JSON-RPC 2.0 ────► │                  │
│   MCP Client     │                           │   MCP Server     │
│  （LLM 应用/     │  • 工具发现 (list_tools)   │  （工具提供方）   │
│   Agent 框架）   │  • 工具调用 (call_tool)    │                  │
│                  │  • 资源读取 (read_resource)│  可以是：         │
│  如：Agents SDK  │  • prompt 获取 (get_prompt)│  • 文件系统服务   │
│      Claude      │                           │  • 数据库查询     │
│      Cursor      │                           │  • GitHub API     │
│                  │                           │  • 搜索引擎       │
└──────────────────┘                           └──────────────────┘
```

关键设计决策：
- **Client/Server 架构**：LLM 应用是 Client，工具提供方是 Server
- **协议层用 JSON-RPC 2.0**：成熟的 RPC 标准，有请求/响应/通知三种消息类型
- **Server 自描述**：Server 启动后会告诉 Client 自己有哪些工具、资源和 prompt
- **传输层可插拔**：stdio、SSE、Streamable HTTP 三种 transport，适配不同部署场景

### 2.3 类比理解

| 类比对象 | MCP 对应 | 说明 |
|---------|---------|------|
| USB 协议 | MCP 协议 | 标准化设备和主机之间的通信 |
| USB 设备（键盘、鼠标） | MCP Server | 提供功能，自描述能力 |
| USB 主机（电脑） | MCP Client | 发现设备、使用设备 |
| USB 接口类型（A/C/Micro） | Transport（stdio/SSE/HTTP） | 物理连接方式不同，协议相同 |
| USB 设备驱动 | 工具 Schema | 告诉主机怎么和设备交互 |

## 3. 核心概念

### 3.1 三大原语（Primitives）

MCP 定义了三种 Server 可以暴露的能力：

#### Tools（工具）—— 最核心

Server 暴露的可调用函数。对应你之前用的 function calling，但由 Server 自动暴露，Client 自动发现。

```python
# MCP Server 端定义工具
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("my-server")

@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"The weather in {city} is sunny."
```

对比你 Week 1 手写的 JSON Schema：

```python
# 之前：手动写 JSON Schema
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}]

# MCP：装饰器自动生成 schema，Server 自动暴露
# Client 通过 list_tools() 自动获取
```

#### Resources（资源）—— 只读数据源

Server 暴露的数据，Client 可以读取但不能修改。类似 REST API 的 GET 端点。

```
资源 URI 示例：
  file:///path/to/document.txt    → 本地文件
  postgres://db/users/123         → 数据库记录
  github://repo/issues/42         → GitHub issue
```

适用场景：给 LLM 提供上下文数据（文档、配置、数据库记录），而不是执行操作。

#### Prompts（提示模板）—— 预置 prompt 片段

Server 提供的 prompt 模板，Client 可以获取并在构建对话时使用。

```python
# Server 端定义 prompt
@mcp.prompt()
def generate_code_review_instructions(
    focus: str = "general code quality",
    language: str = "python"
) -> str:
    """生成代码审查指令"""
    return f"""You are a senior {language} code review specialist.
Focus on {focus}. Analyze code for quality, security, performance..."""
```

适用场景：把领域专家的 prompt 知识封装到 Server 里，不同 Client 共享使用。

### 3.2 三者的对比

| 维度 | Tools | Resources | Prompts |
|------|-------|-----------|---------|
| 操作类型 | 执行动作（有副作用） | 读取数据（只读） | 获取模板（只读） |
| 谁控制调用 | LLM 决定何时调用 | 应用程序决定 | 用户/应用决定 |
| 类比 | POST/PUT 请求 | GET 请求 | 配置模板 |
| 安全风险 | 高（可能执行危险操作） | 低（只是读数据） | 低（只是文本） |
| 典型例子 | 发邮件、写文件、执行代码 | 读文件、查数据库 | 代码审查指令、翻译指令 |

实际使用中，**Tools 是最常用的**，大部分 MCP Server 主要暴露工具。

### 3.3 协议生命周期

一次完整的 MCP 会话经历以下阶段：

```
1. 初始化（Initialize）
   Client → Server:  initialize 请求，告知 Client 支持的协议版本和能力
   Server → Client:  返回 Server 支持的能力（tools? resources? prompts?）
   Client → Server:  initialized 通知，确认初始化完成

2. 能力协商（Capability Negotiation）
   在初始化阶段完成。双方各自声明支持什么：
   - Server 声明：我有 tools / resources / prompts
   - Client 声明：我支持 sampling / roots

3. 正常通信（Normal Operation）
   Client → Server:  tools/list          → Server 返回工具列表
   Client → Server:  tools/call          → Server 执行工具并返回结果
   Client → Server:  resources/list      → Server 返回资源列表
   Client → Server:  resources/read      → Server 返回资源内容
   Client → Server:  prompts/list        → Server 返回 prompt 列表
   Client → Server:  prompts/get         → Server 返回具体 prompt

4. 关闭（Shutdown）
   Client 关闭连接，Server 清理资源
```

### 3.4 消息格式：JSON-RPC 2.0

MCP 的底层消息格式是 JSON-RPC 2.0，有三种消息类型：

```json
// 请求（Request）—— 期望对方回复
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "get_weather",
        "arguments": {"city": "Tokyo"}
    }
}

// 响应（Response）—— 对请求的回复
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "content": [{"type": "text", "text": "Sunny, 25°C"}],
        "isError": false
    }
}

// 通知（Notification）—— 单向消息，不期望回复
{
    "jsonrpc": "2.0",
    "method": "notifications/tools/list_changed"
}
```

## 4. Transport：三种传输方式

MCP 的协议层和传输层是分离的。同一套 JSON-RPC 消息，可以通过不同的传输方式发送。

### 4.1 stdio（标准输入输出）

```
┌────────┐  stdin (JSON-RPC)  ┌────────┐
│ Client │ ────────────────► │ Server │
│        │ ◄──────────────── │(子进程) │
└────────┘  stdout (JSON-RPC) └────────┘
```

- **工作方式**：Client 启动 Server 作为子进程，通过 stdin/stdout 通信
- **适用场景**：本地工具、CLI 工具、开发调试
- **优点**：最简单，不需要网络，不需要端口
- **缺点**：只能本地使用，Server 和 Client 必须在同一台机器

SDK 中的实现：`MCPServerStdio`

```python
# 启动一个 stdio MCP server
async with MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    },
) as server:
    # server 是一个子进程，通过 stdin/stdout 通信
    tools = await server.list_tools()
```

### 4.2 SSE（Server-Sent Events）

```
┌────────┐  HTTP POST (请求)   ┌────────┐
│ Client │ ────────────────► │ Server │
│        │ ◄──────────────── │(HTTP)  │
└────────┘  SSE stream (响应)  └────────┘
```

- **工作方式**：Client 通过 HTTP POST 发请求，Server 通过 SSE 推送响应
- **适用场景**：远程 Server、需要网络访问的场景
- **优点**：支持远程访问、基于 HTTP 基础设施
- **缺点**：单向流（Server→Client），Client→Server 需要单独的 HTTP 请求

SDK 中的实现：`MCPServerSse`

```python
async with MCPServerSse(
    params={"url": "http://localhost:8000/sse"},
) as server:
    tools = await server.list_tools()
```

### 4.3 Streamable HTTP（推荐的新标准）

```
┌────────┐  HTTP POST + SSE   ┌────────┐
│ Client │ ◄───────────────► │ Server │
│        │  双向流式通信       │(HTTP)  │
└────────┘                    └────────┘
```

- **工作方式**：基于 HTTP，支持双向流式通信
- **适用场景**：生产环境、需要高性能的远程 Server
- **优点**：双向流、支持会话恢复、可以穿越代理和负载均衡
- **缺点**：实现复杂度稍高

SDK 中的实现：`MCPServerStreamableHttp`

```python
async with MCPServerStreamableHttp(
    params={"url": "http://localhost:8000/mcp"},
) as server:
    tools = await server.list_tools()
```

### 4.4 三种 Transport 选型

| 维度 | stdio | SSE | Streamable HTTP |
|------|-------|-----|-----------------|
| 部署位置 | 仅本地 | 本地或远程 | 本地或远程 |
| 网络要求 | 无 | HTTP | HTTP |
| 方向 | 双向 | 半双工 | 全双工 |
| 会话恢复 | 不支持 | 不支持 | 支持 |
| 穿越代理 | 不适用 | 部分支持 | 完全支持 |
| 复杂度 | 最低 | 中等 | 较高 |
| 典型场景 | 开发调试、本地CLI工具 | 简单远程服务 | 生产环境 |
| 协议状态 | 稳定 | 即将被替代 | **推荐** |

注意：MCP 官方正在推动从 SSE 迁移到 Streamable HTTP。新项目建议直接用 Streamable HTTP。

## 5. SDK 中的 MCP 实现

### 5.1 类继承关系

```
MCPServer (抽象基类)
│
│  核心接口：
│  ├── connect()          → 连接到 Server
│  ├── cleanup()          → 断开连接、清理资源
│  ├── list_tools()       → 获取工具列表
│  ├── call_tool()        → 调用工具
│  ├── list_prompts()     → 获取 prompt 列表
│  ├── get_prompt()       → 获取具体 prompt
│  ├── list_resources()   → 获取资源列表
│  └── read_resource()    → 读取资源内容
│
├── _MCPServerWithClientSession (内部基类，管理 MCP ClientSession)
│   ├── MCPServerStdio          → stdio transport
│   ├── MCPServerSse            → SSE transport
│   └── MCPServerStreamableHttp → Streamable HTTP transport
│
MCPServerManager
│  管理多个 MCPServer 的生命周期
│  ├── connect_all()     → 并行连接所有 Server
│  ├── cleanup_all()     → 并行清理所有 Server
│  └── 内部 _ServerWorker → 每个 Server 一个独立的 worker task
```

### 5.2 工具转换流程

MCP Server 返回的工具是 MCP 格式，SDK 需要把它转换成内部的 `FunctionTool`：

```
MCP Server                      Agents SDK
──────────                      ──────────
MCPTool {                       FunctionTool {
  name: "get_weather"    ──►      name: "get_weather"
  description: "..."     ──►      description: "..."
  inputSchema: {...}     ──►      params_json_schema: {...}
}                                 on_invoke_tool: <调用 MCP server>
                                }
```

这个转换在 `src/agents/mcp/util.py` 的 `MCPUtil` 类中完成。转换后，Runner 完全不知道这个工具来自 MCP——它和普通的 `@tool` 装饰器定义的工具看起来完全一样。

### 5.3 Agent 使用 MCP 的完整流程

```
1. Agent 定义时指定 mcp_servers=[server1, server2, ...]

2. Runner.run() 启动时：
   ├── 对每个 MCP Server 调用 list_tools()
   ├── MCPUtil 把 MCPTool 转换为 FunctionTool
   ├── 合并到 Agent 的总工具列表（本地工具 + MCP 工具）
   └── 工具列表发给 LLM

3. LLM 返回 tool_calls 时：
   ├── Runner 根据工具名找到对应的 FunctionTool
   ├── 如果是 MCP 工具，FunctionTool.on_invoke_tool 会调用 server.call_tool()
   ├── MCP Server 执行工具，返回 CallToolResult
   └── 结果转换为字符串，喂回 LLM

4. 运行结束后：
   └── server.cleanup() 断开连接
```

### 5.4 关键配置项

```python
MCPServerStdio(
    params={...},                        # 启动命令
    cache_tools_list=False,              # 是否缓存工具列表
    # True: 只在首次调用 list_tools() 时从 Server 获取，之后用缓存
    # False: 每次都从 Server 获取（默认）

    client_session_timeout_seconds=5,    # MCP 会话超时
    tool_filter=None,                    # 工具过滤器（选择暴露哪些工具）
    max_retry_attempts=0,                # 失败重试次数
    retry_backoff_seconds_base=1.0,      # 重试退避基数
    require_approval=None,               # 工具审批策略
    # "always"  → 所有工具都需要审批
    # "never"   → 都不需要
    # {"delete_file": "always", "read_file": "never"} → 按工具配置
)
```

## 6. 应用场景

### 6.1 场景一：文件系统操作

最基础的场景。不需要自己写文件操作工具，直接用社区 MCP Server：

```python
# 用社区的 filesystem MCP server
async with MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    },
) as server:
    agent = Agent(name="File Agent", mcp_servers=[server])
    result = await Runner.run(agent, "列出所有 Python 文件")
```

### 6.2 场景二：数据库查询

```python
# SQLite MCP server
async with MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sqlite", "mydb.sqlite"],
    },
) as server:
    agent = Agent(name="DB Agent", mcp_servers=[server])
    result = await Runner.run(agent, "查询销售额最高的前 10 个产品")
```

### 6.3 场景三：多 Server 组合

MCP 的真正威力在于组合多个 Server：

```python
# 同时接入文件系统 + GitHub + 搜索
async with MCPServerStdio(params=fs_params) as fs_server, \
         MCPServerStdio(params=github_params) as github_server, \
         MCPServerStreamableHttp(params=search_params) as search_server:

    agent = Agent(
        name="Research Agent",
        mcp_servers=[fs_server, github_server, search_server],
    )
    result = await Runner.run(agent, "搜索最新的 MCP 相关论文，总结后写入 report.md")
```

Agent 可以自主决定用哪个 Server 的哪个工具。

### 6.4 场景四：工具共享和复用

团队内部的工具可以包装成 MCP Server，任何 Agent 框架都能接入：

```
团队内部 MCP Server 生态：
├── deploy-server     → 部署相关工具（发布、回滚、查看状态）
├── monitor-server    → 监控工具（查询 metrics、查看告警）
├── db-server         → 数据库工具（查询、报表生成）
└── doc-server        → 文档工具（搜索内部文档、知识库）

任何 Agent 都可以接入这些 Server，不需要重写工具逻辑。
```

### 6.5 场景五：IDE 集成

这是 MCP 当前最广泛的实际应用：

- **Cursor / Windsurf**：通过 MCP 接入各种开发工具
- **Claude Desktop**：支持配置 MCP Server，扩展 Claude 的能力
- **VS Code（Copilot）**：正在集成 MCP 支持

这些 IDE 都是 MCP Client，用户配置的工具 Server 遵循同一协议。

## 7. 安全考量

### 7.1 MCP Server 的信任边界

关键原则：**MCP Server 返回的所有内容都应视为不可信数据。**

```
信任边界分析：
┌─────────────────────────────────────────────────┐
│  你的应用（信任区域）                              │
│  ├── Agent 代码                                  │
│  ├── System Prompt                               │
│  └── 本地工具（你自己写的，可信）                    │
├─────────────────── 信任边界 ─────────────────────┤
│  MCP Server 返回的内容（不可信区域）                │
│  ├── 工具描述 → 可能包含 prompt injection           │
│  ├── 工具执行结果 → 可能包含恶意指令                 │
│  └── 资源内容 → 可能包含恶意数据                    │
└─────────────────────────────────────────────────┘
```

### 7.2 攻击向量

| 攻击方式 | 描述 | 防御 |
|---------|------|------|
| 恶意工具描述 | Server 在工具 description 中嵌入指令，影响 LLM 行为 | 审查第三方 Server 的工具描述 |
| 间接 prompt injection | 工具返回的内容中嵌入恶意指令 | 输出过滤、guardrail |
| 工具名伪装 | Server 暴露一个名为 `transfer_money` 的工具，但实际做其他事 | 工具审批机制（require_approval） |
| 数据泄露 | 恶意 Server 收集 agent 发送的参数 | 只使用可信的 Server |

### 7.3 SDK 提供的安全机制

```python
# 1. 工具过滤：只暴露特定工具给 Agent
from agents.mcp import create_static_tool_filter
server = MCPServerStdio(
    params={...},
    tool_filter=create_static_tool_filter(["read_file", "list_dir"]),
    # 只允许 read_file 和 list_dir，其他工具不暴露给 LLM
)

# 2. 工具审批：危险操作需要确认
server = MCPServerStdio(
    params={...},
    require_approval={
        "write_file": "always",   # 写文件需要审批
        "delete_file": "always",  # 删文件需要审批
        "read_file": "never",     # 读文件不需要
    },
)

# 3. 工具 guardrail：输入/输出检查
server = MCPServerStdio(
    params={...},
    tool_input_guardrails=[my_input_check],
    tool_output_guardrails=[my_output_check],
)
```

## 8. MCP vs 原生 Function Calling 对比

| 维度 | 原生 Function Calling | MCP |
|------|----------------------|-----|
| 工具定义 | 手写 JSON Schema | Server 装饰器自动生成 |
| 工具发现 | 静态硬编码在代码里 | 运行时动态发现（list_tools） |
| 工具执行 | 本地函数调用 | 通过协议远程调用（可以是本地子进程或远程 HTTP） |
| 复用性 | 每个项目重写 | 跨框架、跨项目复用 |
| 生态 | 无标准生态 | 有社区 Server 仓库（github、filesystem、sqlite 等） |
| 部署复杂度 | 低（代码内嵌） | 中（需要 Server 进程） |
| 调试难度 | 低（直接断点） | 中（需要看 Server 日志） |
| 性能开销 | 无额外开销 | 有进程间通信 / 网络开销 |
| 适用场景 | 少量自定义工具、性能敏感 | 大量标准化工具、需要跨团队共享 |

**选型建议**：
- 工具少且简单 → 原生 function calling + `@tool` 装饰器
- 工具多且需要复用 → MCP Server
- 团队共享工具 → MCP Server
- 需要接入社区工具 → MCP Server
- 性能极致要求 → 原生 function calling

## 9. 生态现状（2025 年中）

### 9.1 官方和社区 Server

MCP 的 Server 生态正在快速增长：

| Server | 功能 | 来源 |
|--------|------|------|
| filesystem | 文件读写、目录浏览 | 官方 |
| github | GitHub API 操作 | 官方 |
| sqlite | SQLite 数据库查询 | 官方 |
| postgres | PostgreSQL 查询 | 官方 |
| google-maps | 地图和位置服务 | 官方 |
| brave-search | Web 搜索 | 社区 |
| slack | Slack 消息操作 | 社区 |

完整列表：https://github.com/modelcontextprotocol/servers

### 9.2 Client 支持

主流 LLM 应用和 IDE 正在或已经支持 MCP：

- ✅ Claude Desktop（Anthropic）
- ✅ Cursor（AI IDE）
- ✅ Windsurf（AI IDE）
- ✅ OpenAI Agents SDK（本仓库）
- ✅ Claude Code（CLI）
- 🔜 VS Code Copilot

## 10. 关键要点总结

1. **MCP 是 LLM 工具调用的标准化协议**，解决工具碎片化和复用问题
2. **三大原语**：Tools（执行操作）、Resources（读数据）、Prompts（获取模板），其中 Tools 最常用
3. **三种 Transport**：stdio（本地）、SSE（即将淘汰）、Streamable HTTP（推荐）
4. **底层是 JSON-RPC 2.0**，Client/Server 架构，Server 自描述能力
5. **安全原则**：MCP Server 返回的一切内容都不可信，需要过滤、审批和 guardrail
6. **选型**：简单场景用原生 function calling，复用和生态场景用 MCP
7. **在 SDK 中**，MCP 工具最终被转换为 `FunctionTool`，对 Runner 透明

## 11. 接下来的练习方向

- Day 2-3：自己写 MCP Server，在 SDK 中使用，接社区 Server
- Day 4：深入工具缓存、过滤、transport 对比
- Day 5：Function Calling vs MCP 架构对比，改造 mini_agent
