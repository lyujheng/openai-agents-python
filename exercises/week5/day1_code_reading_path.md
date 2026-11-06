# Day 1：MCP 代码阅读路径

> 理论已掌握后，按此路径阅读代码，从 examples 到源码逐层深入。

## 第一层：看 examples，理解怎么用（30 分钟）

目标：建立体感——Server 怎么定义，Client 怎么连，Agent 怎么用。

### 1. SSE 示例（最完整的入门，同时有 Server 和 Client）

```
examples/mcp/sse_example/
├── server.py   ← 先看这个，42 行，定义了 3 个工具
└── main.py     ← 再看这个，Client 怎么连 Server、创建 Agent、调用
```

重点关注：
- `server.py`：`@mcp.tool()` 装饰器怎么定义工具，`mcp.run(transport="sse")` 怎么启动
- `main.py`：`MCPServerSse` 怎么连接，`Agent(mcp_servers=[server])` 怎么使用

### 2. Filesystem 示例（用社区现成的 Server，不用自己写）

```
examples/mcp/filesystem_example/
└── main.py     ← 不用自己写 Server，用 npx 启动社区 Server
```

重点关注：
- `MCPServerStdio` 怎么通过 `params={"command": "npx", "args": [...]}` 启动现成 Server
- 和 SSE 示例对比，Client 端的 Agent 使用方式完全一样，区别只在连接方式

### 3. Streamable HTTP 示例（推荐的新 transport）

```
examples/mcp/streamablehttp_example/
├── server.py   ← Server 端，和 SSE 的 server.py 几乎一样
└── main.py     ← Client 端，用 MCPServerStreamableHttp 连接
```

重点关注：
- Server 端代码几乎不变，只是 `mcp.run(transport="streamable-http")`
- Client 端换成 `MCPServerStreamableHttp`，其余代码一样
- 体会 transport 可插拔的设计：同一个 Server 逻辑，换个 transport 就行

### 4. Prompt Server 示例（Prompts 原语怎么用）

```
examples/mcp/prompt_server/
├── server.py   ← Server 用 @mcp.prompt() 暴露 prompt 模板
└── main.py     ← Client 获取 prompt，用作 Agent 的 instructions
```

重点关注：
- `@mcp.prompt()` 和 `@mcp.tool()` 的区别
- Client 端怎么通过 `server.get_prompt()` 获取 prompt

### 5. 工具过滤示例（安全相关）

```
examples/mcp/tool_filter_example/
└── main.py     ← 怎么控制只暴露特定工具给 Agent
```

重点关注：
- `tool_filter` 参数的用法
- 理解这是 Client 端的安全机制

---

## 第二层：看 SDK 源码，理解 Client 端实现（1-2 小时）

目标：理解 MCP 工具从 Server 到 LLM 的完整链路。

### 按调用链顺序阅读：

```
Agent(mcp_servers=[server])
  │
  ▼
Runner.run() 启动时调用 list_tools()
  │
  ▼
MCPServer.list_tools()              ← 第1个文件
  │  从 Server 进程获取 MCPTool 列表
  ▼
MCPUtil.convert_tools()             ← 第2个文件
  │  MCPTool → FunctionTool 转换
  ▼
合并到 Agent 的工具列表，发给 LLM
  │
  ▼
LLM 返回 tool_calls
  │
  ▼
FunctionTool.on_invoke_tool()
  │  内部调用 MCPServer.call_tool()  ← 第1个文件
  ▼
Server 执行工具，返回 CallToolResult
```

### 1. MCPServer 抽象接口

```
src/agents/mcp/server.py  →  class MCPServer（约 562 行）
```

阅读重点：
- `connect()` / `cleanup()` — 连接和断开的接口
- `list_tools()` — 获取工具列表
- `call_tool()` — 调用工具
- `list_prompts()` / `get_prompt()` — prompt 相关
- `list_resources()` / `read_resource()` — 资源相关
- `_normalize_needs_approval()` — 审批策略的标准化逻辑

### 2. MCPServerStdio 具体实现

```
src/agents/mcp/server.py  →  class MCPServerStdio（约 1962 行）
```

阅读重点：
- 构造函数的参数：`cache_tools_list`、`tool_filter`、`require_approval`、`max_retry_attempts`
- 怎么通过 `StdioServerParameters` 启动子进程
- 工具缓存机制：`cache_tools_list=True` 时的行为

### 3. 工具转换逻辑

```
src/agents/mcp/util.py  →  class MCPUtil
```

阅读重点：
- MCP 的 `MCPTool` 怎么转换成 SDK 的 `FunctionTool`
- 工具名的前缀和去重处理（多个 Server 可能有同名工具）
- `ToolFilter` 和 `ToolFilterContext` 的设计

### 4. 多 Server 管理

```
src/agents/mcp/manager.py  →  class MCPServerManager
```

阅读重点：
- `_ServerWorker`：每个 Server 一个独立的 asyncio task
- 并行连接和清理的实现
- 超时和错误处理

---

## 第三层：看 mcp 库源码，理解 Server 端实现（可选）

目标：理解 `@mcp.tool()` 装饰器背后发生了什么。

这是第三方库 `mcp`（`pip install mcp`），不在本仓库里。

### Server 端核心类

```python
from mcp.server.mcpserver import MCPServer   # Server 主类
from mcp.server.mcpserver import FastMCP     # 简化版（装饰器风格）
```

看 `MCPServer` 类的：
- `@mcp.tool()` 装饰器怎么注册工具
- `mcp.run()` 怎么启动不同 transport
- JSON-RPC 消息的处理流程

### Client 端核心类（SDK 内部使用的）

```python
from mcp import ClientSession       # MCP 会话管理
from mcp import stdio_client        # stdio transport 的 Client 端
from mcp.client.sse import sse_client  # SSE transport 的 Client 端
```

这部分是 SDK 的 `_MCPServerWithClientSession` 内部调用的，了解即可。

---

## 阅读检查清单

第一层（examples）看完后，应该能回答：
- [ ] 怎么用 `@mcp.tool()` 定义一个 MCP Server
- [ ] 三种 transport 的 Client 端代码有什么区别（MCPServerStdio / Sse / StreamableHttp）
- [ ] 怎么在 Agent 中使用 MCP Server
- [ ] 怎么用 tool_filter 过滤工具

第二层（SDK 源码）看完后，应该能回答：
- [ ] MCPTool 怎么变成 FunctionTool 的
- [ ] Agent 的 mcp_servers 在 Runner.run() 中是怎么被处理的
- [ ] 工具缓存（cache_tools_list）的实现原理
- [ ] 多个 Server 有同名工具时怎么处理

第三层（mcp 库）看完后，应该能回答：
- [ ] `@mcp.tool()` 装饰器内部做了什么
- [ ] Server 怎么处理 JSON-RPC 请求并路由到对应的工具函数
