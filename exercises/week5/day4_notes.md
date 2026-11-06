# Week5 Day4 学习笔记：MCP 进阶

## 1. stdio 下的 stdin/stdout 占用

**只影响 MCP Server 进程，不影响 Client 进程。**

```
Client 进程                          Server 子进程
┌─────────┐                         ┌─────────┐
│         │ ── JSON-RPC 请求 ──→    │ stdin   │  ← 被 MCP 占用
│  SDK    │                         │         │
│         │ ←── JSON-RPC 响应 ──    │ stdout  │  ← 被 MCP 占用
│         │                         │         │
│ print() │ → 正常输出               │ stderr  │  ← 可正常使用
└─────────┘                         └─────────┘
```

- Server 进程里用 `print("debug", file=sys.stderr)` 或 logging 模块打日志
- SSE/HTTP 模式下没有这个限制，print() 随便用

## 2. 三种 Transport 对比

| 维度 | stdio | SSE | Streamable HTTP |
|------|-------|-----|-----------------|
| 通信方式 | 子进程 stdin/stdout | GET(事件流) + POST(请求) | 单一 HTTP 端点 |
| Server 生命周期 | Client 拉起/杀死子进程 | 独立运行的 HTTP 服务 | 独立运行 |
| 多 Client | 不支持（一对一管道） | 支持（session_id 区分） | 支持 |
| 网络 | 无需（进程间通信） | 需要 HTTP | 需要 HTTP |
| 适用场景 | 本地工具、开发调试 | 远程服务、共享工具 | 推荐新标准 |
| 可观测性 | 差（看不到管道内容） | 好（HTTP 日志清晰） | 好 |

### 核心区别

**stdio = "Server 是 Client 的子进程"**
- Client 启动时自动拉起 Server 子进程
- Client 退出时 Server 自动死亡
- 一对一绑定，不能共享

**SSE/HTTP = "Server 是独立的 HTTP 服务"**
- Server 需要单独启动
- 多个 Client 可以同时连接
- 通过 session_id 区分不同会话

## 3. SSE 实战要点

### Server 端

```python
from mcp.server.mcpserver import MCPServer  # mcp 2.0.0 API

mcp = MCPServer("my-server")

@mcp.tool()
def my_tool(arg: str) -> str:
    """工具描述"""
    return "result"

# 启动 SSE HTTP 服务
mcp.run(transport="sse", host="127.0.0.1", port=8080)
```

### Client 端

```python
from agents.mcp import MCPServerSse  # 注意：不是 MCPServerStdio

async with MCPServerSse(
    name="my-server",
    params={"url": "http://127.0.0.1:8080/sse"},  # 连接到 Server 的 SSE 端点
) as server:
    agent = Agent(mcp_servers=[server], ...)
```

### 运行方式

```bash
# 终端 1：启动 Server（独立进程）
uv run python sse_server.py

# 终端 2：运行 Client
uv run python sse_client.py
```

## 4. HTTP 通信细节（从 Server 日志观察）

```
GET /sse              → Client 建立事件流连接（长连接，Server 推送事件）
POST /messages/?session_id=xxx  → Client 发送 JSON-RPC 请求
                                   每个请求返回 202 Accepted
```

- `session_id` 是 Client 连接时自动分配的
- 所有 JSON-RPC 消息（initialize、tools/list、tools/call）都走 POST
- Server 的响应通过 SSE 事件流推回

## 5. mcp 2.0.0 API 变化

- 旧版：`from mcp.server.fastmcp import FastMCP`
- 新版：`from mcp.server.mcpserver import MCPServer`
- 装饰器 API 相同：`@mcp.tool()`
- `run()` 参数相同：`transport="stdio"` / `"sse"` / `"streamable-http"`

## 6. 安全边界回顾

MCP Server 返回的内容（工具描述、工具结果）都是**不可信数据**：

| 风险 | 场景 | 防御 |
|------|------|------|
| 恶意工具描述 | Server 的工具 description 中嵌入 prompt injection | 用 guardrail 检查工具返回内容 |
| 伪造结果 | Server 返回虚假数据 | 不要让 agent 盲目信任工具结果做关键决策 |
| 工具滥用 | 恶意 Server 暴露危险工具 | tool_filter 只允许白名单工具；require_approval 需要人工确认 |

SDK 的防御手段（Day 3 已实验过）：
- `tool_filter`：只暴露白名单工具给 agent
- `require_approval`：危险操作需人工确认（interruption 机制）
- `guardrails`：对 agent 输入/输出做安全检查
