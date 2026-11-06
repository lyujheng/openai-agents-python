# Week5 Day5 学习笔记：Function Calling vs MCP + 迁移实战

## 1. 架构对比

```
Week2 原生 Function Calling:
┌────────────────────────────────────┐
│           mini_agent.py            │
│  func_to_schema()  ← 自己写的      │
│  agent.run() loop  ← 自己写的      │
│  工具路由 dispatch  ← 自己写的      │
│  get_weather()     ← 直接调用      │
│  calculate()       ← 直接调用      │
│  get_exchange_rate()← 直接调用     │
└────────────────────────────────────┘
全在一个进程，紧耦合

Week5 MCP 版:
┌──────────────────┐                  ┌──────────────────────┐
│  Client (SDK)    │   JSON-RPC       │  MCP Server          │
│                  │ ←─────────────→  │  get_weather()       │
│  Agent + Runner  │   stdio/SSE      │  calculate()         │
│  （SDK 自动处理） │                  │  get_exchange_rate() │
└──────────────────┘                  └──────────────────────┘
跨进程，解耦
```

## 2. 逐项对比

| 维度 | Week2 mini_agent | Week5 MCP 版 |
|------|-----------------|--------------|
| 工具定义 | `def func()` + `func_to_schema()` 手动转 | `@mcp.tool()` 装饰器自动暴露 |
| Schema 生成 | 自己写的 `func_to_schema` | MCP SDK 自动生成 |
| 工具发现 | 静态 `agent.register_tool()` | 动态 `tools/list` RPC 调用 |
| 工具执行 | 同进程 `func(**args)` | 跨进程 `tools/call` RPC 调用 |
| 调度循环 | 自己写（~50 行 `agent.run`） | SDK `Runner.run()` 全自动 |
| 工具路由 | 自己写 `name → func` 映射 | MCP Server 自动路由 |
| 错误处理 | 自己写 try/except + 重试 | SDK + MCP 各层都有处理 |
| 代码量 | ~400 行（含 schema 生成、循环、路由） | Server ~80 行 + Client ~60 行 |
| 复用性 | 无（别的项目要 copy 全部代码） | Server 可被任何 MCP Client 使用 |

## 3. 迁移中消失的代码

Week2 自己写的这些，在 MCP 版中全部不需要了：

| 手写代码 | MCP 中谁来做 |
|---------|-------------|
| `func_to_schema()` ~60行 | MCP SDK `@mcp.tool()` 装饰器 |
| `MiniAgent.run()` loop ~50行 | Agents SDK `Runner.run()` |
| `call_llm_with_retry()` ~20行 | Agents SDK 内部重试 |
| JSON 解析容错 ~15行 | Agents SDK 自动处理 |
| 工具路由 dispatch ~10行 | MCP Server 自动路由 |

**总计减少 ~150 行自己写的基础设施代码**，只剩纯业务逻辑。

## 4. 什么时候用哪个

### 用原生 Function Calling
- 工具数量少（< 5个）
- 纯自定义逻辑，不需要复用
- 追求最低延迟（同进程调用 vs 跨进程 RPC）
- 学习理解 Agent 底层原理

### 用 MCP
- 工具多，想动态管理
- 需要跨项目/团队复用工具
- 想接入社区生态（filesystem、github、sqlite 等现成 Server）
- 需要安全隔离（工具在独立进程/容器中）
- 多 Agent 共享同一套工具

### 两者结合（实际生产中常见）
- 核心的、高频的自定义工具用原生 FC（低延迟）
- 通用的、标准化的工具用 MCP（复用生态）
- Agents SDK 本身就支持混用：`Agent(tools=[...], mcp_servers=[...])`

## 5. 代理 API 的 tool_calls 转换 bug

Day 4 实验发现：代理 API (zyapi.xmsxb.com) 把 Claude 的响应转成 OpenAI 格式时，
`tool_calls` 字段被丢失（返回 null），尽管 `finish_reason=tool_calls`。

这是代理 API 的转换问题，不是 MCP 或 FC 本身的问题。
在 stdio 模式下用 SDK 自动管理可能表现不同，因为 SDK 有自己的错误处理和重试逻辑。

## 6. 关键收获

1. **MCP 是 FC 的"标准化+工程化"封装**：底层还是 function calling，但加了协议层解决复用和隔离问题
2. **从零造轮子（Week2）到用标准协议（Week5）**：理解了为什么需要 MCP——不是因为 FC 不好，而是工程化需要标准
3. **SDK 屏蔽了大量复杂性**：schema 转换、工具发现、调度循环、错误处理都自动化了
4. **MCP 不是银弹**：多了进程管理和网络开销，简单场景反而增加复杂度
