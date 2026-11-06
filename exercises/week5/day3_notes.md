# Day 3：MCP 进阶实践笔记

## 练习内容

### 练习 1：多 Server 组合

一个 Agent 同时接入文件 Server 和计算 Server，LLM 自主选择工具。

关键代码：
```python
async with (
    MCPServerStdio(name="File Tools", params={...}) as file_server,
    MCPServerStdio(name="Calc Tools", params={...}) as calc_server,
):
    agent = Agent(mcp_servers=[file_server, calc_server])
```

验证结果：
- 纯文件任务 → LLM 只用 File Server 的工具
- 纯计算任务 → LLM 只用 Calc Server 的工具
- 跨 Server 任务 → LLM 先用 read_file 读数据，再用 analyze_numbers 分析
- LLM 自主判断用哪个 Server 的哪个工具

### 练习 2：工具过滤

用 `tool_filter` 限制 Agent 能看到的工具：

```python
MCPServerStdio(
    params={...},
    tool_filter=create_static_tool_filter(
        allowed_tool_names=["read_file", "list_dir"],
    ),
)
```

验证结果：
- `get_file_info` 被过滤后，LLM 无法看到也无法调用
- Agent 诚实告知用户没有对应工具

### 练习 3：工具审批

用 `require_approval` 实现 Human-in-the-loop：

```python
MCPServerStdio(
    params={...},
    require_approval="always",
)
```

验证结果：
- 工具调用产生 interruption，Runner.run() 返回但任务未完成
- 调用 state.approve() 批准后，继续 Runner.run() 完成任务

## 关键知识点

### 多 Server 时工具名冲突处理

多个 Server 有同名工具时，SDK 会自动添加前缀：`mcp_{server_name}__{tool_name}`

### tool_filter vs require_approval

| | tool_filter | require_approval |
|--|-------------|-----------------|
| 控制点 | LLM 能看到什么工具 | 看到的工具能不能直接执行 |
| 效果 | 工具从列表中消失 | 工具可见但调用产生中断 |
| 适用场景 | 彻底屏蔽不需要的工具 | 危险操作需要人工确认 |

### MCPServerManager vs async with

| | async with | MCPServerManager |
|--|-----------|-----------------|
| 连接 | 逐个 | 并行 |
| 清理 | 自动 | 手动调 cleanup_all() |
| 适用 | Server 少、生命周期简单 | Server 多、需要动态管理 |

### stdio transport 注意事项

- Server 不能用 `print()` 输出调试信息，stdout 是 MCP 通信通道
- 调试信息应写到 stderr 或用 logging
