"""
Week5 Day4: SSE 连通性测试（不需要 LLM API key）

直接用 MCP 协议层验证 SSE Server 是否工作：
- 连接到 SSE Server
- 列出可用工具
- 直接调用工具（不经过 LLM）

先启动 Server:
  uv run python exercises/week5/sse_server.py
"""

import asyncio

from agents.mcp import MCPServerSse


async def main():
    print("连接到 SSE Server...")

    async with MCPServerSse(
        name="file-tools-sse",
        params={"url": "http://127.0.0.1:8080/sse"},
    ) as server:
        # 1. 列出所有工具
        print("\n=== 可用工具 ===")
        tools = await server.list_tools()
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
            if schema and "properties" in schema:
                for param, param_schema in schema["properties"].items():
                    print(f"      参数 {param}: {param_schema}")

        # 2. 直接调用工具（绕过 LLM，直接测 MCP 通信）
        print("\n=== 调用 list_dir ===")
        result = await server.call_tool("list_dir", {"subdir": "."})
        print(f"  结果: {result}")

        print("\n=== 调用 read_file ===")
        result = await server.call_tool("read_file", {"filename": "note.txt"})
        print(f"  结果: {result}")

        print("\n=== 调用 get_file_info ===")
        result = await server.call_tool("get_file_info", {"filename": "scores.csv"})
        print(f"  结果: {result}")

    print("\n✅ SSE 连通性测试通过！")


if __name__ == "__main__":
    asyncio.run(main())
