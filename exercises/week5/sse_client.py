"""
Week5 Day4: SSE transport 客户端

与 Day2 的 stdio 客户端对比：
- stdio: MCPServerStdio(params={"command": "python", "args": ["server.py"]})
    → SDK 自动启动子进程，自动清理
- SSE:   MCPServerSse(params={"url": "http://127.0.0.1:8080/sse"})
    → Server 需要提前手动启动，Client 只负责连接

使用前先在另一个终端启动 SSE Server:
  uv run python exercises/week5/sse_server.py
"""

import asyncio
import os
import sys

from openai import AsyncOpenAI

from agents import Agent, Runner, set_tracing_disabled
from agents.mcp import MCPServerSse
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.run import RunConfig

from agents.model_settings import ModelSettings

# --- 配置 ---
API_BASE_URL = "https://zyapi.xmsxb.com/v1"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL_NAME = "mco-4"

client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
set_tracing_disabled(True)

SSE_URL = "http://127.0.0.1:8080/sse"


async def main():
    # 关键区别：MCPServerSse 只需要一个 URL，不管 Server 的生命周期
    async with MCPServerSse(
        name="file-tools-sse",
        params={"url": SSE_URL},
    ) as server:
        agent = Agent(
            name="File Assistant",
            instructions="你是一个文件助手。你必须使用提供的工具来完成任务，不要自己编造答案。回答请用中文。",
            mcp_servers=[server],
            model=OpenAIChatCompletionsModel(
                model=MODEL_NAME,
                openai_client=client,
            ),
            model_settings=ModelSettings(tool_choice="required"),
        )

        # 测试 1：列出目录
        print("=" * 50)
        print("测试 1：调试 MCP → Function Calling 转换")
        print("=" * 50)

        # 第一步：看 MCP 原始工具定义
        tools_list = await server.list_tools()
        print(f"\n[1] MCP 原始工具（{len(tools_list)} 个）:")
        for t in tools_list:
            schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
            print(f"  name={t.name}, description={t.description}")
            print(f"  input_schema={schema}\n")

        # 第二步：直接把 MCP schema 构造成 OpenAI tools 格式，发给 LLM
        import json
        openai_tools = []
        for t in tools_list:
            schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": schema,
                },
            })
        print(f"[2] 发送给 LLM 的 tools JSON:")
        print(json.dumps(openai_tools, indent=2, ensure_ascii=False))

        raw_resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个文件助手。你必须使用工具来完成任务。"},
                {"role": "user", "content": "请列出目录下有哪些文件"},
            ],
            tools=openai_tools,
            tool_choice="required",
        )
        msg = raw_resp.choices[0].message
        print(f"\n[3] LLM raw response:")
        print(f"  content={repr(msg.content)}")
        print(f"  tool_calls={msg.tool_calls}")
        print(f"  finish_reason={raw_resp.choices[0].finish_reason}")
        # 打印完整 raw JSON
        print(f"\n[3.1] 完整 response JSON:")
        print(raw_resp.model_dump_json(indent=2))

        # 测试 2：读取文件
        print("\n" + "=" * 50)
        print("测试 2：读取文件")
        print("=" * 50)
        result = await Runner.run(
            agent, "读取 note.txt 的内容",
            run_config=RunConfig(tracing_disabled=True),
        )
        print(result.final_output)

        print("\n" + "=" * 50)
        print("全部测试完成")
        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
