"""
Week5 Day5: mini_agent MCP 版客户端

对比 Week2 的 mini_agent.py：
- 原来：自己写 agent loop（LLM 调用 → 解析 tool_calls → 执行 → 喂回）
- 现在：SDK 的 Runner.run() 全自动处理，你只需要声明 agent + mcp_servers

架构变化：
  Week2: [应用代码] = schema生成 + 工具注册 + 调度循环 + 工具执行（全在一个文件）
  Week5: [Client] → JSON-RPC → [MCP Server]（工具定义和执行在 Server 侧）

运行方式：
  uv run python exercises/week5/mini_agent_mcp_client.py

  不需要手动启动 Server——MCPServerStdio 会自动拉起子进程。
"""

import asyncio
import os
import sys

from openai import AsyncOpenAI

from agents import Agent, Runner, set_tracing_disabled
from agents.mcp import MCPServerStdio
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.run import RunConfig

# --- 配置 ---
API_BASE_URL = "https://zyapi.xmsxb.com/v1"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL_NAME = "mco-4"

client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
set_tracing_disabled(True)


async def main():
    # 关键区别：用 MCPServerStdio，SDK 自动启动 Server 子进程
    # 对比 Week2：不需要 func_to_schema()，不需要 agent loop，不需要工具路由
    server_path = os.path.join(os.path.dirname(__file__), "mini_agent_mcp_server.py")

    async with MCPServerStdio(
        name="mini-agent-tools",
        params={
            "command": "uv",
            "args": ["run", "python", server_path],
        },
    ) as server:
        # 先看看 MCP Server 暴露了哪些工具
        tools = await server.list_tools()
        print(f"MCP Server 暴露了 {len(tools)} 个工具:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")
        print()

        agent = Agent(
            name="Mini Agent (MCP版)",
            instructions=(
                "你是一个实用助手。可以查天气、做计算、查汇率。"
                "请使用提供的工具来回答问题，不要编造数据。回答用中文。"
            ),
            mcp_servers=[server],
            model=OpenAIChatCompletionsModel(
                model=MODEL_NAME,
                openai_client=client,
            ),
        )

        config = RunConfig(tracing_disabled=True)

        # 复用 Week2 的三个测试用例
        tests = [
            ("单工具 — 东京天气", "东京今天天气怎么样？"),
            ("多步推理 — 汇率换算", "我有 1000 美元，换成人民币是多少？"),
            ("无需工具 — 直接回答", "你好，你是谁？"),
        ]

        for title, query in tests:
            print("=" * 50)
            print(f"测试: {title}")
            print(f"问题: {query}")
            print("=" * 50)
            try:
                result = await Runner.run(agent, query, run_config=config)
                print(f"回答: {result.final_output}")
            except Exception as e:
                print(f"错误: {e}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
