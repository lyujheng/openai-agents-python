"""
Day 2 练习 2：用 Agents SDK 接入自己写的 MCP Server

演示 MCP 的完整链路：
  1. 启动 MCP Server（stdio transport）
  2. Agent 通过 MCP 协议发现工具
  3. LLM 自主决定调用哪个工具
  4. 工具执行结果通过 MCP 协议返回

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python client.py
"""

import asyncio
import os
import sys

from openai import AsyncOpenAI

from agents import Agent, Runner, RunConfig, ModelProvider, Model, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.mcp import MCPServerStdio


# --- 配置：使用中转站 API ---
API_BASE_URL = "https://zyapi.xmsxb.com/v1"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL_NAME = "mco-4"

if not API_KEY:
    print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
    sys.exit(1)

# 创建异步 OpenAI client，指向中转站。
client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# 关闭 tracing（不使用 OpenAI 平台的 trace 功能）。
set_tracing_disabled(True)


class CustomProvider(ModelProvider):
    """自定义 model provider，指向中转站 API。"""
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(
            model=model_name or MODEL_NAME,
            openai_client=client,
        )


async def main():
    # Server 脚本的路径。
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
    # sample_files 的绝对路径，方便告诉 Agent。
    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_files")

    # 通过 stdio transport 连接到我们自己写的 MCP Server。
    async with MCPServerStdio(
        name="File Tools Server",
        params={
            "command": sys.executable,  # 用当前 Python 解释器启动 Server。
            "args": [server_script],
        },
    ) as server:
        # 连接成功后，查看 Server 暴露了哪些工具。
        tools = await server.list_tools()
        print("=== MCP Server 暴露的工具 ===")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        print()

        # 创建 Agent，接入 MCP Server。
        agent = Agent(
            name="文件助手",
            instructions=(
                f"你是一个文件操作助手，可以读取和浏览文件。"
                f"文件目录的绝对路径是：{samples_dir}"
            ),
            mcp_servers=[server],
        )

        run_config = RunConfig(model_provider=CustomProvider(), model=MODEL_NAME)

        # --- 测试 1：列出目录 ---
        print("=== 测试 1：列出目录 ===")
        result = await Runner.run(
            starting_agent=agent,
            input=f"列出 {samples_dir} 目录下的所有文件",
            run_config=run_config,
        )
        print(f"Agent 回答：{result.final_output}\n")

        # --- 测试 2：读取文件内容 ---
        print("=== 测试 2：读取文件 ===")
        result = await Runner.run(
            starting_agent=agent,
            input=f"读取 {samples_dir}/note.txt 的内容，总结要点",
            run_config=run_config,
        )
        print(f"Agent 回答：{result.final_output}\n")

        # --- 测试 3：多步任务（需要调用多个工具） ---
        print("=== 测试 3：多步任务 ===")
        result = await Runner.run(
            starting_agent=agent,
            input=f"读取 {samples_dir}/scores.csv，告诉我谁的成绩最好，得了多少分",
            run_config=run_config,
        )
        print(f"Agent 回答：{result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())
