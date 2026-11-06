"""
Day 3 练习：多 Server 组合 + 工具过滤 + 审批机制

演示内容：
  1. 多 Server 组合 — 文件 Server + 计算 Server 同时接入
  2. 工具过滤 — 限制文件 Server 只暴露 read_file 和 list_dir
  3. 工具审批 — 计算 Server 的 calculate 需要审批（演示 Human-in-the-loop）
  4. 跨 Server 任务 — 读取文件中的数据，用计算工具分析

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python day3_multi_server.py
"""

import asyncio
import os
import sys

from openai import AsyncOpenAI

from agents import Agent, Runner, RunConfig, ModelProvider, Model, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.mcp import MCPServerStdio
from agents.mcp.util import create_static_tool_filter


# --- API 配置 ---
API_BASE_URL = "https://zyapi.xmsxb.com/v1"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL_NAME = "mco-4"

if not API_KEY:
    print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
    sys.exit(1)

client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
set_tracing_disabled(True)


class CustomProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(
            model=model_name or MODEL_NAME,
            openai_client=client,
        )


# ---------------------------------------------------------------------------
# 练习 1：多 Server 组合
# ---------------------------------------------------------------------------
async def exercise1_multi_server():
    """两个 Server 同时接入，LLM 自主选择工具。"""
    print("=" * 60)
    print("练习 1：多 Server 组合")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(current_dir, "sample_files")

    async with (
        MCPServerStdio(
            name="File Tools",
            params={"command": sys.executable, "args": [os.path.join(current_dir, "server.py")]},
        ) as file_server,
        MCPServerStdio(
            name="Calc Tools",
            params={"command": sys.executable, "args": [os.path.join(current_dir, "calc_server.py")]},
        ) as calc_server,
    ):
        # 查看两个 Server 暴露的工具。
        file_tools = await file_server.list_tools()
        calc_tools = await calc_server.list_tools()
        print(f"\nFile Server 工具 ({len(file_tools)} 个)：")
        for t in file_tools:
            print(f"  - {t.name}")
        print(f"\nCalc Server 工具 ({len(calc_tools)} 个)：")
        for t in calc_tools:
            print(f"  - {t.name}")

        agent = Agent(
            name="全能助手",
            instructions=(
                f"你是一个全能助手，可以操作文件和做数学计算。"
                f"文件目录的绝对路径是：{samples_dir}"
            ),
            mcp_servers=[file_server, calc_server],
        )
        run_config = RunConfig(model_provider=CustomProvider(), model=MODEL_NAME)

        # 测试 1：只用文件工具。
        print("\n--- 测试 1：纯文件任务 ---")
        result = await Runner.run(
            agent, f"列出 {samples_dir} 目录下的文件", run_config=run_config,
        )
        print(f"回答：{result.final_output}\n")

        # 测试 2：只用计算工具。
        print("--- 测试 2：纯计算任务 ---")
        result = await Runner.run(
            agent, "计算 sqrt(144) + 3.14 * 2", run_config=run_config,
        )
        print(f"回答：{result.final_output}\n")

        # 测试 3：跨 Server 任务——读文件 + 计算。
        print("--- 测试 3：跨 Server 任务（读文件 + 计算） ---")
        result = await Runner.run(
            agent,
            f"读取 {samples_dir}/scores.csv，提取所有人的分数，用 analyze_numbers 做统计分析",
            run_config=run_config,
        )
        print(f"回答：{result.final_output}\n")


# ---------------------------------------------------------------------------
# 练习 2：工具过滤
# ---------------------------------------------------------------------------
async def exercise2_tool_filter():
    """限制 Agent 只能使用特定工具。"""
    print("=" * 60)
    print("练习 2：工具过滤")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(current_dir, "sample_files")

    async with MCPServerStdio(
        name="Filtered File Tools",
        params={"command": sys.executable, "args": [os.path.join(current_dir, "server.py")]},
        # 只允许 read_file 和 list_dir，屏蔽 get_file_info。
        tool_filter=create_static_tool_filter(
            allowed_tool_names=["read_file", "list_dir"],
        ),
    ) as server:
        # 查看过滤后的工具列表。
        tools = await server.list_tools()
        print(f"\n过滤后的工具 ({len(tools)} 个)：")
        for t in tools:
            print(f"  - {t.name}")
        print("  (get_file_info 被过滤掉了)\n")

        agent = Agent(
            name="只读助手",
            instructions=(
                f"你是文件浏览助手，只能读取和列出文件。"
                f"文件目录的绝对路径是：{samples_dir}"
                f"如果用户请求你无法完成的操作，请说明你没有对应的工具。"
            ),
            mcp_servers=[server],
        )
        run_config = RunConfig(model_provider=CustomProvider(), model=MODEL_NAME)

        # 测试：请求被过滤掉的工具功能。
        print("--- 测试：请求获取文件详细信息（工具已被过滤） ---")
        result = await Runner.run(
            agent,
            f"获取 {samples_dir}/note.txt 的详细信息（大小、修改时间）",
            run_config=run_config,
        )
        print(f"回答：{result.final_output}\n")


# ---------------------------------------------------------------------------
# 练习 3：工具审批（Human-in-the-loop）
# ---------------------------------------------------------------------------
async def exercise3_approval():
    """危险操作需要人工审批。"""
    print("=" * 60)
    print("练习 3：工具审批")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    async with MCPServerStdio(
        name="Calc Tools (需审批)",
        params={"command": sys.executable, "args": [os.path.join(current_dir, "calc_server.py")]},
        # 所有工具都需要审批。
        require_approval="always",
    ) as server:
        agent = Agent(
            name="计算助手",
            instructions="你是计算助手。",
            mcp_servers=[server],
        )
        run_config = RunConfig(model_provider=CustomProvider(), model=MODEL_NAME)

        print("\n--- 测试：计算任务（需要审批） ---")
        result = await Runner.run(agent, "计算 2 的 10 次方", run_config=run_config)

        if result.interruptions:
            print(f"收到 {len(result.interruptions)} 个审批请求：")
            for interruption in result.interruptions:
                print(f"  工具: {interruption.name}")
                print(f"  → 选择：自动批准")

            # 批准所有中断，继续执行。
            state = result.to_state()
            for interruption in result.interruptions:
                state.approve(interruption, always_approve=True)
            result = await Runner.run(agent, state, run_config=run_config)

        print(f"回答：{result.final_output}\n")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
async def main():
    await exercise1_multi_server()
    await exercise2_tool_filter()
    await exercise3_approval()

    print("=" * 60)
    print("Day 3 练习完成！")
    print("=" * 60)
    print("""
总结：
  1. 多 Server 组合：一个 Agent 同时接入多个 Server，LLM 自主选择工具
  2. 工具过滤：tool_filter 控制 LLM 能看到哪些工具
  3. 工具审批：require_approval 控制哪些工具需要人工确认
  4. 跨 Server 任务：LLM 能协调使用不同 Server 的工具完成复杂任务
""")


if __name__ == "__main__":
    asyncio.run(main())
