"""
Day 4 Streaming 实验：对比 Runner.run() 与 Runner.run_streamed()

目标：
1. 同一个 agent + 同一个任务，分别用两种方式跑
2. 观察输出时机的差异
3. 观察 stream_events() 的三种事件类型
"""

import asyncio
import os
import time

from agents import Agent, Runner
from agents.models.openai_provider import OpenAIProvider


# ---------- 工具 ----------
from agents import function_tool


@function_tool
def get_weather(city: str) -> str:
    """获取城市天气信息。"""
    weather_data = {
        "北京": "晴，气温 22°C",
        "上海": "多云，气温 25°C",
        "广州": "阵雨，气温 28°C",
    }
    return weather_data.get(city, f"{city}：暂无天气数据")


# ---------- Provider & Agent ----------
provider = OpenAIProvider(
    base_url="https://zyapi.xmsxb.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    use_responses=False,
)

agent = Agent(
    name="天气助手",
    instructions="你是一个天气助手，用户问天气时调用工具查询，然后给出简洁回答。",
    tools=[get_weather],
    model="mco-4",
)

TASK = "北京和上海今天天气怎么样？"


# ---------- 实验 1：非流式 ----------
async def run_normal():
    print("=" * 60)
    print("实验 1：Runner.run()（非流式）")
    print("=" * 60)

    start = time.time()
    result = await Runner.run(
        agent,
        TASK,
        run_config={"model_provider": provider},  # type: ignore[arg-type]
    )
    elapsed = time.time() - start

    print(f"最终回答：{result.final_output}")
    print(f"耗时：{elapsed:.2f}s（全部完成后才返回）")
    print()


# ---------- 实验 2：流式 ----------
async def run_streamed():
    print("=" * 60)
    print("实验 2：Runner.run_streamed()（流式）")
    print("=" * 60)

    start = time.time()
    result = Runner.run_streamed(
        agent,
        TASK,
        run_config={"model_provider": provider},  # type: ignore[arg-type]
    )

    first_token_time = None
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            # SDK 把 Chat Completions chunk 转成了 Responses API 事件格式。
            # 文本片段在 ResponseTextDeltaEvent 中，delta 直接是字符串。
            if event.data.type == "response.output_text.delta":
                if first_token_time is None:
                    first_token_time = time.time() - start
                    print(f"[首 token 延迟：{first_token_time:.2f}s]")
                print(event.data.delta, end="", flush=True)

            elif event.data.type == "response.function_call_arguments.delta":
                # 工具调用参数也是逐字推送的。
                print(event.data.delta, end="", flush=True)

            elif event.data.type == "response.completed":
                # 一次 LLM 调用完成，打印 token 用量。
                usage = getattr(event.data.response, "usage", None)
                if usage:
                    print(f"\n  [Usage] input={usage.input_tokens} output={usage.output_tokens}")

        elif event.type == "run_item_stream_event":
            # SDK 加工后的语义事件：工具调用、工具输出、消息完成等。
            print(f"\n  [RunItem] {event.name}: {type(event.item).__name__}")

        elif event.type == "agent_updated_stream_event":
            # Agent 切换（handoff 场景才会出现）。
            print(f"\n  [AgentUpdated] 切换到: {event.new_agent.name}")

    elapsed = time.time() - start
    print(f"\n最终回答：{result.final_output}")
    print(f"耗时：{elapsed:.2f}s")
    print()


# ---------- 入口 ----------
async def main():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("请先设置 DEEPSEEK_API_KEY 环境变量")
        return

    await run_normal()
    await run_streamed()


if __name__ == "__main__":
    asyncio.run(main())
