"""
练习 3：并行工具调用
目标：LLM 一次返回多个 tool_calls，你的代码并行执行，全部结果喂回 LLM。

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python ex3_parallel_tools.py
"""

import os
import json
from openai import OpenAI

client = OpenAI(
    base_url="https://zyapi.xmsxb.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY", "在这里填你的key"),
)

# --- 定义 3 个工具 ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "获取两种货币之间的汇率",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string", "description": "源货币，如 USD、CNY、JPY"},
                    "to_currency": {"type": "string", "description": "目标货币"},
                },
                "required": ["from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "获取指定主题的最新新闻",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "新闻主题"}
                },
                "required": ["topic"],
            },
        },
    },
]


# --- 工具实现（假数据） ---
def get_weather(city: str) -> str:
    fake_data = {
        "东京": "晴，28°C，湿度 65%",
        "北京": "多云，32°C，湿度 45%",
    }
    return fake_data.get(city, f"没有 {city} 的天气数据")


def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    rates = {
        ("JPY", "CNY"): "1 JPY = 0.052 CNY",
        ("USD", "CNY"): "1 USD = 7.24 CNY",
        ("CNY", "JPY"): "1 CNY = 19.2 JPY",
    }
    return rates.get(
        (from_currency.upper(), to_currency.upper()),
        f"没有 {from_currency} → {to_currency} 的汇率数据",
    )


def get_news(topic: str) -> str:
    return f"【{topic}】最新消息：相关领域本周有重要进展，多家机构发布了新的研究报告。"


# 工具路由表：函数名 → 函数对象
tool_map = {
    "get_weather": get_weather,
    "get_exchange_rate": get_exchange_rate,
    "get_news": get_news,
}

# --- 开始 ---
# 这个问题故意涉及多个工具，看 LLM 会不会一次调多个
user_question = "我下周要去东京旅游，帮我查一下东京天气、日元兑人民币汇率，还有东京旅游相关的新闻。"

print("=" * 60)
print(f"用户提问：{user_question}")
print("=" * 60)

messages = [{"role": "user", "content": user_question}]

# 第一次调 LLM
response = client.chat.completions.create(
    model="mco-4",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
assistant_message = response.choices[0].message

if assistant_message.tool_calls:
    print(f"\nLLM 一次要调 {len(assistant_message.tool_calls)} 个工具：")
    for i, tc in enumerate(assistant_message.tool_calls):
        args = json.loads(tc.function.arguments)
        print(f"  [{i + 1}] {tc.function.name}({args})")

    # 把 assistant 消息加入历史
    messages.append(assistant_message)

    # 执行所有工具调用
    print("\n--- 执行工具 ---")
    for tc in assistant_message.tool_calls:
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)

        # 从路由表找到对应函数并执行
        func = tool_map.get(func_name)
        if func:
            result = func(**func_args)
        else:
            result = f"未知工具: {func_name}"

        print(f"  {func_name}: {result}")

        # 每个工具调用都要单独喂回，各带各的 tool_call_id
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }
        )

    # 第二次调 LLM
    print(f"\n对话历史：{len(messages)} 条消息")
    print("\n--- LLM 最终回答 ---")
    response2 = client.chat.completions.create(
        model="mco-4",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print(response2.choices[0].message.content)
else:
    print("\nLLM 没调工具，直接回答了：")
    print(assistant_message.content)

print("\n" + "=" * 60)
print("关键理解：")
print("  1. tool_calls 是列表，LLM 可以一次请求调多个工具")
print("  2. 每个工具结果要单独作为一条 role='tool' 的消息喂回")
print("  3. 每条 tool 消息必须带对应的 tool_call_id，这样 LLM 知道哪个结果对应哪个调用")
print("  4. 你的代码需要一个路由表（tool_map）把函数名映射到真正的函数")
print("=" * 60)
