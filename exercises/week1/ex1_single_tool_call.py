"""
练习 1：单轮 function calling
目标：理解 LLM 是怎么"调用工具"的——它并不真的执行函数，而是返回一个结构化的调用请求。

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    python ex1_single_tool_call.py
"""

import os
import json
from openai import OpenAI

# --- 第一步：创建 client，指向 DeepSeek API ---
client = OpenAI(
    base_url="https://zyapi.xmsxb.com/v1",
    # base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY", "在这里填你的key，或者用环境变量"),
)

# --- 第二步：定义工具的 JSON Schema ---
# 这就是告诉 LLM "你有一个叫 get_weather 的工具可以用"
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，比如 北京、东京、New York",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

# --- 第三步：发请求，让 LLM 决定要不要调工具 ---
print("=" * 60)
print("发送请求：'东京今天天气怎么样？'")
print("=" * 60)

response = client.chat.completions.create(
    model="mco-4",
    messages=[{"role": "user", "content": "东京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto",  # auto = LLM 自己决定要不要调工具
)

# 先看看完整响应长什么样
print("\n--- 完整 response 对象 ---")
print(response)
print(f"\nchoices 数量: {len(response.choices) if response.choices else 'None/空'}")

message = response.choices[0].message

# --- 第四步：观察返回结果 ---
print("\n--- 完整的 message 对象 ---")
print(message)

print("\n--- 逐字段分析 ---")
print(f"role:        {message.role}")
print(f"content:     {message.content}")  # 调工具时通常为 None
print(f"tool_calls:  {message.tool_calls}")

if message.tool_calls:
    print("\n--- LLM 要求调用工具！拆解看看 ---")
    for i, tool_call in enumerate(message.tool_calls):
        print(f"\n  工具调用 #{i + 1}:")
        print(f"    id:        {tool_call.id}")  # 这个 id 后面喂回结果时要用
        print(f"    type:      {tool_call.type}")
        print(f"    函数名:    {tool_call.function.name}")
        print(f"    参数(str): {tool_call.function.arguments}")

        # arguments 是 JSON 字符串，需要解析
        args = json.loads(tool_call.function.arguments)
        print(f"    参数(obj): {args}")
        print(f"    city 值:   {args.get('city')}")
else:
    print("\nLLM 没有调用工具，直接回复了：")
    print(message.content)

print("\n" + "=" * 60)
print("关键理解：")
print("  1. LLM 并没有真的去查天气，它只是说'我想调 get_weather 这个函数'")
print("  2. tool_calls 是一个列表（因为 LLM 可能一次想调多个工具）")
print("  3. arguments 是 JSON 字符串，不是 dict，需要 json.loads 解析")
print("  4. 每个 tool_call 有一个 id，后面喂回结果时必须带上")
print("=" * 60)
