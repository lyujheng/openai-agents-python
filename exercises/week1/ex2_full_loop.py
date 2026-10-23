"""
练习 2：多轮 function calling —— 完整循环
目标：走通"用户提问 → LLM 决定调工具 → 你执行工具 → 结果喂回 LLM → LLM 生成最终回答"

对比练习 1：练习 1 停在了"LLM 说我要调工具"，这次我们把后半段也接上。

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python ex2_full_loop.py
"""

import os
import json
from openai import OpenAI

client = OpenAI(
    base_url="https://zyapi.xmsxb.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY", "在这里填你的key"),
)

# --- 工具定义（和练习 1 一样） ---
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
                        "description": "城市名",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


# --- 这是你写的真实工具函数 ---
# 在真实场景中，这里会调天气 API。现在我们用假数据，重点是理解流程。
def get_weather(city: str) -> str:
    """模拟天气查询。真实场景会调 API。"""
    fake_data = {
        "东京": "晴，28°C，湿度 65%",
        "北京": "多云，32°C，湿度 45%",
        "纽约": "小雨，22°C，湿度 80%",
    }
    return fake_data.get(city, f"抱歉，没有 {city} 的天气数据")


# --- 开始完整循环 ---
print("=" * 60)
print("第一轮：用户提问，LLM 决定要不要调工具")
print("=" * 60)

messages = [{"role": "user", "content": "东京今天天气怎么样？"}]

# 第一次调 LLM
response = client.chat.completions.create(
    model="mco-4",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
assistant_message = response.choices[0].message

print(f"\nLLM 的决策：")
print(f"  content:    {assistant_message.content!r}")
print(f"  tool_calls: 要调 {len(assistant_message.tool_calls)} 个工具")

# --- 关键步骤：执行工具，把结果喂回去 ---
print("\n" + "=" * 60)
print("第二轮：你执行工具，把结果喂回 LLM")
print("=" * 60)

# 1. 先把 LLM 的消息原样加入对话历史（这一步很多人会漏掉）
messages.append(assistant_message)

# 2. 执行每个工具调用，把结果追加到对话历史
for tool_call in assistant_message.tool_calls:
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)

    print(f"\n  执行工具: {func_name}({func_args})")

    # 路由到对应函数
    if func_name == "get_weather":
        result = get_weather(**func_args)
    else:
        result = f"未知工具: {func_name}"

    print(f"  执行结果: {result}")

    # 3. 以 role="tool" 的消息追加结果，必须带上 tool_call_id
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,  # 这个 id 对应 LLM 发出的那个调用
            "content": result,
        }
    )

# --- 第二次调 LLM，这次它有了工具结果，会生成最终回答 ---
print(f"\n  现在对话历史里有 {len(messages)} 条消息：")
for i, msg in enumerate(messages):
    role = msg["role"] if isinstance(msg, dict) else msg.role
    print(f"    [{i}] {role}")

print("\n" + "=" * 60)
print("第三轮：LLM 拿到工具结果，生成最终回答")
print("=" * 60)

response2 = client.chat.completions.create(
    model="mco-4",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
final_message = response2.choices[0].message

print(f"\nLLM 最终回答：{final_message.content}")
print(f"还要调工具吗：{final_message.tool_calls}")

# --- 总结 ---
print("\n" + "=" * 60)
print("完整流程回顾：")
print()
print("  用户: '东京今天天气怎么样？'")
print("    ↓")
print("  LLM:  '我需要调 get_weather(东京)'        ← LLM 做决策")
print("    ↓")
print("  你的代码: 执行 get_weather('东京')          ← 你的代码执行")
print("    ↓")
print("  你的代码: 把 '晴，28°C...' 喂回 LLM        ← 结果回传")
print("    ↓")
print(f"  LLM:  '{final_message.content}'  ← LLM 用结果回答用户")
print()
print("这个循环就是 agent 的核心。")
print("第2周你要做的事：把这个循环放进 while 里，让 LLM 自己决定")
print("什么时候调工具、调哪个、调几次、什么时候停下来。")
print("=" * 60)
