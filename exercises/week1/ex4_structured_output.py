"""
练习 4：Structured Output
目标：让 LLM 的输出严格遵守你定义的 JSON 格式，而不是自由发挥。

为什么需要：
  练习 2、3 中 LLM 的最终回答是自由文本，格式不可控。
  但在 agent 场景中，你经常需要 LLM 返回结构化数据（比如一个 JSON 对象），
  方便你的代码解析和处理。如果 LLM 返回的格式不对，代码就崩了。

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python ex4_structured_output.py
"""

import os
import json
from openai import OpenAI

client = OpenAI(
    base_url="https://zyapi.xmsxb.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY", "在这里填你的key"),
)

# ===================================================================
# 实验 1：不加任何约束，让 LLM 自由输出
# ===================================================================
print("=" * 60)
print("实验 1：无约束 — LLM 自由输出")
print("=" * 60)

response1 = client.chat.completions.create(
    model="mco-4",
    messages=[
        {
            "role": "user",
            "content": "分析一下 Python 这门编程语言，给出名称、适用场景、优缺点。",
        }
    ],
)
print(response1.choices[0].message.content)
print("\n→ 输出是自由文本，格式不确定。你的代码很难可靠地从中提取结构化信息。")

# ===================================================================
# 实验 2：JSON Mode — 告诉 LLM "请返回 JSON"
# ===================================================================
print("\n" + "=" * 60)
print("实验 2：JSON Mode — 要求返回 JSON（但不限定具体结构）")
print("=" * 60)

response2 = client.chat.completions.create(
    model="mco-4",
    messages=[
        {
            "role": "user",
            "content": "分析一下 Python 这门编程语言，给出名称、适用场景、优缺点。请用 JSON 格式返回。",
        }
    ],
    response_format={"type": "json_object"},
)
content2 = response2.choices[0].message.content
print(content2)

# 尝试解析
try:
    parsed = json.loads(content2)
    print(f"\n→ 成功解析为 JSON，字段有：{list(parsed.keys())}")
    print("→ 但注意：字段名和结构完全由 LLM 自己决定，每次可能不一样！")
except json.JSONDecodeError:
    print("\n→ JSON 解析失败！这就是 JSON Mode 不够可靠的地方。")

# ===================================================================
# 实验 3：JSON Schema — 严格限定输出结构
# ===================================================================
print("\n" + "=" * 60)
print("实验 3：JSON Schema — 严格限定输出的字段和类型")
print("=" * 60)

# 定义你期望的输出结构
output_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "language_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "编程语言名称",
                },
                "use_cases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "适用场景列表",
                },
                "pros": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "优点列表",
                },
                "cons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "缺点列表",
                },
                "rating": {
                    "type": "number",
                    "description": "综合评分 1-10",
                },
            },
            "required": ["name", "use_cases", "pros", "cons", "rating"],
            "additionalProperties": False,
        },
    },
}

response3 = client.chat.completions.create(
    model="mco-4",
    messages=[
        {
            "role": "user",
            "content": "分析一下 Python 这门编程语言。",
        }
    ],
    response_format=output_schema,
)
content3 = response3.choices[0].message.content
print(content3)

try:
    parsed3 = json.loads(content3)
    print(f"\n→ 解析成功！")
    print(f"   name:      {parsed3['name']}")
    print(f"   use_cases: {parsed3['use_cases']}")
    print(f"   pros:      {parsed3['pros'][:2]}...")  # 只打印前两个
    print(f"   cons:      {parsed3['cons'][:2]}...")
    print(f"   rating:    {parsed3['rating']}")
    print(f"\n→ 字段名、类型、结构完全由你控制，每次都一样。代码可以放心地用 parsed3['name'] 取值。")
except json.JSONDecodeError:
    print("\n→ JSON 解析失败。可能模型不支持 strict JSON Schema。")

# ===================================================================
# 总结
# ===================================================================
print("\n" + "=" * 60)
print("三种方式对比：")
print()
print("  无约束         → 自由文本，格式不可控")
print("  JSON Mode      → 保证是合法 JSON，但字段结构不确定")
print("  JSON Schema    → 严格限定字段名、类型、结构，100% 可解析")
print()
print("在 agent 场景中的用途：")
print("  - 工具参数解析：LLM 返回的 function.arguments 本质上就是 JSON Schema 约束的")
print("  - agent 最终输出：比如让 agent 输出一份结构化报告，用 JSON Schema 保证格式")
print("  - 决策输出：让 LLM 输出 {\"action\": \"search\", \"query\": \"xxx\"}，代码可靠地路由")
print("=" * 60)
