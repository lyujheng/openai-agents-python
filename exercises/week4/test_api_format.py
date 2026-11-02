"""
快速探测：中转站 API 对 response_format 的支持情况。

测试三种模式：
  1. 无 response_format
  2. response_format={"type": "json_object"}   (JSON Mode)
  3. response_format={"type": "json_schema", ...}  (Strict Schema)

运行：
    export DEEPSEEK_API_KEY="你的key"
    uv run python test_api_format.py
"""

import json
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://zyapi.xmsxb.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
)
MODEL = "mco-4"

PROMPT = "用 JSON 返回：{\"city\": \"东京\", \"temp\": 28, \"unit\": \"celsius\"}。只返回 JSON。"


def test_no_format():
    """测试 1：不加 response_format。"""
    print("=" * 50)
    print("测试 1：无 response_format")
    print("=" * 50)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
        )
        content = resp.choices[0].message.content or ""
        print(f"原始输出:\n{content}\n")
        # 尝试解析。
        try:
            parsed = json.loads(content)
            print(f"✅ JSON 解析成功: {parsed}")
        except json.JSONDecodeError:
            print("❌ JSON 解析失败（输出不是纯 JSON）")
    except Exception as e:
        print(f"❌ API 报错: {type(e).__name__}: {e}")


def test_json_mode():
    """测试 2：JSON Mode。"""
    print("\n" + "=" * 50)
    print("测试 2：response_format={\"type\": \"json_object\"}")
    print("=" * 50)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        print(f"原始输出:\n{content}\n")

        # 先试直接解析。
        try:
            parsed = json.loads(content)
            print(f"✅ 直接解析成功: {parsed}")
        except json.JSONDecodeError:
            # 可能有 markdown 包装，尝试清洗。
            import re
            match = re.match(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content.strip(), re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
                try:
                    parsed = json.loads(cleaned)
                    print(f"⚠️ 需要去掉 markdown 包装后解析成功: {parsed}")
                except json.JSONDecodeError:
                    print("❌ 去掉 markdown 包装后仍然解析失败")
            else:
                print("❌ JSON 解析失败，也没有 markdown 包装")
    except Exception as e:
        print(f"❌ API 报错: {type(e).__name__}: {e}")


def test_json_schema():
    """测试 3：Strict JSON Schema。"""
    print("\n" + "=" * 50)
    print("测试 3：response_format={\"type\": \"json_schema\", ...}")
    print("=" * 50)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "返回东京的天气信息。"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "temp": {"type": "number"},
                            "unit": {"type": "string"},
                        },
                        "required": ["city", "temp", "unit"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        content = resp.choices[0].message.content or ""
        print(f"原始输出:\n{content}\n")
        try:
            parsed = json.loads(content)
            print(f"✅ 解析成功: {parsed}")
            # 检查字段是否严格匹配。
            expected_keys = {"city", "temp", "unit"}
            actual_keys = set(parsed.keys())
            if actual_keys == expected_keys:
                print("✅ 字段完全匹配 schema 定义")
            else:
                extra = actual_keys - expected_keys
                missing = expected_keys - actual_keys
                if extra:
                    print(f"⚠️ 多了字段: {extra}（说明 strict 没生效）")
                if missing:
                    print(f"⚠️ 缺少字段: {missing}")
        except json.JSONDecodeError:
            print("❌ JSON 解析失败")
    except Exception as e:
        print(f"❌ API 报错: {type(e).__name__}: {e}")
        print("   → 如果报 400/422 错误，说明你的 API 不支持 json_schema 模式")


if __name__ == "__main__":
    print("探测 API 的 response_format 支持情况\n")
    test_no_format()
    test_json_mode()
    test_json_schema()

    print("\n" + "=" * 50)
    print("结论")
    print("=" * 50)
    print("根据上面的输出判断：")
    print("  - 测试 2 成功 → 你的 API 支持 JSON Mode")
    print("  - 测试 2 需要清洗 → JSON Mode 可用，但要去 markdown 包装")
    print("  - 测试 3 成功且字段匹配 → 支持 Strict Schema（最理想）")
    print("  - 测试 3 报错或字段不匹配 → 不支持 Strict Schema（预期情况）")
