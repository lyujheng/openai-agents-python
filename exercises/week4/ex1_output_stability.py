"""
第4周 Day 1：输出稳定性对比实验

核心问题：LLM 输出是不可控的自然语言，但程序需要确定性的结构化数据。
三种约束手段从弱到强：
  A. Prompt 约束 — 靠模型自觉，最不可靠
  B. JSON Mode — 保证合法 JSON，但字段不保证
  C. Pydantic 后处理 — prompt 约束 + 解析校验 + 重试

实验结论：强模型（Claude）+ 简单结构下三种方式差异不大，都能 5/5。
约束手段的价值和模型能力成反比，Week 7 用 vLLM 本地小模型时差异会很明显。
即使强模型不出错，Pydantic 校验仍然是生产环境的安全网。

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python ex1_output_stability.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url="https://zyapi.xmsxb.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
)
MODEL = "mco-4"
RUNS_PER_EXPERIMENT = 5  # 每个实验重复次数，观察稳定性。


# ---------------------------------------------------------------------------
# 目标输出结构：用 Pydantic 定义"我们想要什么"
# ---------------------------------------------------------------------------

class BookAnalysis(BaseModel):
    """一本书的分析结果。这是我们期望 LLM 返回的结构。"""
    title: str
    author: str
    genre: str
    rating: float  # 1.0-10.0
    summary: str
    themes: list[str]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def clean_json_from_markdown(text: str) -> str:
    """
    清洗 LLM 返回中的 markdown 代码块包装。

    Week 1 练习 4 的经验：很多模型会把 JSON 包在 ```json ... ``` 里。
    """
    text = text.strip()
    # 去掉 ```json ... ``` 或 ``` ... ``` 包装。
    match = re.match(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


@dataclass
class ExperimentResult:
    """单次实验的结果。"""
    run_id: int
    success: bool
    raw_output: str
    parsed: Any = None
    error: str | None = None


def print_results(name: str, results: list[ExperimentResult]):
    """打印实验结果汇总。"""
    success_count = sum(1 for r in results if r.success)
    print(f"\n{'=' * 60}")
    print(f"实验结果：{name}")
    print(f"成功率：{success_count}/{len(results)}")
    print(f"{'=' * 60}")
    for r in results:
        status = "✅" if r.success else "❌"
        if r.success and r.parsed:
            # 通用打印：取前几个字段预览。
            if isinstance(r.parsed, BaseModel):
                fields = r.parsed.model_dump()
                preview_parts = []
                for k, v in list(fields.items())[:3]:
                    v_str = str(v)
                    if len(v_str) > 40:
                        v_str = v_str[:40] + "..."
                    preview_parts.append(f"{k}={v_str}")
                print(f"  {status} Run {r.run_id}: {', '.join(preview_parts)}")
            else:
                print(f"  {status} Run {r.run_id}: {str(r.parsed)[:100]}")
        else:
            print(f"  {status} Run {r.run_id}: {r.error}")
            # 打印原始输出的前 200 字符，方便调试。
            preview = r.raw_output[:200].replace("\n", "\\n")
            print(f"       原始输出: {preview}")


# ---------------------------------------------------------------------------
# 实验 A：纯 Prompt 约束（最弱）
# ---------------------------------------------------------------------------
# 只在 prompt 里说"请返回 JSON"，不加任何技术手段。
# 预期：LLM 可能返回纯文本、多余解释、格式不一致。

def experiment_a_prompt_only() -> list[ExperimentResult]:
    """实验 A：只靠 prompt 要求返回 JSON。"""
    results = []

    for i in range(RUNS_PER_EXPERIMENT):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "分析《百年孤独》这本书，返回 JSON 格式，包含以下字段：\n"
                    "title, author, genre, rating(1-10的浮点数), summary, themes(字符串数组)\n"
                    "只返回 JSON，不要其他内容。"
                ),
            }],
            # 不加 response_format，完全靠 prompt。
        )
        raw = response.choices[0].message.content or ""

        # 尝试解析。
        try:
            cleaned = clean_json_from_markdown(raw)
            data = json.loads(cleaned)
            parsed = BookAnalysis(**data)
            results.append(ExperimentResult(i + 1, True, raw, parsed=parsed))
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            results.append(ExperimentResult(i + 1, False, raw, error=str(e)[:150]))

    return results


# ---------------------------------------------------------------------------
# 实验 B：JSON Mode（中等）
# ---------------------------------------------------------------------------
# 加 response_format={"type": "json_object"}，LLM 保证返回合法 JSON。
# 但字段名、字段类型不保证，可能多字段、少字段、类型错。
# 注意：你的中转站可能会给 JSON 加 markdown 包装，需要清洗。

def experiment_b_json_mode() -> list[ExperimentResult]:
    """实验 B：JSON Mode，保证合法 JSON。"""
    results = []

    for i in range(RUNS_PER_EXPERIMENT):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "分析《百年孤独》这本书，返回 JSON 格式，包含以下字段：\n"
                    "title(字符串), author(字符串), genre(字符串), "
                    "rating(1-10的浮点数), summary(字符串), themes(字符串数组)\n"
                    "只返回 JSON。"
                ),
            }],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""

        try:
            cleaned = clean_json_from_markdown(raw)
            data = json.loads(cleaned)
            parsed = BookAnalysis(**data)
            results.append(ExperimentResult(i + 1, True, raw, parsed=parsed))
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            results.append(ExperimentResult(i + 1, False, raw, error=str(e)[:150]))

    return results


# ---------------------------------------------------------------------------
# 实验 C：Pydantic 后处理 + 重试（最强，你的环境能用的最佳方案）
# ---------------------------------------------------------------------------
# 思路：prompt 里放完整的 JSON Schema 示例 + JSON Mode + Pydantic 校验 + 失败重试。
# 这是在不支持 strict schema 时的最佳实践。

def experiment_c_pydantic_with_retry(max_retries: int = 2) -> list[ExperimentResult]:
    """实验 C：Pydantic 校验 + 重试。"""
    # 把 Pydantic model 的 schema 放进 prompt，给 LLM 最清晰的指引。
    schema_str = json.dumps(BookAnalysis.model_json_schema(), indent=2, ensure_ascii=False)

    results = []

    for i in range(RUNS_PER_EXPERIMENT):
        messages = [{
            "role": "user",
            "content": (
                "分析《百年孤独》这本书。\n\n"
                f"请严格按照以下 JSON Schema 返回，不要添加任何额外字段：\n"
                f"```json\n{schema_str}\n```\n\n"
                "只返回 JSON 对象，不要其他内容。"
            ),
        }]

        parsed = None
        last_error = None
        raw = ""

        for attempt in range(max_retries + 1):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""

            try:
                cleaned = clean_json_from_markdown(raw)
                data = json.loads(cleaned)
                parsed = BookAnalysis(**data)
                break  # 校验通过，停止重试。
            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                last_error = str(e)[:150]
                if attempt < max_retries:
                    # 把错误信息喂回 LLM，让它修正。
                    # 这和 mini_agent 里工具参数解析失败时的策略一样。
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"JSON 校验失败：{last_error}\n"
                            f"请严格按照上面的 Schema 修正后重新返回。"
                        ),
                    })

        if parsed:
            results.append(ExperimentResult(i + 1, True, raw, parsed=parsed))
        else:
            results.append(ExperimentResult(i + 1, False, raw, error=last_error))

    return results


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Day 1：输出稳定性对比实验")
    print(f"每个实验重复 {RUNS_PER_EXPERIMENT} 次\n")

    # 实验 A：纯 prompt 约束。
    print("▶ 运行实验 A：纯 Prompt 约束...")
    results_a = experiment_a_prompt_only()
    print_results("A: 纯 Prompt 约束", results_a)

    # 实验 B：JSON Mode。
    print("\n▶ 运行实验 B：JSON Mode...")
    results_b = experiment_b_json_mode()
    print_results("B: JSON Mode", results_b)

    # 实验 C：Pydantic + 重试。
    print("\n▶ 运行实验 C：Pydantic + 重试...")
    results_c = experiment_c_pydantic_with_retry()
    print_results("C: Pydantic + 重试", results_c)

    # 汇总对比。
    print("\n" + "=" * 60)
    print("汇总对比")
    print("=" * 60)
    for name, results in [
        ("A: 纯 Prompt", results_a),
        ("B: JSON Mode", results_b),
        ("C: Pydantic+重试", results_c),
    ]:
        success = sum(1 for r in results if r.success)
        print(f"  {name:20s}  成功率: {success}/{len(results)}")

    print("\n结论：")
    print("  - 三种方式在强模型（Claude）+ 简单结构下差异不大")
    print("  - 约束手段的价值和模型能力成反比：模型越弱越需要")
    print("  - Week 7 用 vLLM 跑本地小模型时，差异会非常明显")
    print("  - 即使强模型不出错，Pydantic 校验仍然是生产环境的安全网")