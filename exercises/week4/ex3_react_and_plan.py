"""
第4周 Day 3：自我纠错与规划策略对比

三种策略用同一个任务对比：
  1. 普通循环 — mini_agent 的默认行为，直接调工具
  2. ReAct — 每步先写 Thought 再行动
  3. Plan-then-Execute — 先生成计划再逐步执行

任务：分析 week1 和 week2 的练习代码，找出所有定义了工具函数的文件，
     统计每个文件的工具数量，按数量排序，写入 tool_report.txt。

这个任务需要 3-5 步工具调用，复杂度适中，能体现规划策略的差异。

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python ex3_react_and_plan.py
"""

import json
import os
import sys
import time

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2"))
from mini_agent import MiniAgent, call_llm_with_retry

client = OpenAI(
    base_url="https://zyapi.xmsxb.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
)
MODEL = "mco-4"

# 工作目录：exercises/
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 共用工具（三种策略用同一套工具，公平对比）
# ---------------------------------------------------------------------------

def list_directory(path: str = ".") -> str:
    """列出目录下的文件和子目录。

    Args:
        path: 目录路径，相对于 exercises 目录
    """
    full = os.path.normpath(os.path.join(WORKSPACE, path))
    if not full.startswith(WORKSPACE):
        return f"错误：不允许访问 exercises 目录之外的路径"
    if not os.path.isdir(full):
        return f"错误：'{path}' 不是一个目录"

    entries = []
    for name in sorted(os.listdir(full)):
        entry_path = os.path.join(full, name)
        if name.startswith((".", "__")):
            continue
        if os.path.isdir(entry_path):
            entries.append(f"  [目录] {name}/")
        else:
            entries.append(f"  [文件] {name}")
    return f"目录 '{path}' 下共 {len(entries)} 项：\n" + "\n".join(entries)


def read_file(path: str) -> str:
    """读取文件内容。

    Args:
        path: 文件路径，相对于 exercises 目录
    """
    full = os.path.normpath(os.path.join(WORKSPACE, path))
    if not full.startswith(WORKSPACE):
        return "错误：不允许访问 exercises 目录之外的路径"
    if not os.path.isfile(full):
        return f"错误：文件 '{path}' 不存在"
    try:
        with open(full, encoding="utf-8") as f:
            content = f.read()
        if len(content) > 3000:
            content = content[:3000] + f"\n... [截断，原始 {len(content)} 字符]"
        return content
    except UnicodeDecodeError:
        return f"错误：'{path}' 不是文本文件"


def write_file(path: str, content: str) -> str:
    """将内容写入文件。

    Args:
        path: 文件路径，相对于 exercises 目录
        content: 要写入的内容
    """
    full = os.path.normpath(os.path.join(WORKSPACE, path))
    if not full.startswith(WORKSPACE):
        return "错误：不允许访问 exercises 目录之外的路径"
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"成功写入 '{path}'（{len(content)} 字符）"


TOOLS = [list_directory, read_file, write_file]

# 任务描述。
TASK = (
    "分析 week1 和 week2 目录下的 Python 练习代码，"
    "找出所有定义了工具函数的文件（工具函数指用于给 LLM 调用的函数，"
    "比如 get_weather、get_exchange_rate 这类），"
    "统计每个文件中工具函数的数量，按数量从多到少排序，"
    "结果写入 week4/tool_report.txt。"
)


# ---------------------------------------------------------------------------
# 策略 1：普通循环（baseline）
# ---------------------------------------------------------------------------

def run_baseline():
    """普通循环：直接用 mini_agent 的默认行为。"""
    print("=" * 60)
    print("策略 1：普通循环（mini_agent 默认）")
    print("=" * 60)

    agent = MiniAgent(
        system_prompt=(
            "你是一个代码分析助手，可以浏览目录、读取文件、写入文件。\n"
            "根据用户的任务，调用工具完成分析。"
        ),
        tools=TOOLS,
    )
    start = time.time()
    answer = agent.run(TASK)
    elapsed = time.time() - start
    print(f"\n最终回答: {answer}")
    print(f"耗时: {elapsed:.1f}s")
    return answer


# ---------------------------------------------------------------------------
# 策略 2：ReAct
# ---------------------------------------------------------------------------

def run_react():
    """ReAct：每步先写 Thought 再行动。"""
    print("\n" + "=" * 60)
    print("策略 2：ReAct（Thought → Action → Observation）")
    print("=" * 60)

    react_prompt = (
        "你是一个代码分析助手，可以浏览目录、读取文件、写入文件。\n\n"
        "你必须严格按照以下格式工作：\n\n"
        "Thought: 分析当前情况，说明你要做什么以及为什么\n"
        "然后调用工具（Action），等待结果（Observation）\n\n"
        "Thought: 根据观察到的结果，分析下一步该做什么\n"
        "然后继续调用工具，或者给出最终回答\n\n"
        "每次调用工具之前，必须先写 Thought 说明理由。\n"
        "当你有足够信息回答用户时，写最后一个 Thought 总结所有发现，然后给出最终回答。"
    )

    agent = MiniAgent(
        system_prompt=react_prompt,
        tools=TOOLS,
    )
    start = time.time()
    answer = agent.run(TASK)
    elapsed = time.time() - start
    print(f"\n最终回答: {answer}")
    print(f"耗时: {elapsed:.1f}s")
    return answer


# ---------------------------------------------------------------------------
# 策略 3：Plan-then-Execute
# ---------------------------------------------------------------------------

def run_plan_then_execute():
    """Plan-then-Execute：先生成计划，再逐步执行。"""
    print("\n" + "=" * 60)
    print("策略 3：Plan-then-Execute（先规划再执行）")
    print("=" * 60)

    # 阶段 1：生成计划。
    print("\n--- 阶段 1：生成计划 ---")
    plan_response = call_llm_with_retry(
        client,
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个任务规划助手。根据用户的任务，生成一个分步执行计划。\n"
                    "可用工具：list_directory(path)、read_file(path)、write_file(path, content)\n"
                    "工作目录下有 week1/ 和 week2/ 两个子目录，包含 Python 练习代码。\n\n"
                    "输出格式（纯文本，每行一步）：\n"
                    "Step 1: 具体描述\n"
                    "Step 2: 具体描述\n"
                    "..."
                ),
            },
            {"role": "user", "content": TASK},
        ],
    )
    plan = plan_response.choices[0].message.content or ""
    print(f"计划：\n{plan}")

    # 阶段 2：按计划执行。
    print("\n--- 阶段 2：按计划执行 ---")
    execute_prompt = (
        "你是一个代码分析助手，正在按计划执行任务。\n\n"
        f"完整计划：\n{plan}\n\n"
        "请从 Step 1 开始，逐步执行。每完成一步，说明当前进度，然后继续下一步。\n"
        "如果某一步遇到问题，说明问题并调整后续计划。\n"
        "所有步骤完成后，给出最终总结。"
    )

    agent = MiniAgent(
        system_prompt=execute_prompt,
        tools=TOOLS,
    )
    start = time.time()
    answer = agent.run(TASK)
    elapsed = time.time() - start
    print(f"\n最终回答: {answer}")
    print(f"耗时: {elapsed:.1f}s")
    return answer


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Day 3：规划策略对比实验")
    print(f"任务：{TASK}\n")

    answer1 = run_baseline()
    answer2 = run_react()
    answer3 = run_plan_then_execute()

    # 检查 tool_report.txt 是否生成。
    report_path = os.path.join(WORKSPACE, "week4", "tool_report.txt")
    print("\n" + "=" * 60)
    print("验证：tool_report.txt")
    print("=" * 60)
    if os.path.exists(report_path):
        with open(report_path) as f:
            print(f.read())
    else:
        print("⚠️ tool_report.txt 未生成")

    print("\n" + "=" * 60)
    print("对比观察点")
    print("=" * 60)
    print("1. 三种策略的工具调用次数是否不同？（看 Turn 数）")
    print("2. ReAct 的 Thought 是否让推理过程更清晰？")
    print("3. Plan-then-Execute 的计划质量如何？是否遗漏步骤？")
    print("4. 哪种策略最终的 tool_report.txt 内容最完整准确？")
    print("5. 耗时差异——Plan 阶段额外调了一次 LLM，值得吗？")
