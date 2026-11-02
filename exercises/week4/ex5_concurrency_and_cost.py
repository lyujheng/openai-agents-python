"""
第4周 Day 5：并发执行 + 成本控制

三个改造：
  1. 工具并行执行 — asyncio.gather 同时跑多个工具
  2. Token 统计 — 追踪每轮和总计的 token 消耗
  3. 预算限制 — 超预算自动中断 agent 循环

基于 mini_agent 改造，对比串行 vs 并行的耗时差异。

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python ex5_concurrency_and_cost.py
"""

import asyncio
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import AsyncOpenAI, OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2"))
from mini_agent import func_to_schema

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

MODEL = "mco-4"
API_BASE = "https://zyapi.xmsxb.com/v1"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

if not API_KEY:
    sys.exit("请先设置环境变量 DEEPSEEK_API_KEY。")


# ---------------------------------------------------------------------------
# Token 统计器
# ---------------------------------------------------------------------------

@dataclass
class UsageTracker:
    """追踪 token 消耗。每次 LLM 调用后更新。"""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    budget_limit: int | None = None  # token 预算上限，None 表示不限。
    per_call_history: list[dict] = field(default_factory=list)

    def record(self, usage) -> None:
        """记录一次 LLM 调用的 token 消耗。"""
        if usage is None:
            return
        prompt = usage.prompt_tokens or 0
        completion = usage.completion_tokens or 0
        total = usage.total_tokens or (prompt + completion)

        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.total_tokens += total
        self.llm_calls += 1
        self.per_call_history.append({
            "call": self.llm_calls,
            "prompt": prompt,
            "completion": completion,
            "total": total,
            "cumulative": self.total_tokens,
        })

    def is_over_budget(self) -> bool:
        """是否超预算。"""
        if self.budget_limit is None:
            return False
        return self.total_tokens >= self.budget_limit

    def print_summary(self) -> None:
        """打印 token 消耗汇总。"""
        print(f"\n  Token 消耗汇总：")
        print(f"    LLM 调用次数: {self.llm_calls}")
        print(f"    Prompt tokens:     {self.total_prompt_tokens}")
        print(f"    Completion tokens: {self.total_completion_tokens}")
        print(f"    Total tokens:      {self.total_tokens}")
        if self.budget_limit:
            pct = self.total_tokens / self.budget_limit * 100
            print(f"    预算使用: {self.total_tokens}/{self.budget_limit} ({pct:.0f}%)")

    def print_per_call(self) -> None:
        """打印每次调用的 token 消耗。"""
        print(f"\n  逐次调用明细：")
        for entry in self.per_call_history:
            print(f"    Call {entry['call']}: "
                  f"prompt={entry['prompt']}, completion={entry['completion']}, "
                  f"total={entry['total']}, 累计={entry['cumulative']}")


# ---------------------------------------------------------------------------
# AsyncAgent：支持并行工具执行 + token 统计 + 预算控制
# ---------------------------------------------------------------------------

class AsyncAgent:
    """
    基于 mini_agent 改造的 async 版本。

    改进点：
    1. 工具并行执行（asyncio.gather）
    2. Token 统计（UsageTracker）
    3. 预算限制（超预算自动中断）
    """

    def __init__(
        self,
        system_prompt: str,
        tools: list[Callable],
        model: str = MODEL,
        max_turns: int = 10,
        budget_limit: int | None = None,
    ):
        # 用 AsyncOpenAI，支持 async 调用。
        self.client = AsyncOpenAI(base_url=API_BASE, api_key=API_KEY)
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = system_prompt

        self.tool_map: dict[str, Callable] = {f.__name__: f for f in tools}
        self.tool_schemas: list[dict] = [func_to_schema(f) for f in tools]

        # Token 统计器。
        self.tracker = UsageTracker(budget_limit=budget_limit)

    async def run(self, user_input: str) -> str:
        """agent 主循环（async 版）。"""
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        for turn in range(self.max_turns):
            print(f"\n--- Turn {turn + 1}/{self.max_turns} ---")

            # 预算检查：调 LLM 前先检查是否超预算。
            if self.tracker.is_over_budget():
                print(f"  ⚠️ 超预算！已用 {self.tracker.total_tokens} tokens "
                      f"(上限 {self.tracker.budget_limit})")
                return (
                    f"因 token 预算限制（{self.tracker.budget_limit} tokens），"
                    f"agent 被中断。已消耗 {self.tracker.total_tokens} tokens。"
                )

            # 1. 调 LLM（async）。
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_schemas if self.tool_schemas else None,
                tool_choice="auto" if self.tool_schemas else None,
            )

            # 2. 记录 token 消耗。
            self.tracker.record(response.usage)

            message = response.choices[0].message

            # 3. 判断：工具调用还是最终回答？
            if not message.tool_calls:
                print(f"  LLM 最终回答（无工具调用）")
                return message.content or "(空回复)"

            # 4. 并行执行所有工具调用。
            messages.append(message)

            tool_calls = message.tool_calls
            print(f"  收到 {len(tool_calls)} 个工具调用，并行执行...")

            # 用 asyncio.gather 并行执行。
            tasks = [
                self._execute_tool_async(tc) for tc in tool_calls
            ]
            results = await asyncio.gather(*tasks)

            # 把结果追加到 messages。
            for tool_call, result in zip(tool_calls, results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

        return "达到最大轮次限制。"

    async def _execute_tool_async(self, tool_call) -> str:
        """异步执行单个工具调用。"""
        func_name = tool_call.function.name

        try:
            func_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return f"参数 JSON 解析失败：{e}"

        print(f"    → {func_name}({func_args})")

        func = self.tool_map.get(func_name)
        if not func:
            return f"错误：未知工具 '{func_name}'。可用：{list(self.tool_map.keys())}"

        try:
            # 如果工具函数是 async 的，直接 await。
            # 如果是同步的，用 asyncio.to_thread 扔到线程池。
            if inspect.iscoroutinefunction(func):
                result = await func(**func_args)
            else:
                result = await asyncio.to_thread(func, **func_args)
            return str(result)
        except Exception as e:
            return f"工具执行出错：{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 同步版本的 agent（用于对比串行耗时）
# ---------------------------------------------------------------------------

class SyncAgent:
    """串行版本，用于和 AsyncAgent 对比耗时。"""

    def __init__(self, system_prompt: str, tools: list[Callable], model: str = MODEL,
                 max_turns: int = 10):
        self.client = OpenAI(base_url=API_BASE, api_key=API_KEY)
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.tool_map = {f.__name__: f for f in tools}
        self.tool_schemas = [func_to_schema(f) for f in tools]
        self.tracker = UsageTracker()

    def run(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        for turn in range(self.max_turns):
            print(f"\n--- Turn {turn + 1}/{self.max_turns} ---")
            response = self.client.chat.completions.create(
                model=self.model, messages=messages,
                tools=self.tool_schemas if self.tool_schemas else None,
                tool_choice="auto" if self.tool_schemas else None,
            )
            self.tracker.record(response.usage)
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content or "(空回复)"

            messages.append(message)

            print(f"  收到 {len(message.tool_calls)} 个工具调用，串行执行...")
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                     "content": "参数解析失败"})
                    continue

                print(f"    → {func_name}({func_args})")
                func = self.tool_map.get(func_name)
                if not func:
                    result = f"未知工具 '{func_name}'"
                else:
                    try:
                        result = str(func(**func_args))
                    except Exception as e:
                        result = f"出错：{e}"

                messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                 "content": result})

        return "达到最大轮次限制。"


# ---------------------------------------------------------------------------
# 模拟耗时工具（放大串行 vs 并行的差异）
# ---------------------------------------------------------------------------

def search_database(query: str) -> str:
    """搜索数据库，需要一定时间。

    Args:
        query: 搜索关键词
    """
    time.sleep(1)  # 模拟数据库查询耗时 1 秒。
    return f"数据库查询 '{query}' 的结果：找到 42 条相关记录。"


def call_external_api(endpoint: str) -> str:
    """调用外部 API。

    Args:
        endpoint: API 端点名称
    """
    time.sleep(1)  # 模拟外部 API 调用耗时 1 秒。
    return f"API '{endpoint}' 返回：状态正常，数据已更新。"


def analyze_text(text: str) -> str:
    """分析一段文本的情感和关键词。

    Args:
        text: 要分析的文本
    """
    time.sleep(1)  # 模拟文本分析耗时 1 秒。
    return f"文本分析结果：情感=正面，关键词=[技术, 创新, 发展]"


TOOLS = [search_database, call_external_api, analyze_text]

SYSTEM_PROMPT = (
    "你是一个数据分析助手，可以搜索数据库、调用 API、分析文本。\n"
    "当用户的请求涉及多个方面时，尽量同时调用多个工具来提高效率。"
)

TASK = "帮我做一个综合分析：搜索数据库中关于 AI 的记录，调用趋势分析 API，并分析这段文字的情感：'人工智能正在改变世界的每个角落'"


# ---------------------------------------------------------------------------
# 实验
# ---------------------------------------------------------------------------

async def experiment_1_serial_vs_parallel():
    """实验 1：串行 vs 并行的耗时对比。"""
    print("#" * 60)
    print("实验 1：串行 vs 并行工具执行")
    print("#" * 60)

    # 串行。
    print("\n--- 串行执行 ---")
    sync_agent = SyncAgent(system_prompt=SYSTEM_PROMPT, tools=TOOLS)
    start = time.time()
    answer1 = sync_agent.run(TASK)
    serial_time = time.time() - start
    print(f"\n回答: {answer1[:200]}...")
    print(f"耗时: {serial_time:.1f}s")
    sync_agent.tracker.print_summary()

    # 并行。
    print("\n--- 并行执行 ---")
    async_agent = AsyncAgent(system_prompt=SYSTEM_PROMPT, tools=TOOLS)
    start = time.time()
    answer2 = await async_agent.run(TASK)
    parallel_time = time.time() - start
    print(f"\n回答: {answer2[:200]}...")
    print(f"耗时: {parallel_time:.1f}s")
    async_agent.tracker.print_summary()

    # 对比。
    print(f"\n--- 耗时对比 ---")
    print(f"  串行: {serial_time:.1f}s")
    print(f"  并行: {parallel_time:.1f}s")
    if serial_time > 0:
        speedup = serial_time / parallel_time if parallel_time > 0 else float("inf")
        print(f"  加速比: {speedup:.1f}x")


async def experiment_2_budget_control():
    """实验 2：预算控制——设置 token 上限，超预算自动中断。"""
    print("\n\n" + "#" * 60)
    print("实验 2：Token 预算控制")
    print("#" * 60)

    # 不限预算，看一次任务消耗多少 token。
    print("\n--- 不限预算（基线）---")
    agent_baseline = AsyncAgent(
        system_prompt=SYSTEM_PROMPT, tools=TOOLS, budget_limit=None,
    )
    answer = await agent_baseline.run(TASK)
    print(f"\n回答: {answer[:200]}...")
    agent_baseline.tracker.print_summary()
    agent_baseline.tracker.print_per_call()

    baseline_tokens = agent_baseline.tracker.total_tokens

    # 设置一个紧张的预算（基线的 50%），看 agent 是否被中断。
    tight_budget = max(baseline_tokens // 2, 100)
    print(f"\n--- 紧张预算（{tight_budget} tokens，基线的 50%）---")
    agent_tight = AsyncAgent(
        system_prompt=SYSTEM_PROMPT, tools=TOOLS, budget_limit=tight_budget,
    )
    answer_tight = await agent_tight.run(TASK)
    print(f"\n回答: {answer_tight[:200]}...")
    agent_tight.tracker.print_summary()


async def main():
    print("Day 5：并发执行 + 成本控制\n")

    await experiment_1_serial_vs_parallel()
    await experiment_2_budget_control()

    print("\n" + "=" * 60)
    print("Day 5 总结")
    print("=" * 60)
    print("1. 并行执行：多个无依赖的工具调用用 asyncio.gather 同时执行")
    print("   加速效果取决于工具的实际耗时，IO 密集型工具提升最明显")
    print("2. Token 统计：通过 response.usage 获取每次调用的 token 消耗")
    print("3. 预算控制：每轮调 LLM 前检查累计 token，超预算则中断")
    print("   生产环境中按用户/按任务设置预算，防止成本失控")


if __name__ == "__main__":
    asyncio.run(main())
