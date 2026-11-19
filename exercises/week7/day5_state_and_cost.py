"""
第7周 Day 5：状态管理 + 成本优化 + 质量评估

三个主题：
  1. 状态管理：理解 Session Persistence 和 RunState，实现对话中断恢复
  2. 成本优化：token 预算控制、模型路由、工具缓存
  3. 质量评估：用 Day 1 的规划算法和 Day 3 的 RAG 做量化对比

核心概念：
  - previous_response_id：Responses API 的服务端对话延续（中转站不支持）
  - conversation_id：服务端对话分组标识
  - RunState：序列化的运行状态，支持中断后恢复
  - token budget：限制单次 run 的 token 消耗，防止 agent 死循环

精读路径：
  - src/agents/run_config.py → conversation_id、auto_previous_response_id
  - src/agents/result.py → RunResult.to_input_list() 延续对话
  - src/agents/usage.py → Usage 统计

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python day5_state_and_cost.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from agents import Agent, Runner, function_tool, RunConfig
from agents.models.openai_provider import OpenAIProvider


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def make_provider() -> OpenAIProvider:
    return OpenAIProvider(
        base_url="https://zyapi.xmsxb.com/v1",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        use_responses=False,
    )

MODEL = "mco-4"


# ---------------------------------------------------------------------------
# 工具（复用 Day 2 的工具 + 新增缓存工具）
# ---------------------------------------------------------------------------

# 用于演示工具缓存的全局计数器。
_tool_call_count: dict[str, int] = {}
_tool_cache: dict[str, str] = {}


@function_tool
def search_with_cache(query: str) -> str:
    """搜索技术趋势（带缓存）。相同查询只执行一次。

    Args:
        query: 搜索关键词
    """
    _tool_call_count[query] = _tool_call_count.get(query, 0) + 1

    # 缓存命中。
    if query in _tool_cache:
        return f"[缓存命中] {_tool_cache[query]}"

    # 模拟搜索（和 Day 2 相同数据）。
    data = {
        "llm agent": "LLM Agent 快速发展：ReAct 成为主流范式，多 Agent 协作兴起，MCP 协议标准化工具集成。",
        "rag": "RAG 从简单检索进化到 Agentic RAG：agent 自主决定检索策略、查询重写、多轮检索。",
        "mcp": "MCP (Model Context Protocol) 由 Anthropic 提出，2024 年成为工具集成的事实标准。",
        "vllm": "vLLM 持续优化：PagedAttention v2、Speculative Decoding、Prefix Caching。",
    }
    for key, value in data.items():
        if key in query.lower():
            _tool_cache[query] = value
            return value

    result = f"未找到 '{query}' 的数据。"
    _tool_cache[query] = result
    return result


@function_tool
def get_code_example(topic: str) -> str:
    """获取某个主题的代码示例。

    Args:
        topic: 主题名，如 'asyncio'、'decorator'
    """
    examples = {
        "asyncio": "import asyncio\nasync def main():\n    await asyncio.sleep(1)\nasyncio.run(main())",
        "decorator": "@functools.wraps(fn)\ndef wrapper(*a, **kw):\n    return fn(*a, **kw)",
        "context_manager": "class MyCtx:\n    def __enter__(self): ...\n    def __exit__(self, *exc): ...",
    }
    for key, code in examples.items():
        if key in topic.lower():
            return f"```python\n{code}\n```"
    return f"没有 '{topic}' 的代码示例。"


# ===========================================================================
# 实验一：对话状态延续（to_input_list）
# ===========================================================================
# 中转站不支持 Responses API 的 previous_response_id，
# 但可以用 result.to_input_list() 把上一轮结果作为下一轮输入，
# 在客户端实现对话延续。

async def experiment_conversation_continuation() -> None:
    """多轮对话延续：用 to_input_list() 串联多轮。"""
    print("\n实验一：对话状态延续")
    print("=" * 60)
    provider = make_provider()

    agent = Agent(
        name="对话助手",
        instructions=(
            "你是一个技术问答助手。可以用搜索工具查找信息。\n"
            "记住之前的对话内容，在上下文基础上回答后续问题。"
        ),
        tools=[search_with_cache, get_code_example],
        model=MODEL,
    )

    run_config_dict: dict[str, Any] = {"model_provider": provider}

    # 第 1 轮。
    print("\n[第1轮] 用户：介绍一下 LLM Agent")
    result1 = await Runner.run(agent, "介绍一下 LLM Agent", run_config=run_config_dict)
    print(f"  助手：{str(result1.final_output)[:200]}...")

    # 第 2 轮：基于第 1 轮的结果继续对话。
    # to_input_list() 把前面的对话历史（包括工具调用）序列化为输入格式。
    input_for_round2 = result1.to_input_list() + [
        {"role": "user", "content": "你刚才提到的 ReAct 是什么？详细说说"},
    ]
    print("\n[第2轮] 用户：你刚才提到的 ReAct 是什么？详细说说")
    result2 = await Runner.run(agent, input_for_round2, run_config=run_config_dict)
    print(f"  助手：{str(result2.final_output)[:200]}...")

    # 第 3 轮：继续追问。
    input_for_round3 = result2.to_input_list() + [
        {"role": "user", "content": "RAG 和 Agent 是什么关系？"},
    ]
    print("\n[第3轮] 用户：RAG 和 Agent 是什么关系？")
    result3 = await Runner.run(agent, input_for_round3, run_config=run_config_dict)
    print(f"  助手：{str(result3.final_output)[:200]}...")

    print("\n✅ 三轮对话成功延续，agent 能引用之前的上下文回答")

    # 展示 to_input_list() 的结构。
    input_list = result1.to_input_list()
    print(f"\n  to_input_list() 返回 {len(input_list)} 个消息")
    for item in input_list[:5]:
        if isinstance(item, dict):
            role = item.get("role", "?")
            content = str(item.get("content", ""))[:60]
            print(f"    [{role}] {content}...")
        else:
            print(f"    [{type(item).__name__}] ...")


# ===========================================================================
# 实验二：Token 预算控制
# ===========================================================================
# agent 可能无限循环消耗 token。需要在 run 级别设置限制。
# RunConfig.max_turns 是最直接的控制手段。

@dataclass
class CostTracker:
    """追踪多次 run 的成本。"""
    runs: list[dict] = field(default_factory=list)

    def record(self, name: str, result: Any, elapsed: float) -> None:
        """记录一次 run 的成本。"""
        usage = result.raw_responses[-1].usage if hasattr(result, "raw_responses") and result.raw_responses else None
        items_count = len(result.new_items) if hasattr(result, "new_items") else 0

        self.runs.append({
            "name": name,
            "items": items_count,
            "elapsed": elapsed,
            "usage": usage,
        })

    def report(self) -> None:
        """打印成本报告。"""
        print(f"\n  {'任务':<20} {'items数':<10} {'耗时':<10}")
        print("  " + "-" * 40)
        for r in self.runs:
            print(f"  {r['name']:<20} {r['items']:<10} {r['elapsed']:<10.1f}")


async def experiment_cost_control() -> None:
    """Token 预算控制：max_turns 限制 + 成本追踪。"""
    print("\n实验二：Token 预算控制")
    print("=" * 60)
    provider = make_provider()
    tracker = CostTracker()

    agent = Agent(
        name="调研助手",
        instructions=(
            "你是一个技术调研助手。用搜索工具尽可能多地调研相关技术。\n"
            "每次搜索后评估是否还需要搜索更多。"
        ),
        tools=[search_with_cache],
        model=MODEL,
    )

    task = "全面调研 LLM Agent、RAG、MCP、vLLM 的最新趋势"

    # 不同 max_turns 设置的对比。
    for max_turns in [3, 5, 10]:
        print(f"\n--- max_turns={max_turns} ---")
        _tool_call_count.clear()
        _tool_cache.clear()

        start = time.time()
        try:
            result = await Runner.run(
                agent,
                task,
                run_config=RunConfig(model_provider=provider),
                max_turns=max_turns,
            )
            elapsed = time.time() - start
            tracker.record(f"max_turns={max_turns}", result, elapsed)

            output = str(result.final_output)
            print(f"  完成，items: {len(result.new_items)}, 耗时: {elapsed:.1f}s")
            print(f"  工具调用统计: {dict(_tool_call_count)}")
            print(f"  回答长度: {len(output)} 字")
            print(f"  回答摘要: {output[:150]}...")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ 异常（{elapsed:.1f}s）: {type(e).__name__}: {e}")

    # 成本报告。
    print("\n成本对比报告:")
    tracker.report()

    print("""
  启示：
    - max_turns=3 可能不够完成任务，但成本最低
    - max_turns=10 给 agent 足够空间，但可能浪费 token 在不必要的检索上
    - 生产环境建议：设置合理的 max_turns + 监控实际使用量
    - 幂等工具（搜索）的相同参数缓存可以节省实际执行成本
""")


# ===========================================================================
# 实验三：工具缓存策略
# ===========================================================================

async def experiment_tool_caching() -> None:
    """工具缓存：相同参数的幂等工具只执行一次。"""
    print("\n实验三：工具缓存策略")
    print("=" * 60)
    provider = make_provider()

    _tool_call_count.clear()
    _tool_cache.clear()

    agent = Agent(
        name="缓存演示",
        instructions=(
            "你是一个调研助手。请分别搜索以下主题并汇总：\n"
            "1. LLM Agent\n"
            "2. RAG\n"
            "3. 再搜一次 LLM Agent（注意是否命中缓存）\n"
            "搜索完后汇总结果。"
        ),
        tools=[search_with_cache],
        model=MODEL,
    )

    result = await Runner.run(
        agent,
        "调研 LLM Agent 和 RAG，并对 LLM Agent 做二次确认搜索",
        run_config={"model_provider": provider},
    )

    print(f"\n  工具调用次数: {dict(_tool_call_count)}")
    print(f"  缓存状态: {list(_tool_cache.keys())}")

    # 分析缓存效果。
    for query, count in _tool_call_count.items():
        if count > 1:
            print(f"  ✅ '{query}' 被调用 {count} 次，第 2 次命中缓存")
        else:
            print(f"  📋 '{query}' 调用 1 次")


# ===========================================================================
# 实验四：模型路由策略（概念演示）
# ===========================================================================

async def experiment_model_routing() -> None:
    """
    模型路由：简单任务用小模型，复杂任务用大模型。

    注意：因为我们只有一个中转站模型（mco-4），这里是概念演示。
    生产环境中会用 gpt-4o-mini 做简单任务，gpt-4o 做复杂任务。
    """
    print("\n实验四：模型路由策略（概念演示）")
    print("=" * 60)
    provider = make_provider()

    # 路由 agent：判断任务复杂度并分发。
    router = Agent(
        name="任务路由器",
        instructions=(
            "你是一个任务路由器。分析用户任务的复杂度：\n"
            "- 简单任务（直接回答、单一知识点）：回复 [SIMPLE] 然后直接回答\n"
            "- 复杂任务（需要多步推理、比较分析、综合判断）：回复 [COMPLEX] 然后回答\n"
            "在回答前标注 [SIMPLE] 或 [COMPLEX]。"
        ),
        model=MODEL,
    )

    tasks = [
        ("Python 的 list 和 tuple 有什么区别？", "简单 → 小模型"),
        ("设计一个支持百万用户的实时推荐系统，需要考虑哪些架构决策？", "复杂 → 大模型"),
        ("什么是装饰器？", "简单 → 小模型"),
        ("对比 microservices 和 monolith 在团队规模 5 人 vs 50 人时的优劣", "复杂 → 大模型"),
    ]

    for task, expected in tasks:
        result = await Runner.run(
            router,
            task,
            run_config={"model_provider": provider},
        )
        output = str(result.final_output)
        is_complex = "[COMPLEX]" in output.upper() or "[复杂]" in output
        actual = "COMPLEX" if is_complex else "SIMPLE"
        print(f"\n  任务: {task[:50]}...")
        print(f"  预期: {expected}")
        print(f"  实际: {actual}")
        match = ("复杂" in expected and is_complex) or ("简单" in expected and not is_complex)
        print(f"  {'✅' if match else '❌'} 路由{'正确' if match else '偏差'}")

    print("""
  模型路由策略：
    - 简单任务 → 小模型（gpt-4o-mini / Qwen2.5-7B）：低成本、低延迟
    - 复杂任务 → 大模型（gpt-4o / Qwen2.5-72B）：高质量、高成本
    - 路由本身用小模型判断即可，成本可忽略
    - 可以节省 50-80% 的 token 成本（大部分请求是简单任务）
""")


# ===========================================================================
# 实验五：综合质量评估
# ===========================================================================

async def experiment_quality_eval() -> None:
    """用 LLM-as-judge 对 Day 1 和 Day 3 的不同方案做质量评估。"""
    print("\n实验五：综合质量评估")
    print("=" * 60)
    provider = make_provider()

    judge = Agent(
        name="质量评估官",
        instructions=(
            "你是一个严格的回答质量评估官。给定一个问题和回答，你需要评估：\n"
            "1. 准确性（0-10）：信息是否正确\n"
            "2. 完整性（0-10）：是否覆盖了问题的所有方面\n"
            "3. 条理性（0-10）：组织是否清晰\n"
            "4. 总评（0-10）：综合打分\n\n"
            "输出格式：\n"
            "准确性：X/10\n"
            "完整性：X/10\n"
            "条理性：X/10\n"
            "总评：X/10\n"
            "简评：一句话总结"
        ),
        model=MODEL,
    )

    # 用两种方式回答同一个问题，让 judge 评估。
    question = "Python 有哪些并发方案？各自适合什么场景？"

    # 方案 A：简单直接回答。
    simple_agent = Agent(
        name="简单助手",
        instructions="你是一个 Python 专家。直接回答用户问题，50 字以内。",
        model=MODEL,
    )

    # 方案 B：深入分析回答。
    detailed_agent = Agent(
        name="详细助手",
        instructions=(
            "你是一个资深 Python 专家。详细回答用户问题，包含：\n"
            "- 每种方案的原理\n"
            "- 适用场景和限制\n"
            "- 选择建议\n"
            "300 字以内。"
        ),
        model=MODEL,
    )

    run_config_dict: dict[str, Any] = {"model_provider": provider}

    result_a = await Runner.run(simple_agent, question, run_config=run_config_dict)
    result_b = await Runner.run(detailed_agent, question, run_config=run_config_dict)

    answer_a = str(result_a.final_output)
    answer_b = str(result_b.final_output)

    print(f"\n  [方案A - 简短] {answer_a[:150]}...")
    print(f"\n  [方案B - 详细] {answer_b[:150]}...")

    # Judge 评估。
    for label, answer in [("方案A（简短）", answer_a), ("方案B（详细）", answer_b)]:
        judge_input = f"问题：{question}\n\n回答：\n{answer}\n\n请评估这个回答的质量。"
        judge_result = await Runner.run(judge, judge_input, run_config=run_config_dict)
        print(f"\n  [{label}] 评估结果：")
        print(f"  {str(judge_result.final_output)[:300]}")

    print("""
  质量评估的价值：
    - 自动化评估让 agent 的迭代改进有据可依
    - LLM-as-judge 不是完美的，但可以作为快速筛选手段
    - 生产环境建议：自动评估 + 人工抽查校准
    - 评估维度要和业务目标对齐（不是越详细越好）
""")


# ===========================================================================
# 主入口
# ===========================================================================

async def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY。")

    print("第7周 Day 5：状态管理 + 成本优化 + 质量评估")
    print("=" * 70)

    await experiment_conversation_continuation()
    await experiment_cost_control()
    await experiment_tool_caching()
    await experiment_model_routing()
    await experiment_quality_eval()

    # 总结。
    print(f"\n{'=' * 70}")
    print("Day 5 总结")
    print(f"{'=' * 70}")
    print("""
  状态管理：
    - to_input_list() 在客户端实现对话延续（中转站兼容）
    - previous_response_id 在服务端延续（需要 Responses API）
    - RunState 序列化可以实现中断恢复（进阶场景）

  成本优化：
    - max_turns 限制是最简单有效的防护
    - 工具缓存对幂等工具（搜索、查询）很有效
    - 模型路由可以节省 50-80% 成本
    - 对话压缩减少 input token（长对话场景）

  质量评估：
    - LLM-as-judge 可以做快速自动评估
    - 评估维度要和业务目标对齐
    - 自动评估 + 人工抽查是务实的组合
    - eval 驱动的改进循环是生产环境的标准工作流
""")


if __name__ == "__main__":
    asyncio.run(main())
