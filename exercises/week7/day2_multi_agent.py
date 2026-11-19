"""
第7周 Day 2：Multi-Agent 协作模式

用 OpenAI Agents SDK 实现 5 种 Multi-Agent 协作模式：
  1. Serial Handoff（串行交接）—— agent 之间顺序传递控制权
  2. Parallel Fan-out（并行分发）—— 多个 agent 同时处理不同子任务
  3. Debate（辩论）—— 两个 agent 争论，第三个裁判
  4. Voting（投票）—— 多个 agent 独立回答，取多数一致
  5. Hierarchical（层级）—— Manager 分配任务给 Worker（agents-as-tools）

核心设计决策：
  - handoff = 控制权转移（当前 agent 退出循环，新 agent 接管）
  - agent.as_tool() = 子任务调用（当前 agent 保持控制，子 agent 当工具用）
  - asyncio.gather = 并行执行多个独立 Runner.run()

参考 SDK 示例：
  - examples/agent_patterns/routing.py        — handoff 路由
  - examples/agent_patterns/agents_as_tools.py — agent 当工具
  - examples/agent_patterns/parallelization.py — 并行 + 裁判
  - examples/handoffs/message_filter.py        — handoff 消息过滤

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python day2_multi_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field

from agents import Agent, Runner, function_tool, handoff, ItemHelpers
from agents.models.openai_provider import OpenAIProvider


# ---------------------------------------------------------------------------
# 通用配置
# ---------------------------------------------------------------------------

def make_provider() -> OpenAIProvider:
    return OpenAIProvider(
        base_url="https://zyapi.xmsxb.com/v1",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        use_responses=False,  # 中转站用 Chat Completions API。
    )

MODEL = "mco-4"


@dataclass
class ExperimentResult:
    """一次实验的结果。"""
    pattern: str
    final_output: str = ""
    agent_calls: int = 0
    elapsed_seconds: float = 0.0
    steps: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.steps.append(msg)
        print(f"  [{self.pattern}] {msg}")


# ---------------------------------------------------------------------------
# 模拟工具：让 agent 有事可做
# ---------------------------------------------------------------------------

@function_tool
def search_tech_trends(topic: str) -> str:
    """搜索某个技术方向的发展趋势。

    Args:
        topic: 技术关键词，如 'LLM Agent'、'RAG'、'MCP'
    """
    data = {
        "llm agent": "2024-2025 年 LLM Agent 快速发展：ReAct 成为主流范式，多 Agent 协作兴起，MCP 协议标准化工具集成。主要框架有 LangGraph、CrewAI、OpenAI Agents SDK。",
        "rag": "RAG 从简单检索进化到 Agentic RAG：agent 自主决定检索策略、查询重写、多轮检索。GraphRAG、HyDE、Self-RAG 等变体涌现。",
        "mcp": "MCP (Model Context Protocol) 由 Anthropic 提出，2024 年成为工具集成的事实标准。支持 stdio/SSE/streamable-http 传输，社区 server 生态快速增长。",
        "vllm": "vLLM 持续优化推理性能：PagedAttention v2、Speculative Decoding、Prefix Caching。支持 OpenAI 兼容 API，成为本地部署首选。",
        "prompt engineering": "Prompt 工程从 few-shot 向系统化发展：结构化 prompt、Chain-of-Thought、Self-Consistency、Tree of Thoughts。DSPy 尝试自动优化 prompt。",
    }
    for key, value in data.items():
        if key in topic.lower():
            return value
    return f"未找到 '{topic}' 的趋势数据。可查询：{', '.join(data.keys())}"


@function_tool
def analyze_code_quality(code_snippet: str) -> str:
    """分析一段代码的质量（模拟）。

    Args:
        code_snippet: 要分析的代码片段
    """
    issues = []
    if "eval(" in code_snippet:
        issues.append("安全风险：使用了 eval()，可能导致代码注入")
    if "except:" in code_snippet or "except Exception:" in code_snippet:
        issues.append("过于宽泛的异常捕获，应该捕获具体异常类型")
    if len(code_snippet.split("\n")) > 50:
        issues.append("函数过长，建议拆分")
    if not issues:
        issues.append("未发现明显问题，代码质量良好")
    return "代码分析结果：\n" + "\n".join(f"- {i}" for i in issues)


# ===========================================================================
# 模式一：Serial Handoff（串行交接）
# ===========================================================================
# 场景：流水线式任务，Agent A 负责调研 → Agent B 负责分析 → Agent C 负责总结。
# 关键：handoffs=[next_agent]，控制权直接转移。

async def experiment_serial_handoff() -> ExperimentResult:
    """串行交接：调研 → 分析 → 总结，三个 agent 顺序传递。"""
    r = ExperimentResult(pattern="Serial Handoff")
    start = time.time()
    provider = make_provider()

    # Agent C：总结者（最后一环，没有 handoff）。
    summarizer = Agent(
        name="总结者",
        instructions=(
            "你是一个总结专家。根据之前的调研和分析结果，给出简洁的技术总结报告（200 字以内）。"
            "直接输出总结，不要再调工具。"
        ),
        model=MODEL,
    )

    # Agent B：分析者（分析完交给总结者）。
    analyst = Agent(
        name="分析者",
        instructions=(
            "你是一个技术分析师。根据调研结果，分析技术趋势的关键洞察。"
            "分析完成后，把工作交给总结者。"
        ),
        model=MODEL,
        handoffs=[summarizer],
    )

    # Agent A：调研者（调研完交给分析者）。
    researcher = Agent(
        name="调研者",
        instructions=(
            "你是一个技术调研员。使用搜索工具调研用户问的技术方向。"
            "搜索 1-2 个相关主题后，把工作交给分析者。"
        ),
        tools=[search_tech_trends],
        model=MODEL,
        handoffs=[analyst],
    )

    r.log("启动串行流水线：调研者 → 分析者 → 总结者")
    result = await Runner.run(
        researcher,
        "调研 LLM Agent 和 RAG 的最新发展趋势",
        run_config={"model_provider": provider},
    )

    r.final_output = str(result.final_output)
    r.agent_calls = len([item for item in result.new_items if hasattr(item, "agent")])
    r.log(f"最终由 [{result.last_agent.name}] 输出")
    r.log(f"经过 {len(result.new_items)} 个 items")
    r.elapsed_seconds = time.time() - start
    return r


# ===========================================================================
# 模式二：Parallel Fan-out（并行分发）
# ===========================================================================
# 场景：多个独立子任务可以同时进行，最后汇总。
# 关键：asyncio.gather 并行运行多个 Runner.run()，结果汇总给 synthesizer。

async def experiment_parallel_fanout() -> ExperimentResult:
    """并行分发：3 个 agent 同时调研不同方向，最后汇总。"""
    r = ExperimentResult(pattern="Parallel Fan-out")
    start = time.time()
    provider = make_provider()

    # 三个并行的调研 agent，各自聚焦不同方向。
    agent_llm = Agent(
        name="LLM Agent 调研员",
        instructions="你是 LLM Agent 方向的调研专家。用搜索工具调研后，给出 100 字的调研摘要。",
        tools=[search_tech_trends],
        model=MODEL,
    )
    agent_rag = Agent(
        name="RAG 调研员",
        instructions="你是 RAG 方向的调研专家。用搜索工具调研后，给出 100 字的调研摘要。",
        tools=[search_tech_trends],
        model=MODEL,
    )
    agent_mcp = Agent(
        name="MCP 调研员",
        instructions="你是 MCP 协议方向的调研专家。用搜索工具调研后，给出 100 字的调研摘要。",
        tools=[search_tech_trends],
        model=MODEL,
    )

    # 汇总 agent。
    synthesizer = Agent(
        name="汇总者",
        instructions=(
            "你是技术报告的汇总专家。"
            "根据三个调研员的独立调研结果，综合写一份简洁的技术全景报告（300 字以内）。"
            "要提炼共性趋势和各方向的独特亮点。"
        ),
        model=MODEL,
    )

    r.log("并行启动 3 个调研 agent...")
    run_config = {"model_provider": provider}

    # 并行执行。
    results = await asyncio.gather(
        Runner.run(agent_llm, "调研 LLM Agent 的最新趋势", run_config=run_config),
        Runner.run(agent_rag, "调研 RAG 技术的最新趋势", run_config=run_config),
        Runner.run(agent_mcp, "调研 MCP 协议的最新趋势", run_config=run_config),
    )

    r.log("3 个 agent 全部完成，汇总结果...")

    # 收集各 agent 的输出。
    summaries = []
    for res in results:
        agent_name = res.last_agent.name
        output = str(res.final_output)
        summaries.append(f"[{agent_name}]\n{output}")
        r.log(f"  {agent_name}: {output[:60]}...")

    combined = "\n\n".join(summaries)

    # 汇总。
    final_result = await Runner.run(
        synthesizer,
        f"以下是三个方向的独立调研结果，请综合汇总：\n\n{combined}",
        run_config=run_config,
    )

    r.final_output = str(final_result.final_output)
    r.agent_calls = 4  # 3 个并行 + 1 个汇总。
    r.elapsed_seconds = time.time() - start
    return r


# ===========================================================================
# 模式三：Debate（辩论）
# ===========================================================================
# 场景：对同一问题，两个 agent 从不同立场分析，第三个 agent 裁判。
# 关键：并行获取不同视角 → 裁判综合判断。

async def experiment_debate() -> ExperimentResult:
    """辩论模式：乐观派 vs 悲观派 → 裁判。"""
    r = ExperimentResult(pattern="Debate")
    start = time.time()
    provider = make_provider()

    topic = "LLM Agent 是否已经准备好用于生产环境？"

    optimist = Agent(
        name="乐观派",
        instructions=(
            "你是技术乐观派。对于给定的技术话题，从积极的角度分析，"
            "列出 3 个支持该技术已经成熟的论据。每个论据要有具体例子。200 字以内。"
        ),
        model=MODEL,
    )

    pessimist = Agent(
        name="悲观派",
        instructions=(
            "你是技术悲观派。对于给定的技术话题，从审慎的角度分析，"
            "列出 3 个该技术尚未成熟的风险点。每个风险要有具体案例。200 字以内。"
        ),
        model=MODEL,
    )

    judge = Agent(
        name="裁判",
        instructions=(
            "你是一个公正的技术评审专家。你会看到乐观派和悲观派各自的论述，"
            "请客观评估双方观点的合理性，给出你的综合判断和建议。200 字以内。"
        ),
        model=MODEL,
    )

    r.log(f"辩题：{topic}")
    r.log("乐观派和悲观派同时发言...")
    run_config = {"model_provider": provider}

    # 两个视角并行分析。
    opt_result, pes_result = await asyncio.gather(
        Runner.run(optimist, topic, run_config=run_config),
        Runner.run(pessimist, topic, run_config=run_config),
    )

    opt_output = str(opt_result.final_output)
    pes_output = str(pes_result.final_output)
    r.log(f"乐观派：{opt_output[:60]}...")
    r.log(f"悲观派：{pes_output[:60]}...")

    # 裁判裁决。
    r.log("裁判评判中...")
    judge_input = (
        f"辩题：{topic}\n\n"
        f"【乐观派观点】\n{opt_output}\n\n"
        f"【悲观派观点】\n{pes_output}\n\n"
        "请给出你的综合判断。"
    )
    judge_result = await Runner.run(judge, judge_input, run_config=run_config)

    r.final_output = str(judge_result.final_output)
    r.agent_calls = 3
    r.elapsed_seconds = time.time() - start
    return r


# ===========================================================================
# 模式四：Voting（投票）
# ===========================================================================
# 场景：对同一问题让多个 agent 独立回答，取多数一致的结果提高可靠性。
# 关键：相同 prompt 并行跑多次 → 汇总投票。

async def experiment_voting() -> ExperimentResult:
    """投票模式：3 个 agent 独立回答，aggregator 取共识。"""
    r = ExperimentResult(pattern="Voting")
    start = time.time()
    provider = make_provider()

    question = "在以下场景中应该选哪种 agent 架构？场景：一个客服系统，需要根据用户问题类型（退款、技术支持、投诉）路由到不同的专业 agent 处理。请从以下选项中选一个：A. 单 Agent + 多工具  B. Handoff 路由  C. 并行 Fan-out  D. 层级式 Manager-Worker"

    # 三个独立的投票 agent，用略有差异的 prompt 增加多样性。
    voters = [
        Agent(
            name=f"投票者{i+1}",
            instructions=(
                "你是一个 agent 架构专家。分析用户描述的场景，从给定选项中选出最合适的架构。\n"
                "先简述理由（50 字以内），然后明确给出你的选择（A/B/C/D）。\n"
                "最后一行格式：我的选择：X"
            ),
            model=MODEL,
        )
        for i in range(3)
    ]

    aggregator = Agent(
        name="汇总者",
        instructions=(
            "你会看到 3 位专家各自独立的选择和理由。\n"
            "统计每个选项的票数，给出多数一致的最终选择。\n"
            "如果三人都不同，选你认为理由最充分的。\n"
            "输出格式：\n各专家选择：...\n最终选择：X\n理由：..."
        ),
        model=MODEL,
    )

    r.log(f"问题：{question[:60]}...")
    r.log("3 个投票者独立思考中...")
    run_config = {"model_provider": provider}

    # 并行投票。
    vote_results = await asyncio.gather(
        *[Runner.run(v, question, run_config=run_config) for v in voters]
    )

    votes_text = []
    for i, vr in enumerate(vote_results):
        output = str(vr.final_output)
        votes_text.append(f"[投票者{i+1}] {output}")
        r.log(f"投票者{i+1}: {output[:80]}...")

    # 汇总投票。
    r.log("汇总投票结果...")
    agg_input = f"问题：{question}\n\n各专家独立投票：\n\n" + "\n\n".join(votes_text)
    agg_result = await Runner.run(aggregator, agg_input, run_config=run_config)

    r.final_output = str(agg_result.final_output)
    r.agent_calls = 4  # 3 个投票者 + 1 个汇总。
    r.elapsed_seconds = time.time() - start
    return r


# ===========================================================================
# 模式五：Hierarchical（层级式 Manager-Worker）
# ===========================================================================
# 场景：Manager agent 持有多个 Worker agent 作为 tools，动态分配子任务。
# 关键：agent.as_tool() 把子 agent 变成工具，Manager 保持控制权。
# 对比 handoff：handoff 是"你去做"，as_tool 是"帮我做"。

async def experiment_hierarchical() -> ExperimentResult:
    """层级模式：Manager 调度 Worker，Worker 作为 tools。"""
    r = ExperimentResult(pattern="Hierarchical")
    start = time.time()
    provider = make_provider()

    # Worker agents。
    trend_researcher = Agent(
        name="趋势研究员",
        instructions="你是技术趋势研究员。用搜索工具调研用户指定的技术方向，返回 100 字以内的调研摘要。",
        tools=[search_tech_trends],
        model=MODEL,
        handoff_description="调研某个技术方向的最新趋势",
    )

    code_reviewer = Agent(
        name="代码审查员",
        instructions="你是代码审查专家。分析给定的代码片段，指出潜在问题和改进建议。",
        tools=[analyze_code_quality],
        model=MODEL,
        handoff_description="审查代码质量",
    )

    # Manager agent，把 Worker 作为 tool 使用。
    manager = Agent(
        name="项目经理",
        instructions=(
            "你是一个技术项目经理。用户会给你一个综合性任务，你需要：\n"
            "1. 分析任务需要哪些子工作\n"
            "2. 调用合适的专家（趋势研究员或代码审查员）完成子工作\n"
            "3. 综合各专家的结果，给出最终报告\n\n"
            "你有两个专家可以调用。合理分配任务。"
        ),
        tools=[
            trend_researcher.as_tool(
                tool_name="consult_trend_researcher",
                tool_description="咨询趋势研究员，调研某个技术方向的最新趋势。输入为研究方向的描述。",
            ),
            code_reviewer.as_tool(
                tool_name="consult_code_reviewer",
                tool_description="咨询代码审查员，审查一段代码的质量。输入为代码片段。",
            ),
        ],
        model=MODEL,
    )

    task = (
        "帮我做两件事：\n"
        "1. 调研一下 vLLM 的最新发展趋势\n"
        "2. 审查这段代码的质量：\n"
        "```python\n"
        "def process(data):\n"
        "    try:\n"
        "        result = eval(data['expression'])\n"
        "        return result\n"
        "    except:\n"
        "        return None\n"
        "```"
    )

    r.log("Manager 收到综合任务，开始调度 Worker...")
    result = await Runner.run(
        manager,
        task,
        run_config={"model_provider": provider},
    )

    r.final_output = str(result.final_output)
    r.log(f"Manager 完成调度，最终由 [{result.last_agent.name}] 输出")
    r.elapsed_seconds = time.time() - start
    return r


# ===========================================================================
# 运行所有实验 + 对比总结
# ===========================================================================

async def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY。")

    print("第7周 Day 2：Multi-Agent 协作模式实验")
    print("=" * 70)

    experiments = [
        ("1. Serial Handoff（串行交接）", experiment_serial_handoff),
        ("2. Parallel Fan-out（并行分发）", experiment_parallel_fanout),
        ("3. Debate（辩论）", experiment_debate),
        ("4. Voting（投票）", experiment_voting),
        ("5. Hierarchical（层级 Manager-Worker）", experiment_hierarchical),
    ]

    results: list[ExperimentResult] = []

    for title, func in experiments:
        print(f"\n{'=' * 70}")
        print(title)
        print(f"{'=' * 70}")
        try:
            r = await func()
            results.append(r)
            print(f"\n  最终输出：\n{r.final_output[:400]}")
        except Exception as e:
            print(f"  ❌ 实验失败：{type(e).__name__}: {e}")

    # 汇总对比。
    print(f"\n\n{'=' * 70}")
    print("模式对比总结")
    print(f"{'=' * 70}")

    print(f"\n{'模式':<25} {'Agent调用':<12} {'耗时':<10} {'步骤数':<10}")
    print("-" * 57)
    for r in results:
        print(f"{r.pattern:<25} {r.agent_calls:<12} {r.elapsed_seconds:<10.1f} {len(r.steps):<10}")

    print("""
选型指南：

  Serial Handoff（串行交接）
    ✅ 流水线式任务：每步有明确输入输出（调研→分析→总结）
    ⚠️ 上游错误会传播到下游，不适合需要回溯的场景

  Parallel Fan-out（并行分发）
    ✅ 子任务互相独立、可并行（多方向调研、多语言翻译）
    ⚠️ 需要 synthesizer 整合结果，整合质量依赖 synthesizer 的能力

  Debate（辩论）
    ✅ 需要多角度分析的决策场景（技术选型、风险评估）
    ⚠️ 辩论可能流于表面，需要好的 judge prompt

  Voting（投票）
    ✅ 需要高可靠性的关键判断（分类、选择题）
    ⚠️ 三个 agent 都错的情况下投票也救不了，成本是 3 倍

  Hierarchical（层级 Manager-Worker）
    ✅ 复杂任务需要动态分工（Manager 根据情况决定调谁）
    ⚠️ Manager 的调度能力是瓶颈；Worker 返回的结果 Manager 要能正确整合

核心原则（来自 Anthropic Building Effective Agents）：
  - 能用 workflow（确定性流程）解决就不用 agent
  - 多 Agent 的协调成本（额外 token + 延迟 + 复杂度）大于收益时，用单 Agent
  - 复杂系统通常是 workflow + agent 的混合架构
""")


if __name__ == "__main__":
    asyncio.run(main())
