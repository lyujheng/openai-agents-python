"""
第6周 Day 4：Tracing 深入与自定义 Processor

核心认知：
  Tracing = 给 agent 执行过程加可观测性。
  SDK 把每次执行切成两层：
    - Trace：一次完整的工作流（比如"处理一个用户请求"）
    - Span：Trace 里的单个操作（调用 LLM、执行工具、handoff 等）

架构（从里到外）：
  SpanData（数据）→ Span（生命周期管理）→ TracingProcessor（消费接口）
                                       ↑
                              BatchTraceProcessor（SDK 默认：批量发送到 OpenAI）

关键 SpanData 类型（span_data.py）：
  - AgentSpanData：agent 执行，记录 name/tools/handoffs
  - GenerationSpanData：LLM 调用，记录 input/output/model/usage
  - FunctionSpanData：工具调用，记录 name/input/output
  - HandoffSpanData：agent 切换，记录 from_agent/to_agent
  - GuardrailSpanData：护栏检查，记录 name/triggered

自定义 Processor 的用途：
  - 打印到控制台（调试）
  - 写入本地文件（离线分析）
  - 推送到自建监控系统
  - 过滤特定 span 类型

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python custom_tracing.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from agents import Agent, Runner
from agents.models.openai_provider import OpenAIProvider
from agents.tracing import TracingProcessor, add_trace_processor
from agents.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
)
from agents.tracing.spans import Span
from agents.tracing.traces import Trace

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _span_label(span: Span[Any]) -> str:
    """把 span 的 span_data 类型和关键字段提炼成一行标签。"""
    sd = span.span_data
    if isinstance(sd, AgentSpanData):
        return f"agent:{sd.name}"
    if isinstance(sd, GenerationSpanData):
        model = sd.model or "?"
        usage = sd.usage or {}
        tokens = usage.get("total_tokens", "?")
        return f"generation model={model} tokens={tokens}"
    if isinstance(sd, FunctionSpanData):
        return f"function:{sd.name}({sd.input}) → {str(sd.output)[:60]}"
    if isinstance(sd, HandoffSpanData):
        return f"handoff:{sd.from_agent} → {sd.to_agent}"
    if isinstance(sd, GuardrailSpanData):
        triggered = "✓触发" if sd.triggered else "✗未触发"
        return f"guardrail:{sd.name} {triggered}"
    # 其他类型（response、custom 等）降级显示。
    return f"{type(sd).__name__}"


# ---------------------------------------------------------------------------
# Processor 1：控制台打印（最简版本，理解接口用）
# ---------------------------------------------------------------------------


class PrintProcessor(TracingProcessor):
    """
    把每个 trace/span 的关键信息实时打印到控制台。

    最简单的自定义 processor，用于理解 SDK 的回调时序：
      on_trace_start → on_span_start → on_span_end → on_trace_end
    """

    def on_trace_start(self, trace: Trace) -> None:
        print(f"\n[TRACE START] id={trace.trace_id[:16]}... name={trace.name!r}")

    def on_trace_end(self, trace: Trace) -> None:
        print(f"[TRACE END]   id={trace.trace_id[:16]}...")

    def on_span_start(self, span: Span[Any]) -> None:
        label = _span_label(span)
        print(f"  [SPAN START] {label}")

    def on_span_end(self, span: Span[Any]) -> None:
        label = _span_label(span)
        # started_at 和 ended_at 是 ISO8601 字符串；用 time 差算耗时。
        started = getattr(span, "started_at", None)
        ended = getattr(span, "ended_at", None)
        duration = ""
        if started and ended:
            # 两个都是毫秒精度的时间戳字符串，直接用 span 属性里的 duration 更简单。
            dur_ms = getattr(span, "duration_ms", None)
            if dur_ms is not None:
                duration = f" ({dur_ms:.0f}ms)"
        print(f"  [SPAN END]   {label}{duration}")

    def shutdown(self) -> None:
        pass  # 无资源需要释放。

    def force_flush(self) -> None:
        pass  # 无缓冲区需要冲刷。


# ---------------------------------------------------------------------------
# Processor 2：统计收集（统计各类 span 的次数和耗时）
# ---------------------------------------------------------------------------


@dataclass
class SpanStats:
    """单一 span 类型的统计数据。"""
    count: int = 0
    total_tokens: int = 0
    errors: int = 0


class StatsProcessor(TracingProcessor):
    """
    收集 span 级别的统计数据：次数、token 消耗、错误数。

    用途：在 eval 或批量实验中量化 agent 的资源消耗。
    """

    def __init__(self) -> None:
        # key = span 类型名（"agent"/"generation"/"function"/...）
        self._stats: dict[str, SpanStats] = {}
        self._trace_count = 0

    def on_trace_start(self, trace: Trace) -> None:
        self._trace_count += 1

    def on_trace_end(self, trace: Trace) -> None:
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        sd = span.span_data
        span_type = sd.type  # "agent" / "generation" / "function" / ...
        stats = self._stats.setdefault(span_type, SpanStats())
        stats.count += 1

        # 记录 token 消耗（只有 GenerationSpanData 有 usage）。
        if isinstance(sd, GenerationSpanData) and sd.usage:
            stats.total_tokens += sd.usage.get("total_tokens", 0)

        # 记录错误。
        if getattr(span, "error", None):
            stats.errors += 1

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass

    def print_report(self) -> None:
        """打印统计报告。"""
        print(f"\n{'='*50}")
        print(f"Tracing 统计报告（共 {self._trace_count} 个 trace）")
        print(f"{'='*50}")
        if not self._stats:
            print("  无任何 span 记录。")
            return
        print(f"  {'span 类型':<20} {'次数':>6} {'token':>8} {'错误':>6}")
        print(f"  {'-'*44}")
        for span_type, stats in sorted(self._stats.items()):
            tokens = str(stats.total_tokens) if stats.total_tokens else "-"
            errors = str(stats.errors) if stats.errors else "-"
            print(f"  {span_type:<20} {stats.count:>6} {tokens:>8} {errors:>6}")
        print()


# ---------------------------------------------------------------------------
# Processor 3：事件记录（把每个 span 存成列表，方便事后分析）
# ---------------------------------------------------------------------------


@dataclass
class SpanRecord:
    """一个 span 的快照，用于事后分析。"""
    span_type: str
    label: str
    started_at: str | None
    ended_at: str | None
    duration_ms: float | None
    error: dict | None


class RecordingProcessor(TracingProcessor):
    """
    记录完整的 span 序列，可用于：
    - 断言 agent 是否按预期顺序调用了工具
    - 检测重复调用
    - 验证 handoff 路径
    """

    def __init__(self) -> None:
        self.records: list[SpanRecord] = []

    def on_trace_start(self, trace: Trace) -> None:
        self.records.clear()  # 每个 trace 重置。

    def on_trace_end(self, trace: Trace) -> None:
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        self.records.append(SpanRecord(
            span_type=span.span_data.type,
            label=_span_label(span),
            started_at=getattr(span, "started_at", None),
            ended_at=getattr(span, "ended_at", None),
            duration_ms=getattr(span, "duration_ms", None),
            error=getattr(span, "error", None),
        ))

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass

    def function_calls(self) -> list[str]:
        """返回所有 function span 的工具名列表（按调用顺序）。"""
        return [
            r.label.split(":")[1].split("(")[0]
            for r in self.records
            if r.span_type == "function"
        ]

    def print_timeline(self) -> None:
        """按顺序打印 span 时间线。"""
        print(f"\n{'='*50}")
        print("Span 时间线")
        print(f"{'='*50}")
        for i, r in enumerate(self.records):
            dur = f"{r.duration_ms:.0f}ms" if r.duration_ms is not None else "?"
            err = " ⚠ERROR" if r.error else ""
            print(f"  {i+1:2d}. [{r.span_type:<12}] {r.label[:50]:<50}  {dur:>8}{err}")
        print()


# ---------------------------------------------------------------------------
# 实验
# ---------------------------------------------------------------------------

def setup_processors() -> tuple[StatsProcessor, RecordingProcessor]:
    """注册自定义 processor，返回引用以便事后查询。"""
    # PrintProcessor：实时打印每个 span（调试用）。
    add_trace_processor(PrintProcessor())

    # StatsProcessor：统计各类型 span 的次数和 token。
    stats = StatsProcessor()
    add_trace_processor(stats)

    # RecordingProcessor：记录完整时间线，支持事后断言。
    recorder = RecordingProcessor()
    add_trace_processor(recorder)

    return stats, recorder


async def run_experiment() -> None:
    """
    运行一个带工具的 agent，观察 tracing 输出。

    工具：
      - get_city_info：查城市人口/面积信息（模拟）
      - simple_calc：简单数学计算（eval-based，仅限数字运算）
    """
    from agents.tool import function_tool

    @function_tool
    def get_city_info(city: str, field: str) -> str:
        """查询城市的基本信息。

        Args:
            city: 城市名，支持：北京、上海、东京
            field: 查询字段，支持：人口、面积
        """
        data = {
            "北京": {"人口": "2189万", "面积": "16410平方公里"},
            "上海": {"人口": "2487万", "面积": "6340平方公里"},
            "东京": {"人口": "1396万", "面积": "2191平方公里"},
        }
        city_data = data.get(city)
        if city_data is None:
            return f"没有 {city!r} 的数据。"
        value = city_data.get(field)
        if value is None:
            return f"没有 {city!r} 的 {field!r} 数据。"
        return f"{city}的{field}是 {value}。"

    @function_tool
    def simple_calc(expression: str) -> str:
        """计算一个简单的数学表达式。

        Args:
            expression: 只含数字和 +-*/ 的表达式，比如 '2189 + 2487'
        """
        allowed = set("0123456789 +-*/.() ")
        if not all(c in allowed for c in expression):
            return f"不支持该表达式：{expression!r}"
        try:
            return str(eval(expression))  # noqa: S307 — 教学用途，已做字符过滤。
        except Exception as e:
            return f"计算出错：{e}"

    provider = OpenAIProvider(
        base_url="https://zyapi.xmsxb.com/v1",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        use_responses=False,  # 使用 Chat Completions API，兼容第三方中转。
    )

    agent = Agent(
        name="城市分析助手",
        instructions=(
            "你是一个城市信息分析助手，可以查询城市人口和面积，也可以做简单计算。\n"
            "回答要包含具体数据，如果需要比较，先分别查询再计算或直接比较。"
        ),
        tools=[get_city_info, simple_calc],
        model="mco-4",
    )

    task = "北京和上海的人口加起来是多少？"
    print(f"\n任务：{task}")
    print("=" * 60)

    result = await Runner.run(agent, task, run_config={"model_provider": provider})  # type: ignore[arg-type]
    print(f"\n最终回答：{result.final_output}")


async def main() -> None:
    """主入口：注册 processor，运行实验，输出报告。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY。")

    print("Day 4：自定义 Tracing Processor 实验")
    print("=" * 60)
    print("注册三个 Processor：PrintProcessor / StatsProcessor / RecordingProcessor")

    stats, recorder = setup_processors()

    await run_experiment()

    # 事后分析。
    stats.print_report()
    recorder.print_timeline()

    # 断言工具调用顺序（验证 agent 按预期行动）。
    tool_calls = recorder.function_calls()
    print(f"工具调用序列：{tool_calls}")
    assert "get_city_info" in tool_calls, "预期 agent 调用了 get_city_info"
    print("✅ 工具调用断言通过")


# ---------------------------------------------------------------------------
# 理论小结（放在这里方便对照代码看）
# ---------------------------------------------------------------------------

TRACING_NOTES = """
Tracing 架构要点
=================

1. 层级关系
   Trace（工作流）→ Span（单个操作）→ SpanData（操作数据）

2. TracingProcessor 接口（4 个回调）
   on_trace_start(trace)   ← trace 开始时触发
   on_span_start(span)     ← span 开始时触发
   on_span_end(span)       ← span 结束时触发（数据已填充）
   on_trace_end(trace)     ← trace 结束时触发

3. 注册方式
   add_trace_processor(my_processor)  ← 追加，不替换默认 processor
   set_trace_provider(provider)       ← 完全替换 provider（需要自建）

4. SDK 默认行为
   BatchTraceProcessor + BackendSpanExporter → 批量发送到 OpenAI tracing API
   默认 processor 在 get_trace_provider() 首次调用时懒初始化（避免导入时副作用）

5. Processor 设计原则（来自 tracing-lifecycle.md）
   - 不能影响 agent 执行结果（非侵入）
   - 回调里不要抛异常（非致命）
   - 不要长时间阻塞（快速返回）

6. SpanData 类型速查
   type="agent"      → AgentSpanData（name, tools, handoffs）
   type="generation" → GenerationSpanData（model, usage, input/output）
   type="function"   → FunctionSpanData（name, input, output）
   type="handoff"    → HandoffSpanData（from_agent, to_agent）
   type="guardrail"  → GuardrailSpanData（name, triggered）
   type="response"   → ResponseSpanData（response_id, usage）
   type="custom"     → CustomSpanData（name, data）
"""


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    print(TRACING_NOTES)
