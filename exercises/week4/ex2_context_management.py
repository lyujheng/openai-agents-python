"""
第4周 Day 2：Context 管理

核心问题：agent 的 messages 列表会随着工具调用不断增长，最终超出 context window。
三种策略从简单到复杂：
  1. 滑动窗口 — 只保留最近 N 轮，丢弃早期历史
  2. Token 计数 + 精确截断 — 实时统计 token 数，快到上限时裁剪
  3. 摘要压缩 — 让 LLM 把早期对话总结成摘要，替换原始消息

SDK 的 Session 接口提供了存储/读取，但截断策略由使用者自己实现。
今天就是动手实现这三种策略，理解各自的取舍。

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python ex2_context_management.py
"""

import json
import os
import sys
import time

from openai import OpenAI

# 复用 week2 的 MiniAgent 基础设施。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week2"))
from mini_agent import MiniAgent, func_to_schema, call_llm_with_retry


client = OpenAI(
    base_url="https://zyapi.xmsxb.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
)
MODEL = "mco-4"


# ---------------------------------------------------------------------------
# 工具：用于模拟 token 消耗的工具
# ---------------------------------------------------------------------------

def search_knowledge(query: str) -> str:
    """搜索知识库，返回相关内容。

    Args:
        query: 搜索关键词
    """
    # 模拟返回大段文本，快速撑大 context。
    knowledge = {
        "python": (
            "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。"
            "Python 的设计哲学强调代码的可读性和简洁性，使用缩进来表示代码块。"
            "Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。"
            "Python 的标准库非常丰富，涵盖了文件 I/O、网络编程、数据库接口等。"
            "Python 在人工智能、数据科学、Web 开发、自动化等领域广泛应用。"
            "主要实现包括 CPython、PyPy、Jython、IronPython 等。"
            "Python 3 是当前主流版本，Python 2 已于 2020 年停止维护。"
        ),
        "rust": (
            "Rust 是一种系统编程语言，由 Mozilla 研究院开发，首次发布于 2015 年。"
            "Rust 的核心特性是内存安全和零成本抽象，通过所有权系统在编译期保证内存安全。"
            "Rust 没有垃圾回收器，通过借用检查器（borrow checker）管理内存。"
            "Rust 的性能与 C/C++ 相当，同时提供了更高级的抽象能力。"
            "Rust 在系统编程、WebAssembly、嵌入式开发、CLI 工具等领域广泛应用。"
            "Rust 连续多年被 Stack Overflow 评为最受喜爱的编程语言。"
        ),
        "agent": (
            "LLM Agent 是一种利用大语言模型作为核心推理引擎的自主系统。"
            "Agent 的核心循环：感知（接收输入）→ 推理（LLM 思考）→ 行动（调用工具）→ 观察（获取结果）。"
            "主要组件：规划（Planning）、记忆（Memory）、工具使用（Tool Use）。"
            "常见架构模式：ReAct、Plan-and-Solve、Tree of Thoughts、Reflexion。"
            "生产 agent 需要考虑：输出稳定性、context 管理、安全防护、可观测性。"
            "主流框架：OpenAI Agents SDK、LangGraph、CrewAI、AutoGen 等。"
        ),
    }
    # 模糊匹配。
    for key, content in knowledge.items():
        if key in query.lower():
            return content
    return f"未找到与 '{query}' 相关的知识。可搜索：python、rust、agent"


def get_detail(topic: str, aspect: str) -> str:
    """获取某个主题的某个方面的详细信息。

    Args:
        topic: 主题，比如 python、rust
        aspect: 方面，比如 history、features、ecosystem
    """
    return (
        f"关于 {topic} 的 {aspect}：这是一段详细的说明文字，"
        f"包含了 {topic} 在 {aspect} 方面的丰富信息。"
        f"在实际场景中，工具可能返回几千字的内容，"
        f"这会快速消耗 context window 的空间。"
        f"{'详细内容 ' * 50}"  # 故意填充，模拟大段返回。
    )


# ---------------------------------------------------------------------------
# 策略 1：滑动窗口
# ---------------------------------------------------------------------------
# 最简单的策略：只保留 system prompt + 最近 N 轮消息。
# 优点：实现简单，效果可预测。
# 缺点：早期的重要信息会被丢掉。

def sliding_window(messages: list[dict], max_recent: int = 10) -> list[dict]:
    """
    滑动窗口截断：保留 system prompt + 最近 max_recent 条消息。

    为什么保留 system prompt？
    它定义了 agent 的身份和行为规则，丢了 agent 就失忆了。
    """
    # 第一条消息通常是 system prompt。
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= max_recent:
        return messages  # 还没超，不用截。

    # 保留 system + 最近 N 条。
    truncated = system_msgs + non_system[-max_recent:]
    dropped = len(non_system) - max_recent
    print(f"  [滑动窗口] 丢弃了 {dropped} 条早期消息，保留最近 {max_recent} 条")
    return truncated


# ---------------------------------------------------------------------------
# 策略 2：Token 计数 + 精确截断
# ---------------------------------------------------------------------------
# 根据 token 数量来决定截断，比按消息条数更精确。
# 因为一条消息可能是 10 个 token（短回复），也可能是 2000 个 token（文件内容）。

def estimate_tokens(text: str) -> int:
    """
    简易 token 估算。

    精确计数需要用 tiktoken 库（OpenAI 的 tokenizer）。
    这里用经验公式粗估：
    - 英文大约 1 token ≈ 4 字符
    - 中文大约 1 token ≈ 1.5 字符
    - 混合内容取个中间值
    """
    # 粗略估算：中文字符按 1.5 字符/token，其他按 4 字符/token。
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def token_based_truncation(
    messages: list[dict],
    max_tokens: int = 4000,
) -> list[dict]:
    """
    基于 token 数量的截断：从最早的非 system 消息开始丢，直到总 token 数低于上限。
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    # 计算 system prompt 占用的 token。
    system_tokens = sum(estimate_tokens(m.get("content", "")) for m in system_msgs)
    available = max_tokens - system_tokens

    # 从后往前累计，找到能放下的消息。
    kept = []
    total = 0
    for msg in reversed(non_system):
        msg_tokens = estimate_tokens(msg.get("content", ""))
        if total + msg_tokens > available:
            break
        kept.append(msg)
        total += msg_tokens

    kept.reverse()
    dropped = len(non_system) - len(kept)
    if dropped > 0:
        print(f"  [Token 截断] 丢弃了 {dropped} 条早期消息，"
              f"保留 {len(kept)} 条（~{total} tokens）")
    return system_msgs + kept


# ---------------------------------------------------------------------------
# 策略 3：摘要压缩
# ---------------------------------------------------------------------------
# 把早期对话交给 LLM 总结成一段摘要，用摘要替换原始消息。
# 优点：保留了早期的关键信息。
# 缺点：摘要本身消耗一次 LLM 调用，且可能丢失细节。

def summarize_and_compress(
    messages: list[dict],
    max_recent: int = 6,
) -> list[dict]:
    """
    摘要压缩：把早期消息总结成一段摘要，替换原始消息。

    流程：
    1. 把 messages 分成 [system] + [早期消息] + [最近 N 条]
    2. 让 LLM 把 [早期消息] 总结成一段摘要
    3. 返回 [system] + [摘要] + [最近 N 条]
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= max_recent:
        return messages  # 不够多，不需要压缩。

    # 划分：早期 vs 最近。
    early = non_system[:-max_recent]
    recent = non_system[-max_recent:]

    # 把早期消息格式化成文本，交给 LLM 总结。
    early_text = ""
    for msg in early:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            # 截断过长的单条消息，避免总结请求本身超限。
            if len(content) > 500:
                content = content[:500] + "..."
            early_text += f"[{role}] {content}\n"

    print(f"  [摘要压缩] 正在总结 {len(early)} 条早期消息...")

    # 调 LLM 做总结。
    summary_response = call_llm_with_retry(
        client,
        model=MODEL,
        messages=[{
            "role": "user",
            "content": (
                "请用 2-3 句话总结以下对话历史的关键信息，"
                "包括讨论了什么主题、得到了什么结论、还有什么未完成的任务。\n\n"
                f"对话历史：\n{early_text}"
            ),
        }],
    )
    summary = summary_response.choices[0].message.content or "（摘要生成失败）"
    print(f"  [摘要压缩] 摘要：{summary[:100]}...")

    # 用摘要消息替换早期消息。
    summary_msg = {
        "role": "user",
        "content": f"[之前对话的摘要] {summary}",
    }
    return system_msgs + [summary_msg] + recent


# ---------------------------------------------------------------------------
# 对比测试
# ---------------------------------------------------------------------------

def simulate_long_conversation() -> list[dict]:
    """模拟一次包含多轮工具调用的长对话，生成膨胀的 messages 列表。"""
    messages = [
        {"role": "system", "content": "你是一个知识问答助手，可以搜索知识库。"},
        {"role": "user", "content": "帮我调研 Python 语言"},
    ]

    # 模拟 5 轮工具调用，每轮产生 3 条消息（assistant + tool_result + 后续提问）。
    topics = [
        ("python", "基本介绍"),
        ("python", "生态系统"),
        ("rust", "与 Python 对比"),
        ("agent", "LLM Agent 架构"),
        ("python", "性能优化"),
    ]

    for topic, desc in topics:
        # LLM 决定调工具。
        messages.append({
            "role": "assistant",
            "content": f"让我搜索一下{desc}的相关信息。",
        })
        # 工具返回结果（大段文本）。
        result = search_knowledge(topic)
        detail = get_detail(topic, desc)
        messages.append({
            "role": "tool",
            "content": result + "\n\n" + detail,
        })
        # 用户追问。
        messages.append({
            "role": "user",
            "content": f"好的，关于{desc}，能再详细说说吗？",
        })

    return messages


if __name__ == "__main__":
    print("Day 2：Context 管理对比实验\n")

    # 生成一个膨胀的消息列表。
    messages = simulate_long_conversation()
    total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
    print(f"原始消息：{len(messages)} 条，约 {total_tokens} tokens\n")

    # 策略 1：滑动窗口。
    print("--- 策略 1：滑动窗口（保留最近 6 条）---")
    result1 = sliding_window(messages, max_recent=6)
    tokens1 = sum(estimate_tokens(m.get("content", "")) for m in result1)
    print(f"  结果：{len(result1)} 条，约 {tokens1} tokens\n")

    # 策略 2：Token 截断。
    print("--- 策略 2：Token 截断（上限 2000 tokens）---")
    result2 = token_based_truncation(messages, max_tokens=2000)
    tokens2 = sum(estimate_tokens(m.get("content", "")) for m in result2)
    print(f"  结果：{len(result2)} 条，约 {tokens2} tokens\n")

    # 策略 3：摘要压缩。
    print("--- 策略 3：摘要压缩（保留最近 6 条，早期做摘要）---")
    result3 = summarize_and_compress(messages, max_recent=6)
    tokens3 = sum(estimate_tokens(m.get("content", "")) for m in result3)
    print(f"  结果：{len(result3)} 条，约 {tokens3} tokens\n")

    # 汇总对比。
    print("=" * 60)
    print("汇总对比")
    print("=" * 60)
    print(f"  {'策略':<20s} {'消息数':>6s} {'约 Tokens':>10s} {'信息保留':>10s}")
    print(f"  {'原始':<20s} {len(messages):>6d} {total_tokens:>10d} {'完整':>10s}")
    print(f"  {'滑动窗口':<20s} {len(result1):>6d} {tokens1:>10d} {'仅最近':>10s}")
    print(f"  {'Token 截断':<18s} {len(result2):>6d} {tokens2:>10d} {'仅最近':>10s}")
    print(f"  {'摘要压缩':<20s} {len(result3):>6d} {tokens3:>10d} {'摘要+最近':>10s}")

    print("\n各策略的取舍：")
    print("  滑动窗口：最简单，但按条数截断不精确（一条可能 10 token 或 2000 token）")
    print("  Token 截断：更精确，但需要 token 计数（精确计数要用 tiktoken）")
    print("  摘要压缩：保留最多信息，但额外消耗一次 LLM 调用，且摘要可能丢细节")
    print("  生产环境通常组合使用：单条消息先截断（你 week2 的 _truncate），")
    print("  整体再用 token 截断或摘要压缩")
