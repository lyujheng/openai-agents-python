"""
第7周 Day 3-4：Agentic RAG

从简单 RAG 到 Agentic RAG 的进化：
  简单 RAG：用户问题 → 向量检索 → 拼接上下文 → LLM 回答（固定流程）
  Agentic RAG：agent 自主决定要不要检索、检索什么、结果够不够、要不要换关键词重试

本练习实现：
  1. 简单 RAG（baseline）—— 一次检索就回答
  2. Agentic RAG —— agent 自主循环检索、查询重写、多源聚合
  3. 对比实验 —— 同一组问题，量化两种方案的差异

核心模式：
  - 自适应检索：agent 根据问题复杂度决定检索深度
  - 查询分解：把复杂问题拆成多个子查询
  - 查询重写：检索结果不好时自动重写查询
  - 多源聚合：从多个数据源检索并合并结果
  - 验证回路：检查回答是否有文档支持

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python day3_agentic_rag.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from agents import Agent, Runner, function_tool
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
# 模拟文档知识库
# ---------------------------------------------------------------------------
# 用字典模拟向量检索。真实系统用 chromadb / FAISS 等向量数据库。
# 每个文档有 id、标题、内容、标签。

DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Python asyncio 基础",
        "tags": ["python", "async", "concurrency"],
        "content": (
            "asyncio 是 Python 的异步 IO 框架，基于事件循环（event loop）。"
            "核心概念：coroutine（协程，用 async def 定义）、await（挂起等待）、Task（并发任务）。"
            "事件循环是单线程的，通过协作式调度实现并发。适合 IO 密集型任务（网络请求、文件读写），"
            "不适合 CPU 密集型任务（用 multiprocessing）。"
            "关键 API：asyncio.run()、asyncio.gather()、asyncio.create_task()。"
        ),
    },
    {
        "id": "doc_002",
        "title": "Python GIL 详解",
        "tags": ["python", "concurrency", "gil"],
        "content": (
            "GIL（Global Interpreter Lock）是 CPython 的全局锁，保证同一时刻只有一个线程执行字节码。"
            "影响：多线程无法利用多核做 CPU 密集型计算。"
            "历史原因：CPython 的内存管理（引用计数）不是线程安全的，GIL 是最简单的保护方式。"
            "绕过方案：multiprocessing（多进程）、C 扩展中释放 GIL、subinterpreters（Python 3.12+）。"
            "注意：IO 密集型任务中 GIL 影响很小，因为 IO 等待时 GIL 会释放。"
        ),
    },
    {
        "id": "doc_003",
        "title": "Python 多线程 vs 多进程",
        "tags": ["python", "concurrency", "threading", "multiprocessing"],
        "content": (
            "多线程（threading）：共享内存，受 GIL 限制，适合 IO 密集型。创建开销小。"
            "多进程（multiprocessing）：独立内存空间，不受 GIL 限制，适合 CPU 密集型。创建开销大。"
            "asyncio：单线程协作式并发，适合大量 IO 操作。开销最小。"
            "选择指南：IO 密集→ asyncio 或 threading；CPU 密集→ multiprocessing；混合→组合使用。"
        ),
    },
    {
        "id": "doc_004",
        "title": "Rust 所有权系统",
        "tags": ["rust", "ownership", "memory"],
        "content": (
            "Rust 的所有权系统在编译期保证内存安全，零运行时开销。三条规则：\n"
            "1. 每个值有且只有一个 owner\n"
            "2. owner 离开作用域时值被 drop（自动释放）\n"
            "3. 值可以被借用（&T 不可变借用，&mut T 可变借用），但不能同时有可变和不可变借用\n"
            "Move 语义：赋值或传参默认转移所有权（而非拷贝）。\n"
            "Clone trait：显式深拷贝。Copy trait：栈上数据自动拷贝。\n"
            "所有权系统消除了数据竞争、悬垂引用、double free 等内存安全问题。"
        ),
    },
    {
        "id": "doc_005",
        "title": "Rust 生命周期",
        "tags": ["rust", "lifetime", "borrow"],
        "content": (
            "生命周期标注 'a 告诉编译器引用的有效范围，防止悬垂引用。\n"
            "规则：函数签名中有多个引用参数时，需要标注生命周期以表达输入输出的生命周期关系。\n"
            "三条省略规则（elision rules）：\n"
            "1. 每个引用参数获得独立的生命周期\n"
            "2. 只有一个输入生命周期时，输出生命周期等于输入\n"
            "3. 方法中 &self 的生命周期赋给所有输出引用\n"
            "static 生命周期 'static：整个程序运行期间有效（如字符串字面量）。"
        ),
    },
    {
        "id": "doc_006",
        "title": "Rust 并发模型",
        "tags": ["rust", "concurrency", "send", "sync"],
        "content": (
            "Rust 通过类型系统保证并发安全。两个关键 trait：\n"
            "- Send：值可以安全地发送到另一个线程\n"
            "- Sync：值可以安全地被多个线程共享引用\n"
            "编译器自动推导 Send/Sync，不满足时编译报错。\n"
            "共享状态并发：Mutex<T>（互斥锁）+ Arc<T>（原子引用计数智能指针）。\n"
            "消息传递并发：mpsc channel（多生产者单消费者）。\n"
            "async/await：tokio 运行时提供异步支持，类似 Python 的 asyncio。"
        ),
    },
    {
        "id": "doc_007",
        "title": "LLM Agent 架构概述",
        "tags": ["llm", "agent", "architecture"],
        "content": (
            "LLM Agent 的核心循环：感知→推理→行动→观察。\n"
            "三大组件（Lilian Weng）：\n"
            "1. Planning：任务分解（CoT、ToT）、自我反思（Reflexion）\n"
            "2. Memory：短期（对话历史）+ 长期（向量数据库）\n"
            "3. Tool Use：通过 function calling / MCP 调用外部工具\n"
            "从简到繁的架构：prompt chaining → routing → agent loop → multi-agent。\n"
            "核心原则：能用 workflow 解决的不用 agent。"
        ),
    },
    {
        "id": "doc_008",
        "title": "ReAct 范式",
        "tags": ["llm", "agent", "react", "planning"],
        "content": (
            "ReAct = Reasoning + Acting。在每一步中交替进行推理和行动：\n"
            "Thought → Action → Observation → Thought → ...\n"
            "优点：推理过程透明，可解释性强。\n"
            "缺点：线性执行，无法回溯；容易陷入重复循环。\n"
            "适用场景：单步推理就能决定下一步行动的任务。\n"
            "不适用：需要全局规划、探索多条路径、或从失败中学习的任务。"
        ),
    },
    {
        "id": "doc_009",
        "title": "MCP 协议详解",
        "tags": ["mcp", "protocol", "tools"],
        "content": (
            "MCP (Model Context Protocol) 是 Anthropic 提出的开放协议。\n"
            "目标：标准化 LLM 应用与外部工具的通信。类比：USB 之于外设。\n"
            "三大原语：Resources（数据源）、Tools（可调用函数）、Prompts（预置模板）。\n"
            "传输方式：stdio（本地进程）、SSE（HTTP 长连接）、streamable-http。\n"
            "生命周期：initialize → initialized → 正常通信 → shutdown。\n"
            "安全要点：server 返回的内容应视为不可信数据。"
        ),
    },
    {
        "id": "doc_010",
        "title": "RAG 技术演进",
        "tags": ["rag", "retrieval", "llm"],
        "content": (
            "RAG (Retrieval-Augmented Generation) 的演进路线：\n"
            "1. 简单 RAG：query → 向量检索 → 拼接上下文 → LLM 生成\n"
            "2. Advanced RAG：查询改写、混合检索、重排序\n"
            "3. Agentic RAG：agent 自主决定检索策略、多轮检索、结果验证\n"
            "关键技术：HyDE（假设性文档嵌入）、Self-RAG（自我评估检索质量）、"
            "GraphRAG（知识图谱增强）。\n"
            "评估指标：检索准确率（Recall@K）、回答忠实度（Faithfulness）、回答相关性。"
        ),
    },
]


# ---------------------------------------------------------------------------
# 检索引擎（模拟）
# ---------------------------------------------------------------------------

def _keyword_match_score(query: str, doc: dict) -> float:
    """简单的关键词匹配打分。"""
    query_words = set(query.lower().replace("_", " ").split())
    score = 0.0
    # 标题匹配权重高。
    title_words = set(doc["title"].lower().split())
    score += len(query_words & title_words) * 3.0
    # 标签匹配。
    for tag in doc["tags"]:
        if tag in query.lower():
            score += 2.0
    # 内容匹配。
    content_lower = doc["content"].lower()
    for word in query_words:
        if word in content_lower:
            score += 1.0
    return score


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """检索最相关的文档（模拟向量检索）。"""
    scored = [(doc, _keyword_match_score(query, doc)) for doc in DOCUMENTS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored[:top_k] if score > 0]


# ---------------------------------------------------------------------------
# 方案一：简单 RAG（baseline）
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """RAG 运行结果。"""
    method: str
    question: str
    answer: str = ""
    retrieved_docs: list[str] = field(default_factory=list)
    retrieval_rounds: int = 0
    llm_calls: int = 0
    elapsed_seconds: float = 0.0

    def summary_line(self) -> str:
        return (
            f"[{self.method}] 检索 {self.retrieval_rounds} 轮, "
            f"命中 {len(self.retrieved_docs)} 篇, "
            f"LLM {self.llm_calls} 次, "
            f"耗时 {self.elapsed_seconds:.1f}s"
        )


async def simple_rag(question: str) -> RAGResult:
    """
    简单 RAG：一次检索 → 拼接上下文 → LLM 回答。

    固定流程，不做查询改写、不做多轮检索。
    """
    r = RAGResult(method="Simple RAG", question=question)
    start = time.time()
    provider = make_provider()

    # 一次检索。
    docs = retrieve(question, top_k=3)
    r.retrieval_rounds = 1
    r.retrieved_docs = [d["title"] for d in docs]

    # 拼接上下文。
    context = "\n\n".join(
        f"[{d['title']}]\n{d['content']}" for d in docs
    ) if docs else "未找到相关文档。"

    # LLM 回答。
    rag_agent = Agent(
        name="简单RAG",
        instructions=(
            "你是一个问答助手。根据提供的参考文档回答用户问题。\n"
            "规则：\n"
            "- 只使用参考文档中的信息回答\n"
            "- 如果文档信息不足以回答，明确说明\n"
            "- 引用具体的文档来源"
        ),
        model=MODEL,
    )

    result = await Runner.run(
        rag_agent,
        f"参考文档：\n{context}\n\n用户问题：{question}",
        run_config={"model_provider": provider},
    )

    r.answer = str(result.final_output)
    r.llm_calls = 1
    r.elapsed_seconds = time.time() - start
    return r


# ---------------------------------------------------------------------------
# 方案二：Agentic RAG
# ---------------------------------------------------------------------------
# agent 有多个检索工具，自主决定：
#   - 搜什么关键词
#   - 要不要换关键词重新搜
#   - 要不要读完整文档
#   - 信息够不够回答问题

# 用全局列表跟踪检索行为（教学用途，生产环境用 tracing）。
_retrieval_log: list[str] = []


@function_tool
def semantic_search(query: str) -> str:
    """语义搜索：用自然语言查询知识库中最相关的文档。

    Args:
        query: 搜索关键词或自然语言描述，如 'Python 并发编程'
    """
    docs = retrieve(query, top_k=3)
    _retrieval_log.append(f"semantic_search({query!r}) → {len(docs)} 篇")
    if not docs:
        return f"未找到与 '{query}' 相关的文档。请尝试换一个关键词搜索。"
    results = []
    for d in docs:
        # 返回标题和内容摘要，不是全文。
        preview = d["content"][:150] + "..." if len(d["content"]) > 150 else d["content"]
        results.append(f"[{d['id']}] {d['title']}\n{preview}")
    return "\n\n".join(results)


@function_tool
def keyword_search(keywords: str) -> str:
    """关键词搜索：用精确关键词匹配，适合搜索特定概念。

    Args:
        keywords: 用空格分隔的关键词，如 'GIL 多线程 CPython'
    """
    docs = retrieve(keywords, top_k=3)
    _retrieval_log.append(f"keyword_search({keywords!r}) → {len(docs)} 篇")
    if not docs:
        return f"未找到包含 '{keywords}' 的文档。可用的主题标签有：python, rust, llm, agent, rag, mcp, concurrency"
    results = []
    for d in docs:
        results.append(f"[{d['id']}] {d['title']} (标签: {', '.join(d['tags'])})")
    return "\n".join(results)


@function_tool
def read_document(doc_id: str) -> str:
    """读取指定文档的完整内容。

    Args:
        doc_id: 文档 ID，如 'doc_001'
    """
    for d in DOCUMENTS:
        if d["id"] == doc_id:
            _retrieval_log.append(f"read_document({doc_id}) → {d['title']}")
            return f"[{d['title']}]\n\n{d['content']}"
    _retrieval_log.append(f"read_document({doc_id}) → 未找到")
    return f"文档 {doc_id} 不存在。"


@function_tool
def list_all_topics() -> str:
    """列出知识库中所有文档的标题和标签，帮助确定搜索方向。"""
    _retrieval_log.append("list_all_topics()")
    lines = []
    for d in DOCUMENTS:
        lines.append(f"[{d['id']}] {d['title']} — 标签: {', '.join(d['tags'])}")
    return "\n".join(lines)


async def agentic_rag(question: str) -> RAGResult:
    """
    Agentic RAG：agent 自主决定检索策略。

    agent 有 4 个工具：
      - semantic_search：语义检索
      - keyword_search：关键词检索
      - read_document：读完整文档
      - list_all_topics：浏览知识库目录

    agent 自主决定用哪个工具、搜什么、搜几轮。
    """
    r = RAGResult(method="Agentic RAG", question=question)
    start = time.time()
    provider = make_provider()

    _retrieval_log.clear()

    rag_agent = Agent(
        name="Agentic RAG",
        instructions=(
            "你是一个智能研究助手，擅长通过多轮检索找到准确答案。\n\n"
            "你的检索策略：\n"
            "1. 先分析问题，确定需要搜索哪些方面的信息\n"
            "2. 用 semantic_search 做初步检索\n"
            "3. 评估检索结果是否足够回答问题\n"
            "4. 如果不够：\n"
            "   - 尝试用不同关键词（keyword_search）\n"
            "   - 如果只看到摘要，用 read_document 读完整文档\n"
            "   - 如果不确定知识库有什么，用 list_all_topics 浏览\n"
            "5. 信息充分后，综合所有检索结果给出回答\n\n"
            "回答要求：\n"
            "- 必须基于检索到的文档内容，不要编造\n"
            "- 引用具体的文档来源（如 [doc_001]）\n"
            "- 如果确实找不到答案，诚实告知\n"
            "- 对于比较类问题，要确保检索到所有需要比较的概念"
        ),
        tools=[semantic_search, keyword_search, read_document, list_all_topics],
        model=MODEL,
    )

    result = await Runner.run(
        rag_agent,
        question,
        run_config={"model_provider": provider},
        max_turns=15,
    )

    r.answer = str(result.final_output)
    r.retrieval_rounds = len(_retrieval_log)
    # 提取命中的文档标题。
    for log_entry in _retrieval_log:
        if "read_document" in log_entry and "→" in log_entry:
            doc_title = log_entry.split("→")[-1].strip()
            if doc_title != "未找到" and doc_title not in r.retrieved_docs:
                r.retrieved_docs.append(doc_title)
    r.llm_calls = len([item for item in result.new_items if hasattr(item, "raw_item")])
    r.elapsed_seconds = time.time() - start
    return r


# ---------------------------------------------------------------------------
# 对比实验
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """一组对比实验的结果。"""
    question: str
    simple: RAGResult | None = None
    agentic: RAGResult | None = None


async def run_comparison(questions: list[str]) -> list[ComparisonResult]:
    """对同一组问题运行简单 RAG 和 Agentic RAG，对比结果。"""
    comparisons: list[ComparisonResult] = []

    for i, q in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"问题 {i}: {q}")
        print(f"{'=' * 70}")

        comp = ComparisonResult(question=q)

        # 简单 RAG。
        print("\n--- Simple RAG ---")
        try:
            comp.simple = await simple_rag(q)
            print(f"  {comp.simple.summary_line()}")
            print(f"  回答: {comp.simple.answer[:200]}...")
        except Exception as e:
            print(f"  ❌ 失败: {e}")

        # Agentic RAG。
        print("\n--- Agentic RAG ---")
        try:
            comp.agentic = await agentic_rag(q)
            print(f"  {comp.agentic.summary_line()}")
            print(f"  检索日志: {_retrieval_log}")
            print(f"  回答: {comp.agentic.answer[:200]}...")
        except Exception as e:
            print(f"  ❌ 失败: {e}")

        comparisons.append(comp)

    return comparisons


def print_comparison_table(comparisons: list[ComparisonResult]) -> None:
    """打印对比汇总表。"""
    print(f"\n{'=' * 70}")
    print("对比汇总")
    print(f"{'=' * 70}")

    print(f"\n{'问题':<30} {'方案':<15} {'检索轮次':<10} {'命中文档':<10} {'LLM次数':<10} {'耗时':<8}")
    print("-" * 83)

    for comp in comparisons:
        q_short = comp.question[:28] + ".." if len(comp.question) > 30 else comp.question
        if comp.simple:
            s = comp.simple
            print(f"{q_short:<30} {'Simple':<15} {s.retrieval_rounds:<10} {len(s.retrieved_docs):<10} {s.llm_calls:<10} {s.elapsed_seconds:<8.1f}")
        if comp.agentic:
            a = comp.agentic
            print(f"{'':<30} {'Agentic':<15} {a.retrieval_rounds:<10} {len(a.retrieved_docs):<10} {a.llm_calls:<10} {a.elapsed_seconds:<8.1f}")
        print()

    print("""
关键差异：
  Simple RAG:
    - 固定检索一次，关键词直接从用户问题提取
    - 适合简单直接的问题（"Python asyncio 是什么？"）
    - 成本可预测（固定 1 次检索 + 1 次 LLM）

  Agentic RAG:
    - agent 自主决定检索策略，可以多轮检索、换关键词、读全文
    - 适合复杂问题（比较类、需要多个文档综合的问题）
    - 成本不可预测，但回答质量通常更好

  什么时候用 Agentic RAG：
    - 问题可能需要多个文档的信息才能回答
    - 用户的措辞和文档的关键词可能不一致（需要查询重写）
    - 需要从检索结果中判断信息是否充分
    - 愿意为更好的回答质量付出更多 token 成本
""")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

async def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY。")

    print("第7周 Day 3-4：Agentic RAG 实验")
    print("=" * 70)

    # 测试问题：从简单到复杂，考察两种方案的差异。
    questions = [
        # 简单问题：一次检索就能回答。
        "Python 的 GIL 是什么？有什么影响？",
        # 比较类问题：需要检索多个概念再综合。
        "对比 Python asyncio 和 Rust 的 async/await，它们有什么异同？",
        # 跨领域问题：需要从多篇文档中收集信息。
        "LLM Agent 中的 Planning 模块有哪些方法？ReAct 的局限是什么？",
        # 深度问题：可能需要查询重写。
        "如何在 Python 中实现真正的并行计算？为什么多线程不行？",
    ]

    comparisons = await run_comparison(questions)
    print_comparison_table(comparisons)


if __name__ == "__main__":
    asyncio.run(main())
