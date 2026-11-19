"""
第7周 Day 1：高级规划算法

ReAct（Thought → Action → Observation）是线性的，一步一步走。
当任务需要探索多条路径、回溯、或全局规划时就不够用了。

本练习实现三种进阶算法的简化版：
  1. Plan-and-Solve：先生成完整计划再逐步执行
  2. Tree of Thoughts (ToT)：在思维树上搜索最佳路径
  3. Reflexion：执行后自我反思，把经验存入记忆

对比维度：
  - 任务完成质量
  - LLM 调用次数（token 效率）
  - 适用场景

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python day1_planning_algorithms.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# API 配置
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url="https://zyapi.xmsxb.com/v1",
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
)
MODEL = "mco-4"


# ---------------------------------------------------------------------------
# 模拟工具集（和 week2 类似，但更聚焦规划场景）
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = {
    "python_gc": "Python 使用引用计数为主、分代回收为辅的垃圾回收机制。引用计数在对象引用归零时立即释放，分代回收处理循环引用。",
    "python_gil": "GIL（全局解释器锁）保证同一时刻只有一个线程执行 Python 字节码。CPU 密集型任务用多进程绕过，IO 密集型任务影响较小。",
    "python_async": "asyncio 是 Python 的异步 IO 框架，基于事件循环。async/await 语法让异步代码看起来像同步代码。适合 IO 密集型任务。",
    "python_decorator": "装饰器是一种语法糖，用 @decorator 形式修饰函数或类。本质是高阶函数：接收函数，返回新函数。常用于日志、鉴权、缓存。",
    "python_metaclass": "元类是类的类。type 是默认元类。用 __metaclass__ 或 metaclass= 自定义类的创建行为。常用于 ORM、单例模式等框架设计。",
    "rust_ownership": "Rust 的所有权系统：每个值有且只有一个 owner；owner 离开作用域时值被 drop。借用（&T / &mut T）不转移所有权。",
    "rust_lifetime": "生命周期标注 'a 告诉编译器引用的有效期。函数签名中出现多个引用时需要标注，让编译器检查借用是否合法。",
    "vllm_paged_attention": "PagedAttention 把 KV cache 分成固定大小的 block，像操作系统管理虚拟内存一样动态分配。消除了显存碎片，让 batch size 更大。",
}


def search_knowledge(query: str) -> str:
    """在知识库中搜索相关信息。"""
    query_lower = query.lower()
    results = []
    for key, value in KNOWLEDGE_BASE.items():
        # 简单的关键词匹配。
        if any(word in key for word in query_lower.replace("_", " ").split()):
            results.append(f"[{key}] {value}")
    if not results:
        return "未找到相关信息。"
    return "\n\n".join(results)


def compare_concepts(concept_a: str, concept_b: str) -> str:
    """对比两个概念。"""
    info_a = search_knowledge(concept_a)
    info_b = search_knowledge(concept_b)
    return f"--- {concept_a} ---\n{info_a}\n\n--- {concept_b} ---\n{info_b}"


def summarize_text(text: str) -> str:
    """将文本压缩为不超过 100 字的摘要（模拟）。"""
    if len(text) <= 100:
        return text
    return text[:97] + "..."


TOOLS = {
    "search_knowledge": search_knowledge,
    "compare_concepts": compare_concepts,
    "summarize_text": summarize_text,
}

# 给 LLM 看的工具签名描述，比只给名字列表靠谱得多。
TOOLS_DESCRIPTION = """可用工具（严格使用以下参数名）：
- search_knowledge(query: str) — 在知识库中搜索，query 是搜索关键词
- compare_concepts(concept_a: str, concept_b: str) — 对比两个概念，只接受这两个参数
- summarize_text(text: str) — 压缩文本为摘要"""


def call_tool(tool_name: str, tool_args: dict | str) -> str:
    """安全地调用工具，处理 LLM 传参不规范的情况。"""
    if tool_name not in TOOLS:
        return f"未知工具: {tool_name}"
    fn = TOOLS[tool_name]
    if isinstance(tool_args, str):
        return fn(tool_args)
    # LLM 经常用错参数名，做一层适配。
    import inspect
    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())
    # 如果参数名完全匹配，直接调用。
    if set(tool_args.keys()).issubset(set(param_names)):
        return fn(**tool_args)
    # 否则按位置顺序填充。
    positional_values = list(tool_args.values())
    bound_args = {}
    for i, name in enumerate(param_names):
        if i < len(positional_values):
            bound_args[name] = positional_values[i]
    return fn(**bound_args)


def call_llm(messages: list[dict], temperature: float = 0.0) -> str:
    """统一的 LLM 调用入口，返回纯文本。"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def call_llm_json(messages: list[dict], temperature: float = 0.0) -> dict:
    """调用 LLM 并解析 JSON 返回。如果解析失败做一次清洗重试。"""
    raw = call_llm(messages, temperature)
    # 清洗 markdown 代码块包装。
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # 去掉首尾的 ``` 行。
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 给 LLM 一次纠错机会。
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "你的回复不是合法 JSON，请只返回 JSON，不要加任何其他内容。"},
        ]
        raw2 = call_llm(retry_messages, temperature=0.0)
        cleaned2 = raw2.strip()
        if cleaned2.startswith("```"):
            lines2 = cleaned2.split("\n")
            lines2 = [l for l in lines2 if not l.strip().startswith("```")]
            cleaned2 = "\n".join(lines2)
        return json.loads(cleaned2)


# ---------------------------------------------------------------------------
# 指标收集
# ---------------------------------------------------------------------------

@dataclass
class PlanningMetrics:
    """记录一次规划算法运行的指标。"""
    algorithm: str
    task: str
    result: str = ""
    llm_calls: int = 0
    tool_calls: int = 0
    elapsed_seconds: float = 0.0
    steps_log: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.steps_log.append(msg)
        print(f"  [{self.algorithm}] {msg}")

    def summary(self) -> str:
        return (
            f"算法: {self.algorithm}\n"
            f"LLM 调用: {self.llm_calls} 次\n"
            f"工具调用: {self.tool_calls} 次\n"
            f"耗时: {self.elapsed_seconds:.1f}s\n"
            f"步骤数: {len(self.steps_log)}\n"
            f"结果前 200 字: {self.result[:200]}"
        )


# ===========================================================================
# 算法一：Plan-and-Solve
# ===========================================================================
# 思路：先让 LLM 生成完整执行计划（步骤列表），再逐步执行。
# 优点：全局视野、步骤有序。
# 缺点：计划可能过时（执行中获得新信息后无法调整）。

def plan_and_solve(task: str) -> PlanningMetrics:
    """Plan-and-Solve：先做计划，再逐步执行。"""
    m = PlanningMetrics(algorithm="Plan-and-Solve", task=task)
    start = time.time()

    # --- 阶段 1：生成计划 ---
    plan_prompt = [
        {"role": "system", "content": (
            "你是一个规划助手。用户会给你一个研究任务，你需要拆解成 3-6 个执行步骤。\n"
            "每个步骤包含：step_number、action（search_knowledge / compare_concepts / summarize_text / think）、"
            "args（工具参数，think 时为思考内容描述）、purpose（这步是为了什么）。\n"
            f"{TOOLS_DESCRIPTION}\n"
            "只返回 JSON 数组，不要加任何解释。"
        )},
        {"role": "user", "content": task},
    ]
    m.log("生成执行计划...")
    plan = call_llm_json(plan_prompt)
    m.llm_calls += 1

    if not isinstance(plan, list):
        # 有些模型会包一层 {"steps": [...]}。
        plan = plan.get("steps", plan.get("plan", [plan]))

    m.log(f"计划包含 {len(plan)} 个步骤")
    for step in plan:
        m.log(f"  步骤 {step.get('step_number', '?')}: {step.get('action', '?')} — {step.get('purpose', '')}")

    # --- 阶段 2：逐步执行 ---
    context_parts: list[str] = []

    for step in plan:
        action = step.get("action", "think")
        args = step.get("args", {})

        if action == "think":
            # 纯思考步骤，让 LLM 基于已有上下文推理。
            think_prompt = [
                {"role": "system", "content": "根据已收集的信息进行分析推理。"},
                {"role": "user", "content": f"任务：{task}\n\n已有信息：\n{''.join(context_parts)}\n\n请分析。"},
            ]
            thought = call_llm(think_prompt)
            m.llm_calls += 1
            context_parts.append(f"[分析] {thought}\n\n")
            m.log(f"思考完成，产出 {len(thought)} 字")

        elif action in TOOLS:
            # 执行工具，用 call_tool 处理 LLM 传参不规范的情况。
            result = call_tool(action, args)
            m.tool_calls += 1
            context_parts.append(f"[{action}] {result}\n\n")
            m.log(f"工具 {action} 返回 {len(result)} 字")

        else:
            m.log(f"未知动作 {action}，跳过")

    # --- 阶段 3：综合回答 ---
    synthesis_prompt = [
        {"role": "system", "content": "你是一个研究助手。根据收集到的所有信息，给出完整、有条理的回答。"},
        {"role": "user", "content": f"任务：{task}\n\n收集到的信息：\n{''.join(context_parts)}\n\n请给出最终回答。"},
    ]
    m.log("综合回答...")
    m.result = call_llm(synthesis_prompt)
    m.llm_calls += 1

    m.elapsed_seconds = time.time() - start
    return m


# ===========================================================================
# 算法二：Tree of Thoughts (ToT) — 简化版
# ===========================================================================
# 思路：生成多条初始思路 → LLM 打分 → 选最优展开 → 逐层深入。
# 优点：能探索多条路径，避免单一思路的盲区。
# 缺点：LLM 调用次数翻倍（生成 + 打分），成本高。

@dataclass
class ThoughtNode:
    """思维树的节点。"""
    thought: str
    score: float = 0.0
    parent: ThoughtNode | None = None
    children: list[ThoughtNode] = field(default_factory=list)
    depth: int = 0
    tool_results: list[str] = field(default_factory=list)

    def path(self) -> list[str]:
        """从根到当前节点的完整思考路径。"""
        nodes = []
        node: ThoughtNode | None = self
        while node is not None:
            nodes.append(node.thought)
            node = node.parent
        return list(reversed(nodes))


def tree_of_thoughts(task: str, breadth: int = 3, depth: int = 2) -> PlanningMetrics:
    """
    Tree of Thoughts 简化版。

    流程：
      1. 生成 breadth 条初始思路
      2. 用 LLM 给每条思路打分（0-10）
      3. 选得分最高的继续展开
      4. 重复 depth 层
      5. 最终用得分最高的完整路径生成回答
    """
    m = PlanningMetrics(algorithm="Tree-of-Thoughts", task=task)
    start = time.time()

    root = ThoughtNode(thought=f"任务：{task}", depth=0)

    # 逐层展开。
    current_level = [root]

    for d in range(depth):
        m.log(f"--- 第 {d + 1} 层（共 {depth} 层）---")
        next_level: list[ThoughtNode] = []

        for parent in current_level:
            # 生成子思路。
            path_so_far = "\n".join(f"第{i}步：{t}" for i, t in enumerate(parent.path(), 1))

            gen_prompt = [
                {"role": "system", "content": (
                    "你是一个创造性思考助手。给你一个任务和已有的思考路径，"
                    f"请提出 {breadth} 条不同的下一步思路。\n"
                    "每条思路包含：thought（具体想法）和 tool（如果需要用工具则指定 {\"name\": \"工具名\", \"args\": {参数}}，否则为 null）。\n"
                    f"{TOOLS_DESCRIPTION}\n"
                    f"返回 JSON 数组，恰好 {breadth} 个元素。"
                )},
                {"role": "user", "content": f"任务：{task}\n\n已有路径：\n{path_so_far}"},
            ]
            children_data = call_llm_json(gen_prompt, temperature=0.3)
            m.llm_calls += 1

            if not isinstance(children_data, list):
                children_data = children_data.get("thoughts", children_data.get("children", []))

            for cd in children_data[:breadth]:
                thought_text = cd.get("thought", str(cd))
                child = ThoughtNode(thought=thought_text, parent=parent, depth=d + 1)

                # 如果思路涉及工具调用，执行之。
                tool_spec = cd.get("tool")
                if tool_spec and isinstance(tool_spec, dict):
                    tool_name = tool_spec.get("name", "")
                    tool_args = tool_spec.get("args", {})
                    if tool_name in TOOLS:
                        result = call_tool(tool_name, tool_args)
                        child.tool_results.append(f"[{tool_name}] {result}")
                        m.tool_calls += 1

                parent.children.append(child)
                next_level.append(child)

            m.log(f"节点 '{parent.thought[:30]}...' 展开了 {len(parent.children)} 条子思路")

        # 给本层所有节点打分。
        if next_level:
            scoring_items = []
            for i, node in enumerate(next_level):
                tool_info = "; ".join(node.tool_results) if node.tool_results else "无工具调用"
                scoring_items.append(f"{i}: {node.thought} [工具结果: {tool_info}]")

            score_prompt = [
                {"role": "system", "content": (
                    "你是一个评估助手。给定一个研究任务和多条思路，请给每条思路打分（0-10）。\n"
                    "评分标准：相关性、深度、是否能推进任务。\n"
                    "返回 JSON 数组，每个元素是 {\"index\": 编号, \"score\": 分数, \"reason\": 简短理由}。"
                )},
                {"role": "user", "content": f"任务：{task}\n\n思路列表：\n" + "\n".join(scoring_items)},
            ]
            scores_data = call_llm_json(score_prompt)
            m.llm_calls += 1

            if not isinstance(scores_data, list):
                scores_data = scores_data.get("scores", scores_data.get("results", []))

            for item in scores_data:
                idx = item.get("index", 0)
                score = item.get("score", 5.0)
                if 0 <= idx < len(next_level):
                    next_level[idx].score = float(score)
                    m.log(f"  思路 {idx} 得分 {score}: {next_level[idx].thought[:40]}...")

            # 只保留得分最高的节点继续展开（剪枝）。
            next_level.sort(key=lambda n: n.score, reverse=True)
            current_level = next_level[:max(1, breadth // 2)]
        else:
            break

    # 选得分最高的叶子节点，用其完整路径生成回答。
    best = max(current_level, key=lambda n: n.score) if current_level else root
    best_path = best.path()
    all_tool_results: list[str] = []

    # 收集路径上所有工具结果。
    node: ThoughtNode | None = best
    while node is not None:
        all_tool_results.extend(node.tool_results)
        node = node.parent

    m.log(f"最佳路径（得分 {best.score}）：{' → '.join(t[:30] for t in best_path)}")

    final_prompt = [
        {"role": "system", "content": "你是一个研究助手。根据最佳思考路径和收集到的信息，给出完整回答。"},
        {"role": "user", "content": (
            f"任务：{task}\n\n"
            f"思考路径：\n" + "\n".join(f"  {i}. {t}" for i, t in enumerate(best_path, 1)) + "\n\n"
            f"工具调查结果：\n" + ("\n".join(all_tool_results) if all_tool_results else "无") + "\n\n"
            "请给出最终回答。"
        )},
    ]
    m.result = call_llm(final_prompt)
    m.llm_calls += 1

    m.elapsed_seconds = time.time() - start
    return m


# ===========================================================================
# 算法三：Reflexion
# ===========================================================================
# 思路：先执行任务 → 自我评估 → 如果不满意，反思失败原因存入记忆 → 带着经验重来。
# 优点：能从失败中学习，越做越好。
# 缺点：需要多次完整执行，token 消耗大。

def reflexion(task: str, max_attempts: int = 3) -> PlanningMetrics:
    """
    Reflexion：执行 → 评估 → 反思 → 重试。

    关键组件：
      - executor：执行任务的 agent
      - evaluator：评估结果质量的 LLM 调用
      - reflector：分析失败原因的 LLM 调用
      - memory：存放历次反思的经验
    """
    m = PlanningMetrics(algorithm="Reflexion", task=task)
    start = time.time()
    memory: list[str] = []  # 累积的经验教训。

    for attempt in range(1, max_attempts + 1):
        m.log(f"=== 第 {attempt} 次尝试 ===")

        # --- 执行 ---
        memory_text = "\n".join(f"- {exp}" for exp in memory) if memory else "无"
        exec_prompt = [
            {"role": "system", "content": (
                "你是一个研究助手。用户给你一个任务，你需要：\n"
                "1. 决定要搜索什么信息（返回 tool_calls）\n"
                "2. 综合信息给出回答\n\n"
                f"{TOOLS_DESCRIPTION}\n"
                "请先返回一个 JSON，格式：\n"
                '{{"tool_calls": [{{"tool": "工具名", "args": {{参数}}}}], "thinking": "你的推理过程"}}'
            )},
            {"role": "user", "content": (
                f"任务：{task}\n\n"
                f"从之前尝试中学到的经验：\n{memory_text}\n\n"
                "请先搜索所需信息。"
            )},
        ]
        exec_plan = call_llm_json(exec_prompt)
        m.llm_calls += 1

        # 执行工具调用。
        tool_results: list[str] = []
        tool_calls = exec_plan.get("tool_calls", [])
        for tc in tool_calls:
            tool_name = tc.get("tool", "")
            tool_args = tc.get("args", {})
            if tool_name in TOOLS:
                result = call_tool(tool_name, tool_args)
                tool_results.append(f"[{tool_name}] {result}")
                m.tool_calls += 1
                m.log(f"工具 {tool_name} → {len(result)} 字")

        # 综合回答。
        answer_prompt = [
            {"role": "system", "content": "根据收集到的信息，给出完整回答。回答要有条理，引用具体数据。"},
            {"role": "user", "content": (
                f"任务：{task}\n\n"
                f"收集到的信息：\n" + "\n".join(tool_results) + "\n\n"
                f"之前的经验教训：\n{memory_text}\n\n"
                "请给出回答。"
            )},
        ]
        answer = call_llm(answer_prompt)
        m.llm_calls += 1
        m.log(f"生成回答，{len(answer)} 字")

        # --- 评估 ---
        eval_prompt = [
            {"role": "system", "content": (
                "你是一个严格的评估者。评估以下回答的质量。\n"
                "返回 JSON：{\"score\": 1-10, \"pass\": true/false, \"issues\": [\"问题1\", \"问题2\"]}\n"
                "评分标准：准确性、完整性、条理性、是否有具体依据。\n"
                "7 分以上算通过。"
            )},
            {"role": "user", "content": f"任务：{task}\n\n回答：\n{answer}"},
        ]
        evaluation = call_llm_json(eval_prompt)
        m.llm_calls += 1

        score = evaluation.get("score", 0)
        passed = evaluation.get("pass", False)
        issues = evaluation.get("issues", [])

        m.log(f"评估：得分 {score}/10，通过={passed}")
        if issues:
            m.log(f"问题：{'; '.join(issues)}")

        if passed or attempt == max_attempts:
            m.result = answer
            if passed:
                m.log(f"✅ 第 {attempt} 次尝试通过")
            else:
                m.log(f"⚠️ 达到最大尝试次数，使用最后一次结果")
            break

        # --- 反思 ---
        reflect_prompt = [
            {"role": "system", "content": (
                "你是一个反思助手。分析任务执行的失败原因，提取可以指导下次尝试的经验教训。\n"
                "返回 JSON：{\"reflection\": \"具体的经验教训，下次应该怎么做\"}"
            )},
            {"role": "user", "content": (
                f"任务：{task}\n"
                f"本次回答：\n{answer}\n"
                f"评估问题：{issues}\n"
                f"使用的工具：{[tc.get('tool') for tc in tool_calls]}\n"
                "请反思哪里做得不够好，下次应该怎么改进。"
            )},
        ]
        reflection = call_llm_json(reflect_prompt)
        m.llm_calls += 1

        lesson = reflection.get("reflection", str(reflection))
        memory.append(f"[尝试{attempt}] {lesson}")
        m.log(f"反思：{lesson[:100]}...")

    m.elapsed_seconds = time.time() - start
    return m


# ===========================================================================
# 对比实验
# ===========================================================================

def run_comparison(task: str) -> None:
    """在同一个任务上运行三种算法，对比结果。"""
    print(f"\n{'=' * 70}")
    print(f"任务：{task}")
    print(f"{'=' * 70}")

    results: list[PlanningMetrics] = []

    # Plan-and-Solve。
    print(f"\n{'—' * 40}")
    print("1. Plan-and-Solve")
    print(f"{'—' * 40}")
    try:
        m1 = plan_and_solve(task)
        results.append(m1)
    except Exception as e:
        print(f"  ❌ 失败：{e}")

    # Tree of Thoughts。
    print(f"\n{'—' * 40}")
    print("2. Tree of Thoughts")
    print(f"{'—' * 40}")
    try:
        m2 = tree_of_thoughts(task, breadth=3, depth=2)
        results.append(m2)
    except Exception as e:
        print(f"  ❌ 失败：{e}")

    # Reflexion。
    print(f"\n{'—' * 40}")
    print("3. Reflexion")
    print(f"{'—' * 40}")
    try:
        m3 = reflexion(task, max_attempts=3)
        results.append(m3)
    except Exception as e:
        print(f"  ❌ 失败：{e}")

    # 汇总对比。
    print(f"\n{'=' * 70}")
    print("对比结果")
    print(f"{'=' * 70}")

    if not results:
        print("所有算法都失败了。")
        return

    print(f"\n{'算法':<20} {'LLM调用':<10} {'工具调用':<10} {'耗时':<10} {'步骤数':<10}")
    print("-" * 60)
    for m in results:
        print(f"{m.algorithm:<20} {m.llm_calls:<10} {m.tool_calls:<10} {m.elapsed_seconds:<10.1f} {len(m.steps_log):<10}")

    print("\n--- 各算法回答 ---")
    for m in results:
        print(f"\n[{m.algorithm}]")
        print(m.result[:500])
        print()


def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY。")

    print("第7周 Day 1：高级规划算法对比实验")
    print("=" * 70)

    # 测试任务：需要多步检索和综合分析的问题。
    tasks = [
        "对比 Python 的 GIL 和 Rust 的 Ownership 系统，分析它们分别怎么解决并发安全问题，各有什么优劣。",
        "解释 Python 的装饰器和元类各自适合什么场景，并给出选择建议。",
    ]

    for task in tasks:
        run_comparison(task)

    # 总结各算法适用场景。
    print("\n" + "=" * 70)
    print("总结：算法适用场景")
    print("=" * 70)
    print("""
Plan-and-Solve:
  ✅ 适合：有明确子任务分解的场景（写报告、做调研、流水线式工作）
  ❌ 不适合：需要动态调整计划的场景（执行过程中发现新信息）
  💡 核心：全局规划 + 顺序执行

Tree of Thoughts:
  ✅ 适合：需要探索多种可能性的推理（诊断问题、创意方案、开放式问题）
  ❌ 不适合：简单直接的任务（杀鸡用牛刀，token 消耗翻倍）
  💡 核心：广度探索 + 择优深入

Reflexion:
  ✅ 适合：需要迭代改进的任务（写代码、写文章、需要达到质量标准的场景）
  ❌ 不适合：一次性就能做好的简单任务
  💡 核心：试错 + 记忆 + 改进
""")


if __name__ == "__main__":
    main()
