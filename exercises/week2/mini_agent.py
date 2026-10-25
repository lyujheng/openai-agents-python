"""
第2周：最小 Agent 核心循环

把 ex2 里手动走的流程自动化：
  用户输入 → [LLM 决策 → 执行工具 → 结果喂回] × N → 最终回答

核心设计决策：
  1. 工具注册：传入普通 Python 函数，自动生成 JSON Schema
  2. 循环控制：LLM 不再返回 tool_calls 时停止，max_turns 兜底
  3. 工具路由：根据 function.name 自动分发到对应函数

Day 2 改进：
  - func_to_schema 从 docstring 解析参数描述
  - 支持可选参数（有默认值的参数不放进 required）

Day 5 改进：
  - JSON 解析失败时的容错处理（不崩溃，给 LLM 纠错机会）
  - API 调用失败时的指数退避重试
  - 更友好的错误信息，引导 LLM 自我纠正

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python mini_agent.py            # 跑自动测试
    uv run python mini_agent.py --chat     # 交互对话模式
"""

import inspect
import json
import os
import re
import sys
import time
from typing import Any, Callable, get_type_hints

from openai import OpenAI, APIConnectionError, RateLimitError, APITimeoutError


# ---------------------------------------------------------------------------
# 第一块：API 调用重试
# ---------------------------------------------------------------------------
# 为什么需要重试？
# API 调用会因为网络抖动、服务端限流、超时等原因失败。
# 这些都是临时性错误，等一会儿再试通常就好了。
# 指数退避（exponential backoff）：每次重试等更久，避免雪崩式重试加重服务端负担。

def call_llm_with_retry(
    client: OpenAI,
    max_retries: int = 3,
    **kwargs,
) -> Any:
    """
    带指数退避重试的 LLM API 调用。

    只对临时性错误重试（网络、限流、超时），其他错误直接抛出。
    """
    # 值得重试的错误类型。
    retryable = (APIConnectionError, RateLimitError, APITimeoutError)

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except retryable as e:
            if attempt == max_retries:
                raise  # 重试用完了，抛出原始错误。
            # 指数退避：1s → 2s → 4s。
            wait = 2 ** attempt
            print(f"  [重试] {type(e).__name__}，{wait}秒后第{attempt + 1}次重试...")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# 第二块：从 Python 函数自动生成 JSON Schema
# ---------------------------------------------------------------------------
# 第一块：从 Python 函数自动生成 JSON Schema
# ---------------------------------------------------------------------------
# 为什么需要这个？
# ex1-3 里你手写了 tools 数组，每加一个工具就要写一大段 JSON。
# 生产框架（Agents SDK 的 function_schema.py）也做同样的事，只是更完善。
# 我们这里写一个最简版本，理解原理就够了。

# Python 类型 → JSON Schema 类型的映射。
TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def parse_param_descriptions(docstring: str) -> dict[str, str]:
    """
    从 Google 风格的 docstring 中解析参数描述。

    支持的格式：
        Args:
            city: 城市名，比如"东京"
            unit: 温度单位，默认摄氏度

    返回 {"city": "城市名，比如\"东京\"", "unit": "温度单位，默认摄氏度"}
    """
    descriptions: dict[str, str] = {}
    if not docstring:
        return descriptions

    # 找到 Args: 段落，提取每个 "参数名: 描述" 行。
    in_args = False
    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            # 遇到空行或下一个段落标题（如 Returns:）就结束。
            if not stripped or (stripped.endswith(":") and not stripped.startswith(" ")):
                break
            # 匹配 "param_name: description" 或 "param_name (type): description"。
            match = re.match(r"(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+)", stripped)
            if match:
                descriptions[match.group(1)] = match.group(2).strip()

    return descriptions


def func_to_schema(func: Callable) -> dict:
    """
    把一个 Python 函数转成 OpenAI function calling 的 JSON Schema。

    Day 2 改进：
    - 从 docstring 的 Args 段落解析参数描述（Google 风格）
    - 有默认值的参数不放进 required
    - 函数描述只取 docstring 第一行（不混入参数说明）
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    param_docs = parse_param_descriptions(func.__doc__ or "")

    # 函数描述只取 docstring 第一行，不把 Args 段落混进去。
    raw_doc = (func.__doc__ or "").strip()
    func_description = raw_doc.split("\n")[0].strip() if raw_doc else ""

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        # 从 type hints 拿类型，映射到 JSON Schema 类型。
        py_type = hints.get(name, str)
        json_type = TYPE_MAP.get(py_type, "string")

        # 优先用 docstring 里的描述，没有就用参数名兜底。
        description = param_docs.get(name, name)
        properties[name] = {"type": json_type, "description": description}

        # 有默认值的参数是可选的，不加入 required。
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# 第二块：MiniAgent 核心类
# ---------------------------------------------------------------------------

class MiniAgent:
    """
    最小 agent 实现。核心就是一个 while 循环：
    调 LLM → 有 tool_calls 就执行 → 没有就结束。
    """

    def __init__(
        self,
        system_prompt: str,
        tools: list[Callable],
        model: str = "mco-4",
        max_turns: int = 10,
    ):
        self.client = OpenAI(
            base_url="https://zyapi.xmsxb.com/v1",
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        )
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = system_prompt

        # 工具注册：name → callable 的映射，用于路由。
        self.tool_map: dict[str, Callable] = {f.__name__: f for f in tools}

        # 自动生成每个工具的 JSON Schema。
        self.tool_schemas: list[dict] = [func_to_schema(f) for f in tools]

    def run(self, user_input: str) -> str:
        """
        agent 主循环。这是整个文件最核心的部分。

        对比 ex2：ex2 手动调了两次 LLM，这里用 while 循环自动化，
        LLM 可以连续调多轮工具，直到它自己决定停下来。

        Day 5 改进：
        - API 调用用 call_llm_with_retry 包装，临时错误自动重试
        - tool_call 参数 JSON 解析失败时不崩溃，把错误喂回 LLM 让它重试
        """
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        for turn in range(self.max_turns):
            print(f"\n--- Turn {turn + 1}/{self.max_turns} ---")

            # 1. 调 LLM（带重试）。
            response = call_llm_with_retry(
                self.client,
                model=self.model,
                messages=messages,
                tools=self.tool_schemas if self.tool_schemas else None,
                tool_choice="auto" if self.tool_schemas else None,
            )
            message = response.choices[0].message

            # 2. 判断：LLM 要调工具，还是直接给出最终回答？
            if not message.tool_calls:
                # 没有 tool_calls → agent 认为任务完成，返回最终回答。
                print(f"  LLM 最终回答（无工具调用）")
                return message.content or "(空回复)"

            # 3. 有 tool_calls → 执行每个工具，把结果喂回去。
            # 先把 LLM 的消息加入历史（包含 tool_calls 信息）。
            messages.append(message)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name

                # Day 5：JSON 解析容错。
                # LLM 返回的 arguments 可能不是合法 JSON（尤其是弱模型）。
                # 不能让整个 agent 崩掉，把解析错误当作工具执行错误处理。
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    print(f"  调用工具: {func_name}(JSON 解析失败)")
                    error_msg = (
                        f"参数 JSON 解析失败：{e}\n"
                        f"你提供的原始内容：{tool_call.function.arguments!r}\n"
                        f"请检查 JSON 格式是否正确（引号、逗号、括号），然后重新调用。"
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg,
                    })
                    continue

                print(f"  调用工具: {func_name}({func_args})")

                # 路由到对应的 Python 函数并执行。
                result = self._execute_tool(func_name, func_args)
                print(f"  工具结果: {result[:200]}")  # 截断显示，防止刷屏。

                # 以 role="tool" 喂回 LLM，和 ex2 一样。
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

            # 循环继续 → 回到 step 1，让 LLM 看到工具结果后做下一步决策。

        return "达到最大轮次限制，agent 被强制停止。"

    def _execute_tool(self, name: str, args: dict) -> str:
        """
        执行工具，返回结果字符串。

        Day 5 改进：
        - 未知工具时列出可用工具，帮 LLM 修正
        - 参数错误时提示正确的参数列表
        - 所有异常都转成对 LLM 友好的错误信息
        """
        func = self.tool_map.get(name)
        if not func:
            return (
                f"错误：未知工具 '{name}'。\n"
                f"可用工具：{list(self.tool_map.keys())}\n"
                f"请从可用工具中选择一个重新调用。"
            )
        try:
            return str(func(**args))
        except TypeError as e:
            # 参数不匹配（多了、少了、名字错了）。
            sig = inspect.signature(func)
            return (
                f"参数错误：{e}\n"
                f"工具 '{name}' 的正确参数：{sig}\n"
                f"请修正参数后重新调用。"
            )
        except Exception as e:
            return (
                f"工具执行出错：{type(e).__name__}: {e}\n"
                f"请分析错误原因，尝试换一种方式完成任务。"
            )


# ---------------------------------------------------------------------------
# 第三块：定义几个测试工具，验证循环
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    """获取指定城市的天气信息。

    Args:
        city: 城市名，比如"东京""北京""纽约"
    """
    data = {
        "东京": "晴，28°C，湿度 65%",
        "北京": "多云，32°C，湿度 45%",
        "纽约": "小雨，22°C，湿度 80%",
    }
    return data.get(city, f"没有 {city} 的天气数据")


def get_exchange_rate(currency_from: str, currency_to: str) -> str:
    """查询两种货币之间的汇率。

    Args:
        currency_from: 源货币代码，比如 USD、CNY、JPY
        currency_to: 目标货币代码，比如 USD、CNY、JPY
    """
    rates = {
        ("USD", "CNY"): 7.24,
        ("USD", "JPY"): 149.5,
        ("CNY", "JPY"): 20.65,
    }
    pair = (currency_from.upper(), currency_to.upper())
    rate = rates.get(pair)
    if rate:
        return f"1 {currency_from} = {rate} {currency_to}"
    # 试试反向。
    reverse = (pair[1], pair[0])
    rate = rates.get(reverse)
    if rate:
        return f"1 {currency_from} = {1/rate:.4f} {currency_to}"
    return f"没有 {currency_from} → {currency_to} 的汇率数据"


def calculate(expression: str) -> str:
    """计算一个数学表达式。

    Args:
        expression: 数学表达式，比如 '2 + 3 * 4'、'100 / 7'
    """
    try:
        # 安全限制：只允许数字和基本运算符。
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"不支持的表达式: {expression}"
        return str(eval(expression))  # noqa: S307 — 教学用途，已做字符过滤。
    except Exception as e:
        return f"计算错误: {e}"


# ---------------------------------------------------------------------------
# 第四块：运行测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = MiniAgent(
        system_prompt=(
            "你是一个有用的助手，可以查天气、查汇率、做计算。\n"
            "根据用户的问题，决定需要调用哪些工具来回答。\n"
            "如果一个问题需要多步，就分步调用工具。"
        ),
        tools=[get_weather, get_exchange_rate, calculate],
    )

    # --chat 模式：交互对话，方便调试和体验。
    if "--chat" in sys.argv:
        print("=" * 60)
        print("MiniAgent 交互模式（输入 quit 退出）")
        print(f"已注册工具：{list(agent.tool_map.keys())}")
        print("=" * 60)
        while True:
            try:
                user_input = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break
            answer = agent.run(user_input)
            print(f"\nAgent: {answer}")
        print("\n再见！")
        sys.exit(0)

    # 默认：跑自动测试。
    # 测试 1：单工具调用（和 ex2 类似，但现在是自动循环）。
    print("=" * 60)
    print("测试 1：单工具 — 东京天气")
    print("=" * 60)
    answer = agent.run("东京今天天气怎么样？")
    print(f"\n最终回答: {answer}")

    # 测试 2：多工具 + 多步推理。
    # LLM 需要先查汇率，再做计算 — 这需要至少两轮工具调用。
    print("\n\n" + "=" * 60)
    print("测试 2：多步推理 — 汇率换算")
    print("=" * 60)
    answer = agent.run("我有 1000 美元，换成人民币是多少？再换成日元呢？")
    print(f"\n最终回答: {answer}")

    # 测试 3：LLM 自己判断不需要工具。
    print("\n\n" + "=" * 60)
    print("测试 3：无需工具 — 直接回答")
    print("=" * 60)
    answer = agent.run("你好，你是谁？")
    print(f"\n最终回答: {answer}")

    # 测试 4（Day 2 新增）：验证 schema 生成质量。
    print("\n\n" + "=" * 60)
    print("测试 4：检查生成的 JSON Schema")
    print("=" * 60)
    for schema in agent.tool_schemas:
        func_info = schema["function"]
        print(f"\n工具: {func_info['name']}")
        print(f"  描述: {func_info['description']}")
        params = func_info["parameters"]
        print(f"  必填参数: {params['required']}")
        for pname, pinfo in params["properties"].items():
            print(f"  - {pname}: type={pinfo['type']}, desc=\"{pinfo['description']}\"")
