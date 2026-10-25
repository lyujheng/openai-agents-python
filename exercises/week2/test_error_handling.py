"""
第2周 Day 5：错误处理测试

用故意会出错的工具测试 MiniAgent 的容错能力：
  1. 工具执行抛异常 → agent 能否自我纠正
  2. 工具名写错 → agent 能否换对的工具
  3. 参数传错 → agent 能否修正参数

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python test_error_handling.py
"""

import random
import sys

from mini_agent import MiniAgent


# ---------------------------------------------------------------------------
# 故意出错的工具
# ---------------------------------------------------------------------------

def flaky_search(query: str) -> str:
    """搜索信息，但这个服务不太稳定，有时候会出错。

    Args:
        query: 搜索关键词
    """
    # 模拟不稳定的外部服务：前两次调用必定失败，第三次成功。
    if not hasattr(flaky_search, "_call_count"):
        flaky_search._call_count = 0
    flaky_search._call_count += 1

    if flaky_search._call_count <= 2:
        raise ConnectionError(
            f"搜索服务暂时不可用（第{flaky_search._call_count}次尝试）"
        )

    # 第三次成功。
    flaky_search._call_count = 0
    return f"搜索 '{query}' 的结果：Python 是一种通用编程语言，由 Guido van Rossum 于 1991 年创建。"


def divide(a: float, b: float) -> str:
    """计算 a 除以 b 的结果。

    Args:
        a: 被除数
        b: 除数，不能为 0
    """
    if b == 0:
        raise ValueError("除数不能为 0，请提供一个非零的除数")
    return str(a / b)


def get_user_info(user_id: int) -> str:
    """根据用户 ID 查询用户信息。

    Args:
        user_id: 用户 ID，必须是正整数
    """
    # 模拟数据库。
    users = {
        1: "张三，工程师",
        2: "李四，设计师",
        3: "王五，产品经理",
    }
    if user_id not in users:
        raise KeyError(
            f"用户 ID {user_id} 不存在，有效 ID 范围：{list(users.keys())}"
        )
    return users[user_id]


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = MiniAgent(
        system_prompt=(
            "你是一个助手，可以搜索信息、做除法计算、查询用户信息。\n"
            "如果工具出错了，仔细阅读错误信息，尝试换一种方式解决。\n"
            "不要因为一次失败就放弃，至少尝试两种不同的方法。"
        ),
        tools=[flaky_search, divide, get_user_info],
    )

    # 测试 1：工具间歇性失败，agent 需要自我重试。
    print("=" * 60)
    print("测试 1：不稳定的工具 — agent 能否坚持重试")
    print("  (flaky_search 前2次必失败，第3次才成功)")
    print("=" * 60)
    answer = agent.run("帮我搜一下 Python 的基本介绍")
    print(f"\n最终回答: {answer}")

    # 测试 2：参数导致的错误，agent 需要修正。
    print("\n\n" + "=" * 60)
    print("测试 2：除以零 — agent 能否理解错误并调整")
    print("=" * 60)
    answer = agent.run("帮我算 100 除以 0")
    print(f"\n最终回答: {answer}")

    # 测试 3：查询不存在的数据，agent 需要根据错误信息调整。
    print("\n\n" + "=" * 60)
    print("测试 3：查不存在的用户 — agent 能否利用错误信息")
    print("=" * 60)
    answer = agent.run("帮我查一下用户 ID 为 99 的用户信息")
    print(f"\n最终回答: {answer}")
