"""
Week5 Day5: 把 Week2 mini_agent 的工具迁移到 MCP Server

对比 Week2 的 mini_agent.py：
- 原来：工具是普通 Python 函数，schema 靠 func_to_schema() 手动生成
- 现在：工具用 @mcp.tool() 装饰器，schema 自动暴露给任何 MCP Client

三个工具完全保留原始逻辑，只改了注册方式：
  get_weather    → @mcp.tool()
  calculate      → @mcp.tool()
  get_exchange_rate → @mcp.tool()

启动方式：
  uv run python exercises/week5/mini_agent_mcp_server.py
"""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("mini-agent-tools")


# ---------------------------------------------------------------------------
# 工具 1：天气查询（模拟）
# ---------------------------------------------------------------------------
# 和 Week2 mini_agent.py 里的 get_weather 完全一样的逻辑，
# 区别只在注册方式：从 agent.register_tool(get_weather) 变成 @mcp.tool()
@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。

    Args:
        city: 城市名称，如"东京"、"北京"
    """
    # 模拟数据，实际场景会调天气 API。
    weather_data = {
        "东京": "晴天，25°C，湿度 60%",
        "北京": "多云，18°C，湿度 45%",
        "纽约": "小雨，15°C，湿度 80%",
    }
    return weather_data.get(city, f"没有 {city} 的天气数据")


# ---------------------------------------------------------------------------
# 工具 2：数学计算
# ---------------------------------------------------------------------------
@mcp.tool()
def calculate(expression: str) -> str:
    """计算数学表达式。

    Args:
        expression: 数学表达式，如 "1000 * 7.18"
    """
    try:
        # 安全限制：只允许数字和基本运算符。
        import re
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
            return f"不安全的表达式: {expression}"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


# ---------------------------------------------------------------------------
# 工具 3：汇率查询（模拟）
# ---------------------------------------------------------------------------
@mcp.tool()
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """查询货币汇率。

    Args:
        from_currency: 源货币代码，如 "USD"
        to_currency: 目标货币代码，如 "CNY"
    """
    rates = {
        ("USD", "CNY"): 7.18,
        ("USD", "JPY"): 149.5,
        ("CNY", "JPY"): 20.8,
        ("EUR", "USD"): 1.09,
    }
    rate = rates.get((from_currency, to_currency))
    if rate:
        return f"1 {from_currency} = {rate} {to_currency}"
    # 反向查询。
    reverse = rates.get((to_currency, from_currency))
    if reverse:
        return f"1 {from_currency} = {round(1/reverse, 4)} {to_currency}"
    return f"没有 {from_currency} → {to_currency} 的汇率数据"


if __name__ == "__main__":
    print("MCP Server (mini-agent-tools) 启动中...")
    print("工具: get_weather, calculate, get_exchange_rate")
    print("Transport: stdio")
    print("提示: Client 会自动启动这个进程，通常不需要手动运行")
    print()
    # 用 stdio 模式，这样 Client 可以自动拉起子进程。
    # 想测试 SSE 模式的话，改成：mcp.run(transport="sse", host="127.0.0.1", port=8081)
    mcp.run(transport="stdio")
