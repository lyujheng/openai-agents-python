"""
Day 3 练习：计算工具 MCP Server
提供数学计算和数据处理工具，和文件 Server 配合使用。

演示多 Server 组合：一个 Agent 同时接入文件 Server 和计算 Server，
LLM 自主决定用哪个 Server 的工具。
"""

import json
import math
import statistics

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("calc-tools")


@mcp.tool()
def calculate(expression: str) -> str:
    """计算数学表达式。支持 +, -, *, /, **, sqrt, sin, cos, pi 等。
    示例：calculate("2 + 3 * 4") → "14"
    """
    # 安全的数学函数白名单。
    safe_dict = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


@mcp.tool()
def analyze_numbers(numbers_json: str) -> str:
    """对一组数字进行统计分析。输入是 JSON 格式的数字数组，如 "[1, 2, 3, 4, 5]"。
    返回：平均值、中位数、标准差、最大值、最小值。
    """
    try:
        numbers = json.loads(numbers_json)
        if not isinstance(numbers, list) or not all(isinstance(n, (int, float)) for n in numbers):
            return "错误：输入必须是数字数组，如 [1, 2, 3]"
        if len(numbers) == 0:
            return "错误：数组不能为空"

        result = {
            "count": len(numbers),
            "mean": round(statistics.mean(numbers), 2),
            "median": round(statistics.median(numbers), 2),
            "stdev": round(statistics.stdev(numbers), 2) if len(numbers) > 1 else 0,
            "max": max(numbers),
            "min": min(numbers),
            "sum": sum(numbers),
        }
        lines = [f"  {k}: {v}" for k, v in result.items()]
        return "统计结果：\n" + "\n".join(lines)
    except json.JSONDecodeError:
        return "错误：输入不是合法的 JSON 格式"


@mcp.tool()
def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
    """单位转换。支持温度（C/F/K）和长度（m/km/mile/ft）。"""
    # 温度转换。
    temp_conversions = {
        ("C", "F"): lambda v: v * 9 / 5 + 32,
        ("F", "C"): lambda v: (v - 32) * 5 / 9,
        ("C", "K"): lambda v: v + 273.15,
        ("K", "C"): lambda v: v - 273.15,
        ("F", "K"): lambda v: (v - 32) * 5 / 9 + 273.15,
        ("K", "F"): lambda v: (v - 273.15) * 9 / 5 + 32,
    }
    # 长度转换（统一转成米再转目标单位）。
    length_to_meter = {"m": 1, "km": 1000, "mile": 1609.344, "ft": 0.3048}

    from_u, to_u = from_unit.upper(), to_unit.upper()

    if (from_u, to_u) in temp_conversions:
        result = temp_conversions[(from_u, to_u)](value)
        return f"{value} {from_unit} = {round(result, 2)} {to_unit}"

    from_l, to_l = from_unit.lower(), to_unit.lower()
    if from_l in length_to_meter and to_l in length_to_meter:
        meters = value * length_to_meter[from_l]
        result = meters / length_to_meter[to_l]
        return f"{value} {from_unit} = {round(result, 4)} {to_unit}"

    return f"不支持的单位转换：{from_unit} → {to_unit}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
