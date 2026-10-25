"""
第2周 Day 3-4：接入真实工具

Day 1-2 的 MiniAgent 用的是假工具（硬编码数据）。
现在接上能操作文件系统的真实工具，让 agent 完成端到端任务。

关键学习点：
  1. 有副作用的工具（write_file 会真的写文件）需要什么安全考量
  2. 工具返回大量内容时的截断策略
  3. shell 命令执行的超时控制
  4. 错误信息要对 LLM 友好（让它能自我纠正）

测试任务：
  "分析当前目录下所有 Python 文件的代码行数，按从多到少排序，写入 report.txt"

运行方式：
    export DEEPSEEK_API_KEY="你的key"
    uv run python real_tools_agent.py              # 跑测试任务
    uv run python real_tools_agent.py --chat       # 交互模式
"""

import os
import subprocess
import sys

from mini_agent import MiniAgent


# ---------------------------------------------------------------------------
# 真实工具定义
# ---------------------------------------------------------------------------
# 每个工具就是一个普通函数。和 Day 1 的假工具唯一的区别：
# 这些函数有真实的副作用（读写文件、执行命令），所以需要考虑安全和错误处理。

# 工具操作的根目录。限制 agent 只能在这个目录下操作，防止它乱改系统文件。
WORKSPACE = os.path.dirname(os.path.abspath(__file__))

# 工具返回内容的最大长度。超过就截断，防止塞爆 context window。
MAX_OUTPUT_CHARS = 4000


def _safe_path(path: str) -> str:
    """
    把用户给的路径转成绝对路径，并检查是否在 WORKSPACE 内。

    为什么需要这个？
    - agent 可能被 LLM 指示去读 /etc/passwd 或 ~/.ssh/id_rsa
    - 限制在工作目录内是最基本的安全边界
    """
    # 如果是相对路径，基于 WORKSPACE 解析。
    if not os.path.isabs(path):
        full = os.path.normpath(os.path.join(WORKSPACE, path))
    else:
        full = os.path.normpath(path)

    if not full.startswith(WORKSPACE):
        raise PermissionError(
            f"不允许访问工作目录之外的路径: {path}\n工作目录: {WORKSPACE}"
        )
    return full


def _truncate(text: str) -> str:
    """截断过长的输出，并告知 LLM 内容被截断了。"""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n\n... [内容被截断，原始长度 {len(text)} 字符]"


def list_directory(path: str = ".") -> str:
    """列出目录下的文件和子目录。

    Args:
        path: 目录路径，相对于工作目录，默认为当前目录
    """
    full = _safe_path(path)
    if not os.path.isdir(full):
        return f"错误：'{path}' 不是一个目录"

    entries = []
    for name in sorted(os.listdir(full)):
        entry_path = os.path.join(full, name)
        if os.path.isdir(entry_path):
            entries.append(f"  [目录] {name}/")
        else:
            size = os.path.getsize(entry_path)
            entries.append(f"  [文件] {name}  ({size} bytes)")

    header = f"目录 '{path}' 下共 {len(entries)} 项：\n"
    return header + "\n".join(entries)


def read_file(path: str) -> str:
    """读取文件内容并返回。

    Args:
        path: 文件路径，相对于工作目录
    """
    full = _safe_path(path)
    if not os.path.isfile(full):
        return f"错误：文件 '{path}' 不存在"

    try:
        with open(full, encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        return f"错误：'{path}' 不是文本文件（二进制文件无法读取）"

    return _truncate(content)


def write_file(path: str, content: str) -> str:
    """将内容写入文件。如果文件已存在会被覆盖。

    Args:
        path: 文件路径，相对于工作目录
        content: 要写入的文本内容
    """
    full = _safe_path(path)

    # 自动创建父目录。
    parent = os.path.dirname(full)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    with open(full, "w", encoding="utf-8") as f:
        f.write(content)

    size = os.path.getsize(full)
    return f"成功写入 '{path}'（{size} bytes）"


def run_command(cmd: str, timeout: int = 30) -> str:
    """执行一条 shell 命令并返回输出。

    Args:
        cmd: 要执行的 shell 命令
        timeout: 超时秒数，默认 30 秒
    """
    # 安全检查：拦截明显的危险命令。
    dangerous = ["rm -rf /", "rm -rf ~", "mkfs", "dd if=", ":(){", "fork bomb"]
    cmd_lower = cmd.lower().strip()
    for pattern in dangerous:
        if pattern in cmd_lower:
            return f"安全拦截：拒绝执行危险命令 '{cmd}'"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=WORKSPACE,  # 在工作目录下执行。
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[退出码: {result.returncode}]"

        return _truncate(output) if output.strip() else "(命令执行成功，无输出)"

    except subprocess.TimeoutExpired:
        return f"错误：命令超时（{timeout}秒限制）"
    except Exception as e:
        return f"命令执行出错：{type(e).__name__}: {e}"


def count_lines(path: str) -> str:
    """统计一个文本文件的行数。

    Args:
        path: 文件路径，相对于工作目录
    """
    full = _safe_path(path)
    if not os.path.isfile(full):
        return f"错误：文件 '{path}' 不存在"

    try:
        with open(full, encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        return f"{path}: {lines} 行"
    except UnicodeDecodeError:
        return f"错误：'{path}' 不是文本文件"


# ---------------------------------------------------------------------------
# 组装 agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
你是一个文件操作助手，可以浏览目录、读写文件、执行命令、统计行数。

工作原则：
1. 先用 list_directory 了解目录结构，再做具体操作
2. 操作前确认路径正确，避免误操作
3. 遇到错误时分析原因，尝试换一种方式解决
4. 任务完成后给出清晰的总结
"""

ALL_TOOLS = [list_directory, read_file, write_file, run_command, count_lines]


if __name__ == "__main__":
    agent = MiniAgent(
        system_prompt=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
    )

    # --chat 交互模式。
    if "--chat" in sys.argv:
        print("=" * 60)
        print("文件操作 Agent 交互模式（输入 quit 退出）")
        print(f"工作目录：{WORKSPACE}")
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

    # 默认：跑端到端测试任务。
    print("=" * 60)
    print("端到端测试：分析 Python 文件行数")
    print(f"工作目录：{WORKSPACE}")
    print(f"已注册工具：{list(agent.tool_map.keys())}")
    print("=" * 60)

    answer = agent.run(
        "分析当前目录及上级目录(week1)下所有 Python 文件的代码行数，"
        "按从多到少排序，结果写入 report.txt"
    )
    print(f"\n最终回答:\n{answer}")

    # 验证：检查 report.txt 是否生成。
    report_path = os.path.join(WORKSPACE, "report.txt")
    if os.path.exists(report_path):
        print("\n" + "=" * 60)
        print("验证：report.txt 内容")
        print("=" * 60)
        with open(report_path) as f:
            print(f.read())
    else:
        print("\n⚠️ report.txt 未生成，agent 可能没有完成任务")
