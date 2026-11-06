"""
Week5 Day4: SSE transport 版本的 file-tools MCP Server

与 Day2 的 stdio 版本对比：
- stdio: mcp.run(transport="stdio")  → 通过 stdin/stdout 通信，Client 拉起子进程
- SSE:   mcp.run(transport="sse", host=..., port=...)  → 启动 HTTP 服务，独立运行

启动方式：
  uv run python exercises/week5/sse_server.py
  # Server 会在 http://127.0.0.1:8080/sse 监听

停止方式：Ctrl+C

注意：mcp 2.0.0 使用 MCPServer（不是旧版的 FastMCP）
"""

import os

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("file-tools-sse")

# 安全边界：限制可访问的目录。
ALLOWED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_files")


def _safe_path(filename: str) -> str:
    """防止路径穿越，限制在 ALLOWED_DIR 内。"""
    resolved = os.path.realpath(os.path.join(ALLOWED_DIR, filename))
    if not resolved.startswith(os.path.realpath(ALLOWED_DIR)):
        raise ValueError(f"路径不在允许范围内: {filename}")
    return resolved


@mcp.tool()
def read_file(filename: str) -> str:
    """读取 sample_files 目录下的文件内容。"""
    path = _safe_path(filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if len(content) > 5000:
        content = content[:5000] + "\n... (truncated)"
    return content


@mcp.tool()
def list_dir(subdir: str = ".") -> str:
    """列出 sample_files 目录下的文件和子目录。"""
    path = _safe_path(subdir)
    if not os.path.isdir(path):
        return f"不是目录: {subdir}"
    entries = os.listdir(path)
    return "\n".join(entries) if entries else "(空目录)"


@mcp.tool()
def get_file_info(filename: str) -> str:
    """获取文件的基本信息：大小、行数。"""
    path = _safe_path(filename)
    if not os.path.isfile(path):
        return f"文件不存在: {filename}"
    size = os.path.getsize(path)
    with open(path, "r", encoding="utf-8") as f:
        lines = sum(1 for _ in f)
    return f"文件: {filename}, 大小: {size} bytes, 行数: {lines}"


SSE_HOST = "127.0.0.1"
SSE_PORT = 8080

if __name__ == "__main__":
    # SSE transport: 启动 HTTP 服务器。
    # 和 stdio 不同，这里可以 print，因为 stdout 不用于协议通信。
    print(f"启动 file-tools SSE Server: http://{SSE_HOST}:{SSE_PORT}/sse")
    print(f"允许访问的目录: {ALLOWED_DIR}")
    print("按 Ctrl+C 停止\n")
    mcp.run(transport="sse", host=SSE_HOST, port=SSE_PORT)
