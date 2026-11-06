"""
Day 2 练习 1：自己写一个 MCP Server
提供文件操作工具，通过 stdio transport 暴露给 Client。
"""

import os
from datetime import datetime

from mcp.server.mcpserver import MCPServer

# 创建 Server 实例，名字会在 Client 端显示
mcp = MCPServer("file-tools")

# 限制 Server 只能访问的目录（安全边界）
ALLOWED_DIR = os.path.dirname(os.path.abspath(__file__))


def _safe_path(path: str) -> str:
    """确保路径在允许的目录内，防止路径穿越攻击。"""
    real_path = os.path.realpath(path)
    if not real_path.startswith(os.path.realpath(ALLOWED_DIR)):
        raise ValueError(f"访问被拒绝：路径 {path} 不在允许的目录内")
    return real_path


@mcp.tool()
def read_file(path: str) -> str:
    """读取指定文件的内容。path 应该是绝对路径或相对于 Server 工作目录的路径。"""
    safe = _safe_path(path)
    if not os.path.isfile(safe):
        return f"错误：{path} 不是文件或不存在"
    with open(safe, encoding="utf-8") as f:
        content = f.read()
    # 截断过长内容，避免撑爆 LLM 的 context。
    if len(content) > 5000:
        content = content[:5000] + "\n...(内容被截断，共 {} 字符)".format(len(content))
    return content


@mcp.tool()
def list_dir(path: str) -> str:
    """列出指定目录下的文件和子目录。返回格式：每行一个条目，目录名后带 / 标记。"""
    safe = _safe_path(path)
    if not os.path.isdir(safe):
        return f"错误：{path} 不是目录或不存在"
    entries = sorted(os.listdir(safe))
    lines = []
    for entry in entries:
        full = os.path.join(safe, entry)
        if os.path.isdir(full):
            lines.append(f"  {entry}/")
        else:
            lines.append(f"  {entry}")
    return f"目录 {path} 下共 {len(entries)} 个条目：\n" + "\n".join(lines)


@mcp.tool()
def get_file_info(path: str) -> str:
    """获取文件或目录的详细信息：大小、修改时间、类型。"""
    safe = _safe_path(path)
    if not os.path.exists(safe):
        return f"错误：{path} 不存在"
    stat = os.stat(safe)
    file_type = "目录" if os.path.isdir(safe) else "文件"
    size = stat.st_size
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"类型: {file_type}\n大小: {size} 字节\n修改时间: {modified}\n路径: {safe}"


if __name__ == "__main__":
    # stdio transport 下不能用 print()，stdout 是给 JSON-RPC 消息用的。
    mcp.run(transport="stdio")
