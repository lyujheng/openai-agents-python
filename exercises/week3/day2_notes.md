# Day 2 笔记：工具系统

## 核心收获

Day 2 的核心问题：SDK 的工具系统比我的 mini_agent 多了什么？答案是**类型体系 + Pydantic 驱动的 schema 生成 + 运行时上下文**。

---

## Tool 类型体系

SDK 的 `Tool` 不是一个抽象基类，而是 13 种工具类型的 Union：

```python
Tool = FunctionTool | FileSearchTool | WebSearchTool | ComputerTool | ...
```

这些工具类型的接口差异很大（本地函数 vs 服务端托管 vs 屏幕操控），硬塞进一个基类不自然，所以用 Union 组合。

实际搭 agent 时常用的：

| 工具类型 | 什么时候用 |
|---------|----------|
| **FunctionTool** | 自定义 Python 函数（`@tool` 创建），90% 场景用这个 |
| **WebSearchTool** | 让 agent 搜索网页 |
| **FileSearchTool** | 在文件/向量库中搜索（OpenAI 托管） |
| **CodeInterpreterTool** | 让 agent 执行 Python 代码（OpenAI 沙箱） |
| **HostedMCPTool** | 接入 MCP 服务器的工具（第5周） |

---

## FunctionTool 的字段

对比我的 mini_agent，SDK 的 FunctionTool 多了很多控制能力：

| 字段 | 作用 | 我的 mini_agent 有吗 |
|------|------|---------------------|
| `name` | 工具名 | ✅ `func.__name__` |
| `description` | 工具描述 | ✅ docstring 第一行 |
| `params_json_schema` | 参数 JSON Schema | ✅ `func_to_schema()` 生成 |
| `on_invoke_tool` | 执行工具的函数 | ✅ `tool_map[name](**args)` |
| `strict_json_schema` | 强制 LLM 严格按 schema 填参 | ❌ |
| `is_enabled` | 动态启用/禁用工具 | ❌ |
| `needs_approval` | 人工审批（Human-in-the-loop） | ❌ 第4周会做 |
| `timeout_seconds` | 工具执行超时 | ❌ |
| `tool_input_guardrails` | 工具级输入检查 | ❌ |
| `tool_output_guardrails` | 工具级输出检查 | ❌ |

---

## @tool 装饰器的完整链路

```
@tool
def get_weather(city: Annotated[str, "..."]) -> Weather:
    """Get the current weather..."""

装饰器内部三步：
  ① function_schema() → 从签名 + docstring + 类型注解生成 JSON Schema
  ② 构建 _on_invoke_tool_impl() → 包装执行逻辑（参数验证 + sync/async 适配）
  ③ 返回 FunctionTool 对象
```

`@tool` 就是 `@function_tool` 的别名（`decorators.py` 只有一行 `tool = function_tool`）。

---

## function_schema.py：schema 生成的核心

这个文件和我的 `func_to_schema()` 直接对标，核心流程 5 步：

```
1. 从 docstring 提取函数描述和参数描述
2. 从函数签名和类型注解提取参数信息
3. 动态创建一个 Pydantic Model（关键差异！）
4. 从 Pydantic Model 生成 JSON Schema
5. 打包返回 FuncSchema
```

### 和我的 func_to_schema() 对比

| | 我的 func_to_schema() | SDK 的 function_schema() |
|--|----------------------|--------------------------|
| 参数描述来源 | 只从 docstring 解析 | docstring + Annotated 两个来源 |
| 类型处理 | TYPE_MAP 手动映射 6 种基础类型 | 交给 Pydantic，自动支持任意类型 |
| schema 生成 | 手动拼 JSON | create_model() → model_json_schema() |
| 参数验证 | 无 | 生成的 Pydantic Model 同时用来验证 LLM 传入的参数 |
| *args/**kwargs | 不支持 | 支持，转成 list/dict |

### 关键设计思路

SDK 用 Pydantic Model 作为中间层，一次构建同时服务 schema 生成和参数验证。我的版本分别处理这两件事（func_to_schema 生成 schema，_execute_tool 里直接 func(**args) 没有验证），SDK 把它们统一了。

---

## Pydantic 是什么

Pydantic = **带运行时类型验证的 dataclass + 自带 JSON Schema 生成 + JSON 序列化/反序列化**。

```python
# dataclass：有类型注解但不检查
User(name=123, age="abc")  # 不报错

# Pydantic：运行时检查
User(name=123, age="abc")  # ValidationError!
```

三个核心能力：
1. **运行时类型验证**：传错类型直接报错
2. **自动生成 JSON Schema**：`Model.model_json_schema()` 一行搞定
3. **JSON 序列化/反序列化**：`model_dump_json()` / `model_validate_json()`

在 SDK 里用在三个地方：
- 工具参数 schema 生成（function_schema.py）
- 工具参数验证（tool.py 里的 _on_invoke_tool_impl）
- 输出约束（Agent(output_type=MyModel)）

### 以后要记住的

定义工具参数和输出结构，用 Pydantic BaseModel + Field + Annotated，不用手写 JSON Schema。

---

## ToolContext：工具的运行时上下文

ToolContext 继承自 RunContextWrapper，是工具执行时拿到的"周围环境信息"。

```python
@tool
def read_file(ctx: ToolContext, path: str) -> str:
    ctx.tool_name          # "read_file" — 我叫什么
    ctx.tool_call_id       # "call_abc123" — 这次调用的唯一 ID
    ctx.tool_arguments     # '{"path": "/tmp/a.txt"}' — LLM 传的原始 JSON
    ctx.context            # 用户自定义的共享数据
    ctx.usage              # token 消耗统计
    ctx.agent              # 哪个 agent 在调我
```

如果函数第一个参数是 ToolContext，SDK 自动注入，不放进 JSON Schema（function_schema.py 第 395 行的特殊处理）。

### 核心价值

让工具从"只知道参数"变成"知道自己在什么环境下被谁调用"。

对比我的 mini_agent：
- 我的做法：安全检查（_safe_path）硬编码在工具里，不同用户不同权限只能靠全局变量
- SDK 的做法：通过 ctx.context 拿到运行时用户信息，同一个工具函数自动适配不同用户，并发安全

---

## Day 2 检查点

- [x] 知道 SDK 支持 13 种工具类型，实际常用 FunctionTool + WebSearchTool + FileSearchTool
- [x] 理解 @tool 装饰器的完整链路：function_schema → 构建调用函数 → 组装 FunctionTool
- [x] 理解 function_schema.py 的核心设计：用 Pydantic Model 作为中间层统一 schema 生成和参数验证
- [x] 知道 Pydantic 是什么，以后定义工具参数和输出结构用 Pydantic
- [x] 理解 ToolContext 让工具感知运行环境，不再是孤立函数

---

## 遗留疑问

1. FunctionTool 的 `needs_approval` 具体怎么中断和恢复执行？（第4周实践）
2. MCP 工具（HostedMCPTool）和 FunctionTool 在 Runner 循环里的执行路径有什么区别？（第5周）
3. `strict_json_schema` 在不支持 strict mode 的后端上会怎样？（第7周 vLLM 接入时验证）
