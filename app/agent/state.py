"""Agent 的状态定义。

LangGraph 的核心思想是「状态机」：每个节点接收完整状态，返回**增量**，
框架负责合并。所以这里要明确哪些字段用 reducer（可累加），哪些直接覆盖。

    trace / data_records 用 operator.add —— 每个节点都能往里追加，
        不需要先读出来再拼回去（这也是 LangGraph 比手写流程更省心的地方）。
    其余字段是覆盖语义 —— 后写入的节点会替换掉旧值。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # ---- 输入 ----
    question: str
    history: list[dict[str, str]]    # 最近几轮对话 [{role, content}]，由调用方维护

    # ---- 路由 ----
    route: str                       # document / structured / hybrid / planning / general
    plan: dict[str, Any]             # RoutePlan 序列化后的检索参数

    # ---- RAG 分支产物 ----
    doc_hits: list[dict[str, Any]]       # chunk 级检索结果
    doc_documents: list[dict[str, Any]]  # 按文件去重后的结果（给前端展示卡片）
    doc_context: str                     # 拼好的上下文文本
    doc_context_count: int               # 上下文里实际给了几段资料，决定引用编号的上限

    # ---- 结构化数据分支产物 ----
    data_facts: str                                    # 工具结果的事实性总结
    data_records: Annotated[list[dict], operator.add]  # 原始工具调用记录（给前端展示）

    # ---- 输出 ----
    # 注意：这里刻意不用 operator.add。来源是「本轮检索结果」，该覆盖不该累加；
    # 一旦改成累加，同一个文件在每轮都会被追加一次，编号也会错位。
    sources: list[dict[str, Any]]
    answer: str

    # ---- 可观测性 ----
    trace: Annotated[list[dict], operator.add]  # 每个节点的执行轨迹


def new_state(question: str, history: list[dict[str, str]] | None = None) -> AgentState:
    return {
        "question": question,
        "history": history or [],
        "trace": [],
        "data_records": [],
        "sources": [],
    }
