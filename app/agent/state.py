"""Agent 的状态定义。

LangGraph 的核心思想是「状态机」：每个节点接收完整状态，返回**增量**，
框架负责合并。所以这里要明确哪些字段用 reducer（可累加），哪些直接覆盖。

    trace / data_records 用 operator.add —— 每个节点都能往里追加，
        不需要先读出来再拼回去（这也是 LangGraph 比手写流程更省心的地方）。
    其余字段是覆盖语义 —— 后写入的节点会替换掉旧值。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

# 一次取数行为的真实结局。刻意不用 bool，因为「查空了」和「查询出错」
# 需要不同的下游处理：前者可以回退到另一条数据源，后者不能当成"没有"。
DataQueryStatus = Literal["success", "empty", "error", "incomplete"]


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
    # 这次检索的真实结局。None = 还没跑过（不能默认成 empty，否则
    # 未经检索的 general 分支会被误判为"检索为空"）。
    doc_query_status: DataQueryStatus | None

    # ---- 结构化数据分支产物 ----
    data_facts: str                                    # 工具结果的事实性总结
    data_records: Annotated[list[dict], operator.add]  # 原始工具调用记录（给前端展示）
    # 这两个字段是「回退决策」的唯一依据。**不能用 data_records 是否为空代替**：
    # 工具被调用了但返回 0 条时，records 非空而结果为空 —— 这正是要区分的场景。
    data_query_status: DataQueryStatus | None
    data_has_results: bool

    # ---- 回退控制 ----
    # 只允许回退一次。没有这个标记，hybrid 会 docs→data→docs→data 无限循环。
    doc_fallback_attempted: bool

    # ---- 证据与预算（P0 契约层）----
    # evidence 刻意**不用** operator.add：hybrid / planning 会跑两条分支，
    # 两条都要写证据；但一次重试或重跑会让同一个 task_id 出现两条，
    # 累加会让下游的覆盖度判断重复计数。
    # 正确做法：节点读旧值 → evidence.merge_evidence() 按 task_id 替换 → 整体写回。
    evidence: list[dict[str, Any]]
    budget: dict[str, Any]        # max_steps / max_tool_calls，缺省值见 graph.DEFAULT_BUDGET

    # ---- 校验结论 ----
    verdict: str                  # pass / warn / degrade
    caveats: list[str]            # 给用户看的提醒，前端渲染在回答下方

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
        "evidence": [],
        "budget": {},
        "caveats": [],
        # 查询状态刻意初始化为 None / False，不是 "empty"。
        # "还没查" 和 "查了但没结果" 是两件事，混在一起会让回退条件误触发。
        "data_query_status": None,
        "data_has_results": False,
        "doc_query_status": None,
        "doc_fallback_attempted": False,
    }
