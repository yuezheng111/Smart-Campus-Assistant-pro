"""LangGraph 工作流：Agent 的主干。

图结构（节点之间的箭头就是「什么时候该做什么」的显式表达）：

                        ┌──────────────┐
                        │   classify   │  意图识别 + 参数抽取
                        └──────┬───────┘
              ┌────────────┬───┴────┬─────────────┬─────────┐
              ▼            ▼        ▼             ▼         ▼
         document     structured  hybrid      planning   general
              │            │        │             │         │
              ▼            │        ▼             │         │
        retrieve_docs      │   retrieve_docs      │         │
              │            │        │             │         │
              │            ▼        ▼             ▼         │
              │       query_data ───┘        query_data     │
              │                             │              │
              │                             ▼              │
              │                        retrieve_docs        │
              ▼                             │              ▼
            ┌───────────────────────────────┴──────────────┐
            │                   compose                     │  整合生成最终回答
            └───────────────────┬───────────────────────────┘
                                ▼
                               END

为什么值得用 LangGraph 而不是 if/else：
1. 分支、汇合、执行顺序都是**图结构**，一眼能看懂，改流程不用改动节点内部代码；
2. 每个节点的输入输出就是 state 的增量，天然可观测 —— trace 直接把执行路径吐给前端；
3. 后面要加「重排」「多轮追问」「人工确认」这些环节，只是往图里插节点，不用重构。
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.agent.llm import get_llm
from app.agent.router import classify
from app.agent.state import AgentState, new_state
from app.rag.retriever import format_context, search_documents
from app.tools import DATA_TOOLS

TOOL_MAP = {t.name: t for t in DATA_TOOLS}
MAX_TOOL_STEPS = 4
# 拼进 LLM 上下文的文件数上限（与 format_context 的 limit_docs 保持一致）
CONTEXT_DOC_LIMIT = 5

# ----------------------------------------------------------------------
# 节点 1：意图识别
# ----------------------------------------------------------------------


def classify_node(state: AgentState) -> dict:
    plan = classify(state["question"])
    return {
        "route": plan.route,
        "plan": plan.model_dump(),
        "trace": [
            {
                "节点": "classify",
                "步骤": "意图识别",
                "结果": f"路由 → {plan.route}",
                "说明": plan.reason,
            }
        ],
    }


# ----------------------------------------------------------------------
# 节点 2：RAG 检索（走第一条线：学校文件）
# ----------------------------------------------------------------------


def retrieve_docs_node(state: AgentState) -> dict:
    plan = state.get("plan") or {}
    query = plan.get("rewritten_query") or state["question"]
    result = search_documents(
        query,
        doc_type=plan.get("doc_type") or None,
        college=plan.get("college") or None,
        year=plan.get("year") or None,
        department=plan.get("department") or None,
    )

    sources = [
        {
            "title": d["title"],
            "publish_date": d["publish_date"],
            "department": d["department"],
            "doc_type": d["doc_type"],
            "college": d.get("college"),
            "url": d["url"],
            "file_path": d["file_path"],
            # 命中片段：前端打开原文时用它把「回答实际参考的那段话」高亮出来
            "snippet": d.get("snippet") or "",
            "score": d["score"],
        }
        for d in result["documents"]
    ]

    detail = f"命中 {len(result['documents'])} 份文件"
    if result["fallback"]:
        detail += "（带过滤条件无结果，已放宽条件重查）"

    return {
        "doc_hits": result["hits"],
        "doc_documents": result["documents"],
        "doc_context": format_context(result["hits"]),
        # 上下文里实际给出了几段资料，compose 要靠它来约束引用编号范围
        "doc_context_count": min(len(result["documents"]), CONTEXT_DOC_LIMIT),
        "sources": sources,
        "trace": [
            {
                "节点": "retrieve_docs",
                "步骤": "学校文件检索（RAG）",
                "结果": detail,
                "说明": f"检索语句：{query}｜过滤条件：{result['filter_used'] or '无'}",
            }
        ],
    }


# ----------------------------------------------------------------------
# 节点 3：结构化数据查询（走第二条线：MySQL，ReAct 循环）
# ----------------------------------------------------------------------

DATA_SYSTEM = """你是一个校园数据查询助手。你的唯一任务是通过调用工具**取回事实数据**，
然后客观地把结果整理成「事实清单」。

必须遵守：
1. 只依据工具返回的结果陈述，绝对不要编造数字、日期或名称。
2. 逐条列出事实，格式如：「2024 年蓝桥杯报名时间：2023-11-24 至 2024-02-07」。
3. 工具没有返回结果时，写「未查询到相关数据」，不要猜测，也不要换个说法敷衍。
4. 不要给建议、不要下结论、不要寒暄 —— 建议由后续环节统一生成。
5. 如果字段里带有「阶段」（正在报名 / 未开始报名 等），请原样保留。
"""


def _plan_hints(plan: dict[str, Any]) -> str:
    """把路由抽出的参数整理成提示，减少模型「瞎猜参数」的概率。"""
    mapping = [
        ("category", "竞赛类别"),
        ("level", "竞赛级别"),
        ("exam_category", "考试类别"),
        ("keyword", "关键词"),
        ("grade", "学生年级"),
        ("major", "学生专业"),
        ("interests", "兴趣方向"),
        ("year", "年份"),
    ]
    hints = [f"{label}={plan[key]}" for key, label in mapping if plan.get(key)]
    return ("；".join(hints)) if hints else "（无额外限定）"


def _text_of(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(content).strip()


def _safe_json(raw: Any, limit: int = 12000) -> Any:
    text = raw if isinstance(raw, str) else str(raw)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text[:limit]


def query_data_node(state: AgentState) -> dict:
    plan = state.get("plan") or {}
    llm = get_llm().bind_tools(DATA_TOOLS)

    user_prompt = (
        f"学生问题：{state['question']}\n"
        f"已识别出的查询限定条件：{_plan_hints(plan)}\n\n"
        "请调用合适的工具取数，然后把结果整理成事实清单。"
    )
    messages: list[Any] = [SystemMessage(content=DATA_SYSTEM), HumanMessage(content=user_prompt)]

    records: list[dict[str, Any]] = []
    facts = ""

    for _ in range(MAX_TOOL_STEPS):
        ai = llm.invoke(messages)
        messages.append(ai)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            facts = _text_of(ai)
            break

        for call in calls:
            name = call["name"]
            tool = TOOL_MAP.get(name)
            if tool is None:
                messages.append(
                    ToolMessage(content=f"没有名为 {name} 的工具", tool_call_id=call["id"])
                )
                continue
            try:
                raw = tool.invoke(call["args"])
            except Exception as exc:  # 参数不合法、查库异常都回吐给模型，让它自己修
                raw = json.dumps(
                    {"错误": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
                )
            records.append(
                {"工具": name, "参数": call["args"], "结果": _safe_json(raw)}
            )
            messages.append(ToolMessage(content=str(raw), tool_call_id=call["id"]))

    if not facts:  # 达到步数上限还没收敛，用原始结果兜底，避免空气泡
        facts = "\n".join(
            f"- {r['工具']}({r['参数']}) 返回：{json.dumps(r['结果'], ensure_ascii=False)[:600]}"
            for r in records
        )

    tool_names = ", ".join(dict.fromkeys(r["工具"] for r in records)) or "未调用工具"
    return {
        "data_facts": facts,
        "data_records": records,
        "trace": [
            {
                "节点": "query_data",
                "步骤": "结构化数据查询（MySQL Tools）",
                "结果": f"调用 {len(records)} 次工具：{tool_names}",
                "说明": f"限定条件：{_plan_hints(plan)}",
            }
        ],
    }


# ----------------------------------------------------------------------
# 节点 4：整合生成
# ----------------------------------------------------------------------

COMPOSE_SYSTEM = """你是「校园学习助手」，面向本校学生回答问题。

今天是 {today}。

严格按以下规则作答：
1. 只能依据【学校文件原文片段】和【结构化数据查询结果】里的信息回答，不得编造。
2. 资料里没有的内容，明确写「未检索到相关内容」，不要用常识去补，也不要假装知道。
3. 引用了学校文件的句子，在句末标注 [资料1]、[资料2] 这样的编号。
   本次一共只提供了 {doc_count} 段资料，编号范围就是 [资料1] 到 [资料{doc_count}]——
   绝对不要引用不存在的编号。
   不要在回答末尾另列「信息来源」「参考资料」清单 —— 前端会按编号自动列出文件卡片，
   你再抄一遍就是重复。
4. 涉及时间、金额、名额、条件等事实，必须与资料完全一致，一个字都不要改。
5. 【关于日期，最容易出错，务必注意】学校文件里的日期（例如「9 月 15 日—9 月 21 日」）
   属于**该文件发布的那一年**，不是今年。判断「现在还能不能报名」时：
   - 必须先看文件的发布年份。如果文件是往年发布的，那个时间窗口早就过去了，
     绝不能因为月日与今天接近就写成「正在报名」；
   - 只有【结构化数据查询结果】里明确给出了「阶段」字段时，才可以断言当前报名状态；
   - 没有「阶段」字段时，如实说明「资料未给出当前报名状态，请以官网/最新通知为准」。
6. 规划类问题要给出可执行的分步建议，并写明时间节点。
7. 用简体中文，结构清晰，善用列表和小标题。不要输出 JSON，不要提「资料片段」这种内部措辞。
"""


def _history_block(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    lines = []
    for turn in history[-6:]:
        role = "学生" if turn.get("role") == "user" else "助手"
        lines.append(f"{role}：{turn.get('content', '')}")
    return "【前几轮对话】\n" + "\n".join(lines) + "\n\n"


def compose_node(state: AgentState) -> dict:
    plan = state.get("plan") or {}
    doc_context = state.get("doc_context") or "（本轮没有检索学校文件）"
    data_facts = state.get("data_facts") or "（本轮没有查询结构化数据）"

    system = COMPOSE_SYSTEM.format(
        today=dt.date.today().isoformat(),
        doc_count=state.get("doc_context_count") or 0,
    )
    human = (
        f"{_history_block(state.get('history') or [])}"
        f"【学生问题】\n{state['question']}\n\n"
        f"【本题意图】\n{plan.get('reason', '')}\n\n"
        f"【学校文件原文片段】\n{doc_context}\n\n"
        f"【结构化数据查询结果】\n{data_facts}"
    )

    answer = get_llm().invoke(
        [SystemMessage(content=system), HumanMessage(content=human)]
    )
    return {
        "answer": _text_of(answer),
        "trace": [
            {
                "节点": "compose",
                "步骤": "整合生成回答",
                "结果": "已生成最终回答",
                "说明": f"上下文来源：{'学校文件 ' + str(len(state.get('doc_hits') or [])) + ' 段' if state.get('doc_hits') else '无文件'}；"
                        f"{'工具记录 ' + str(len(state.get('data_records') or [])) + ' 条' if state.get('data_records') else '无数据'}",
            }
        ],
    }


# ----------------------------------------------------------------------
# 组装图
# ----------------------------------------------------------------------


def _after_classify(state: AgentState) -> str:
    return state.get("route") or "general"


def _after_docs(state: AgentState) -> str:
    # hybrid 需要「文件原文 + 结构化字段」，所以检索完文件接着查库
    return "query_data" if state.get("route") == "hybrid" else "compose"


def _after_data(state: AgentState) -> str:
    # planning 需要「个性化推荐 + 章程原文」，所以查完库再检索文件
    return "retrieve_docs" if state.get("route") == "planning" else "compose"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify_node)
    builder.add_node("retrieve_docs", retrieve_docs_node)
    builder.add_node("query_data", query_data_node)
    builder.add_node("compose", compose_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        _after_classify,
        {
            "document": "retrieve_docs",
            "structured": "query_data",
            "hybrid": "retrieve_docs",
            "planning": "query_data",
            "general": "compose",
        },
    )
    builder.add_conditional_edges(
        "retrieve_docs",
        _after_docs,
        {"query_data": "query_data", "compose": "compose"},
    )
    builder.add_conditional_edges(
        "query_data",
        _after_data,
        {"retrieve_docs": "retrieve_docs", "compose": "compose"},
    )
    builder.add_edge("compose", END)

    return builder.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in sources:
        key = item.get("title", "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _attach_source_ids(sources: list[dict[str, Any]]) -> None:
    """给来源补上 MySQL 里的 document.id。

    向量库只存了 file_path，而前端要「打开原文」必须拿到 id。
    文件表只有几十行，直接全量取一次映射，比按路径拼 IN 查询更省事，
    也让前端不必再为了定位一份文件去拉整个列表。
    """
    if not sources:
        return
    try:
        from app import db

        id_map = {
            _normalize_path(row["file_path"]): row["id"]
            for row in db.fetch_all("SELECT id, file_path FROM document")
            if row.get("file_path")
        }
    except Exception:
        return
    for item in sources:
        item["id"] = id_map.get(_normalize_path(item.get("file_path")))


def run_agent(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """跑一次完整的 Agent，返回最终 state（含回答、来源、执行轨迹）。"""
    final = get_graph().invoke(new_state(question, history))
    final["sources"] = _dedupe_sources(final.get("sources") or [])
    _attach_source_ids(final["sources"])

    # 引用编号要和 compose 里的 [资料N] 一一对应，所以按检索顺序回填，
    # 且只给真正进了上下文的那些文件编号（超出上限的不标）。
    ref_limit = final.get("doc_context_count") or 0
    for index, doc in enumerate(final.get("doc_documents") or [], start=1):
        if index > ref_limit:
            break
        for source in final["sources"]:
            if source["title"] == doc["title"]:
                source["ref"] = index
                break

    return final


if __name__ == "__main__":
    samples = [
        "帮我找一下2025年国家奖学金评选通知",
        "2022到2026年蓝桥杯分别什么时候报名？",
        "帮我找2025年计算机学院的竞赛通知，顺便看看报名截止时间",
        "我是大二软件工程专业的学生，正在学 Web 开发，明年适合参加哪些竞赛？",
        "你好呀",
    ]
    for question in samples:
        print("=" * 72)
        print(f"❓ {question}")
        result = run_agent(question)
        print(f"🧭 路由：{result['route']}")
        for step in result["trace"]:
            print(f"   · {step['步骤']}：{step['结果']}")
        print(f"💬 回答：\n{result['answer']}")
        if result["sources"]:
            print("📎 来源：" + "；".join(s["title"] for s in result["sources"][:3]))
        print()
