"""管理端 Agent：ReAct 循环，模型自己决定查什么、调哪个工具、什么时候收尾。

为什么不像学生端那样「分类 → 固定分支」
--------------------------------------
学生端的问题类型有限（找文件 / 查数据 / 两者都要 / 做规划），可以把路径枚举成五个分支。
管理端枚举不出来：同一个人今天问「库里有多少份奖学金通知」，明天问「12 号那份为什么
搜不到」，后天问「转专业相关的都找出来看看，顺便把标题改一下」。这些问题的工具序列
事前不可知，硬编码分支只会写出一个永远追不上需求的 if/else。

所以这里只有两个节点，模型每一步自己判断下一步：

    agent ──有 tool_calls──> tools ──> agent
       └────无 tool_calls────> END

框架只负责三件事：执行它要的工具、把结果喂回去、别让它无限转。
"""

from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.llm import get_llm
from app.logging_config import get_logger
from app.tools.admin import ADMIN_TOOLS
from app.tools.common import tool_error_payload

log = get_logger(__name__)

# 最多几轮「决策 → 调工具」。管理端的查询往往要 2–3 步（先概览再定位），
# 留到 6 步足够，再多基本就是模型在绕圈，不如让它停下来问人。
MAX_STEPS = 6

TOOL_MAP = {t.name: t for t in ADMIN_TOOLS}


class AdminState(TypedDict, total=False):
    question: str
    # messages 必须用 add_messages：工具调用是「多轮消息」结构，普通 list 覆盖会把
    # tool_call 与 ToolMessage 的配对关系弄丢，模型下一轮就报「tool_call_id 不匹配」。
    messages: Annotated[list[Any], add_messages]
    answer: str
    steps: int
    trace: Annotated[list[dict], operator.add]
    records: Annotated[list[dict], operator.add]


ADMIN_SYSTEM = """你是「校园智助」的管理端助手，服务负责维护学校知识库的工作人员。

## 你自己决定怎么查
不要等对方告诉你用哪个工具，也不要按固定顺序走。每一步先想「要回答这个问题，
我还缺什么信息」，再决定调哪个工具、传什么参数；拿到结果后判断「够了吗」——
够了就直接回答，不够就继续调，最多 6 步。

## 工具怎么选
- 问总量、分布、异常（多少份、哪些类型、有没有原件丢了）→ admin_knowledge_overview
- 有明确筛选条件（类型 / 学院 / 年份 / 部门 / 标题关键词）→ admin_list_documents
- 不知道是哪份、只能按内容找（"哪些通知提到助学金额度"）→ admin_search_knowledge
- 已经知道 id、要看完整档案（状态、原件在不在、切片数）→ admin_get_document
- 要改标题 / 分类 / 学院 / 部门 / 年份 → admin_propose_metadata_change
- 要给文件改名（同目录内）→ admin_propose_move
- 管理员在对话框里传了附件 → 先 admin_list_attachments，再 admin_read_attachment 读正文

## 关于附件
附件只属于**本次对话**，没有进入知识库，所以检索工具搜不到它们。
附件正文也不会自动出现在你的上下文里，必须调 admin_read_attachment 才能看到。
用户说「照这份改」「参考这个格式」时，先把附件读出来再说；长文档按 offset 分段读。

## 这是多轮对话
用户会用「它」「那份」「刚才那个」指代前面提到的文件。往上读对话记录，
确认他指的是哪一份、哪个字段，不要每轮都重新问一遍。
只有两种情况下才反问：① 上一轮提到的候选文件不止一个、无法确定是哪个；
② 他没说是哪个字段但要改。其他情况直接接着做。

优先用精确查询，别一上来就语义检索：有条件的用 list，没条件的才用 search。

## 修改类请求的处理
你只能**提出**变更（生成待确认的变更单），没有执行权。所以：
1. 先用 list / get_document 确认是哪一份文件，拿到真实 id —— 绝不能猜 id
2. 再调 admin_propose_metadata_change，把要改的字段传进去
3. 把 diff 原样告诉用户，说明"需要你在管理端确认后才生效"
4. 绝对不要说"已经改好了"

## 回答要求
1. 简体中文，先给结论，再给依据。
2. 提到文件时必须带 id 和标题，管理员才能接着操作。
3. 只能依据工具返回的数据。没查到就说"知识库里没有找到"，不要换个说法敷衍。
4. 不要输出 JSON，不要提"工具调用"这类内部措辞，像同事一样说话。
"""


def _text_of(message: Any) -> str:
    content = getattr(message, "content", "")
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


def _brief(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _history_block(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    lines = [
        f"{'管理员' if turn.get('role') == 'user' else '助手'}：{turn.get('content', '')}"
        for turn in history[-6:]
    ]
    return "【前几轮对话】\n" + "\n".join(lines) + "\n\n"


def _agent_node(state: AdminState) -> dict[str, Any]:
    """一步决策：要么要工具，要么收尾。选哪个由模型决定。"""
    llm = get_llm().bind_tools(ADMIN_TOOLS)
    ai: AIMessage = llm.invoke(state["messages"])
    calls = getattr(ai, "tool_calls", None) or []
    steps = int(state.get("steps") or 0) + 1

    if calls:
        detail = "；".join(f"{c['name']}({_brief(c.get('args'), 80)})" for c in calls)
        trace = {
            "节点": "agent",
            "步骤": f"第 {steps} 步决策",
            "结果": f"决定调用 {len(calls)} 个工具",
            "说明": detail,
        }
    else:
        trace = {
            "节点": "agent",
            "步骤": f"第 {steps} 步决策",
            "结果": "信息足够，直接回答",
            "说明": "",
        }

    update: dict[str, Any] = {"messages": [ai], "steps": steps, "trace": [trace]}
    if not calls:
        update["answer"] = _text_of(ai)
    return update


def _tools_node(state: AdminState, config: Any = None) -> dict[str, Any]:
    """执行模型这一轮要的全部工具，结果原样回喂。

    工具报错**不抛出去**：把错误整理成结构化 JSON 塞回 ToolMessage，
    模型下一轮看到「参数不合法 / 没有这个 id」就能自己改一次。
    这是 Agent 和普通 API 的区别 —— 错误也是上下文的一部分。

    config 要一路传给工具：附件工具靠它拿 session_id，模型看不到这个参数。
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []

    messages: list[Any] = []
    records: list[dict[str, Any]] = []
    briefs: list[str] = []

    for call in calls:
        name = call.get("name") or ""
        args = call.get("args") or {}
        tool = TOOL_MAP.get(name)
        ok = True

        if tool is None:
            raw = json.dumps(
                {"错误": f"没有名为 {name} 的工具", "可用工具": list(TOOL_MAP)},
                ensure_ascii=False,
            )
            ok = False
        else:
            try:
                raw = tool.invoke(args, config=config)
            except Exception as exc:
                # 带上 retryable：参数写错可以让模型改一次，数据库断了再试也没用。
                raw = tool_error_payload(exc)
                ok = False

        messages.append(ToolMessage(content=str(raw), tool_call_id=call.get("id") or ""))
        records.append({"工具": name, "参数": args, "成功": ok, "结果": _brief(raw, 600)})
        briefs.append(f"{name} → {'成功' if ok else '失败'}")

    trace = {
        "节点": "tools",
        "步骤": f"执行 {len(calls)} 个工具",
        "结果": "；".join(briefs),
        "说明": "",
    }
    return {"messages": messages, "records": records, "trace": [trace]}


def _should_continue(state: AdminState) -> str:
    last = state["messages"][-1]
    if not (getattr(last, "tool_calls", None) or []):
        return END
    if int(state.get("steps") or 0) >= MAX_STEPS:
        log.warning("管理端 Agent 达到步数上限 %d，强制收尾", MAX_STEPS)
        return END
    return "tools"


def build_admin_graph():
    builder = StateGraph(AdminState)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", _tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


_GRAPH = None


def get_admin_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_admin_graph()
    return _GRAPH


def _attachment_block(attachments: list[dict[str, Any]] | None) -> str:
    """附件清单只放"元信息"，不塞正文 —— 正文由模型自己调工具按需读。"""
    if not attachments:
        return ""
    lines = [
        f"- {item.get('name')}（{item.get('chars', 0)} 字）"
        for item in attachments
        if item.get("name")
    ]
    if not lines:
        return ""
    return (
        "【本轮附件】\n" + "\n".join(lines)
        + "\n（附件正文不会自动进入上下文，需要时用 admin_read_attachment 读取）\n\n"
    )


def new_admin_state(
    question: str,
    history: list[dict[str, str]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
) -> AdminState:
    human = (
        f"{_history_block(history or [])}"
        f"{_attachment_block(attachments)}"
        f"【管理员的要求】\n{question}"
    )
    return {
        "question": question,
        "messages": [SystemMessage(content=ADMIN_SYSTEM), HumanMessage(content=human)],
        "steps": 0,
        "trace": [],
        "records": [],
    }


def _graph_config(session_id: str | None) -> dict[str, Any]:
    return {"configurable": {"session_id": session_id or "default"}}


def run_admin_agent(
    question: str,
    history: list[dict[str, str]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    final = get_admin_graph().invoke(
        new_admin_state(question, history, attachments),
        config=_graph_config(session_id),
    )

    answer = (final.get("answer") or "").strip()
    records = final.get("records") or []

    if not answer:
        # 达到步数上限还没收敛：把已经查到的东西整理成一句话，别给用户一个空气泡
        used = "、".join(dict.fromkeys(r["工具"] for r in records)) or "未调用工具"
        answer = (
            f"我查了 {len(records)} 次（{used}）还没得到确定结论。"
            "可以把条件说得更具体一点吗？例如文件类型、年份或标题关键词。"
        )

    return {
        "answer": answer,
        "trace": final.get("trace") or [],
        "records": records,
        "steps": final.get("steps") or 0,
    }


_CHANGE_RE = re.compile(r'"变更单"\s*:\s*"([0-9a-f]{32})"')


def _extract_change_id(record: dict[str, Any]) -> str | None:
    """从 propose 类工具的结果里抠出变更单号，好让前端直接渲染确认卡。"""
    match = _CHANGE_RE.search(str(record.get("结果") or ""))
    return match.group(1) if match else None


def _stream_impl(
    question: str,
    history: list[dict[str, str]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
):
    """按节点产出事件的流式版本。

    不是 token 级流式（那要把 LLM 也切成流式，收益在管理端不大），
    而是**步骤级**：模型每次决定调什么、每个工具返回什么、最后答案是什么，
    各发一个事件。前端因此能显示"正在查…正在改…"，
    而不是对着一个转圈的空白等 20 秒。
    """
    state = new_admin_state(question, history, attachments)
    records: list[dict[str, Any]] = []
    answer = ""
    steps = 0

    for chunk in get_admin_graph().stream(
        state, config=_graph_config(session_id), stream_mode="updates"
    ):
        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue

            if node == "agent":
                steps = max(steps, int(update.get("steps") or 0))
                messages = update.get("messages") or []
                ai = messages[-1] if messages else None
                calls = getattr(ai, "tool_calls", None) or []
                if calls:
                    yield {
                        "type": "plan",
                        "text": "决定调用：" + "、".join(c.get("name") or "" for c in calls),
                        "tools": [
                            {"name": c.get("name"), "args": c.get("args")} for c in calls
                        ],
                    }
                elif update.get("answer"):
                    answer = str(update["answer"])

            elif node == "tools":
                for record in update.get("records") or []:
                    records.append(record)
                    yield {
                        "type": "tool",
                        "name": record.get("工具"),
                        "ok": record.get("成功"),
                        "preview": str(record.get("结果") or "")[:300],
                    }
                    change_id = _extract_change_id(record)
                    if change_id:
                        yield {
                            "type": "change",
                            "change_id": change_id,
                            "tool": record.get("工具"),
                        }

    if not answer:
        used = "、".join(dict.fromkeys(r["工具"] for r in records)) or "未调用工具"
        answer = (
            f"我查了 {len(records)} 次（{used}）还没得到确定结论。"
            "可以把条件说得更具体一点吗？例如文件类型、年份或标题关键词。"
        )

    yield {"type": "answer", "text": answer}
    yield {
        "type": "done",
        "steps": steps,
        "records": records,
    }


def stream_admin_agent(
    question: str,
    history: list[dict[str, str]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
):
    """流式入口。

    会话 id 走 LangGraph 的 config（`configurable.session_id`）而不是 ContextVar：
    StreamingResponse 是一段一段 next() 的，可能换线程执行，ContextVar 在
    换线程后会回到默认值 —— 表现就是「附件明明传了，工具却说没有」。
    config 是随调用传递的，不依赖线程上下文。
    """
    yield from _stream_impl(question, history, attachments, session_id)


if __name__ == "__main__":
    samples = [
        "库里现在有多少份文件？按类型分布是什么样的",
        "帮我找 2025 年计算机学院的奖学金通知",
        "把国家奖学金那份的标题改成「2025 年国家奖学金评审通知」",
    ]
    for question in samples:
        print("=" * 72)
        print(f"问：{question}")
        result = run_admin_agent(question)
        for step in result["trace"]:
            print(f"  · [{step['节点']}] {step['步骤']}：{step['结果']}  {step['说明']}")
        print(f"答：{result['answer']}\n")
