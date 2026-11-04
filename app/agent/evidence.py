"""证据（Evidence）：Agent 的「可判定中间产物」。

问题不在「没有结构」，而在结构选错了位置：
    检索结果被 format_context 拼成一大段文本，交给 compose 之后就消失了。
    此后所有环节——校验、引用编号、评测归因——都只能对着字符串猜。

Evidence 把「一次取数行为的结果」固定成一条记录：

    谁产生的（tool）          这条证据是哪一步来的
    回答了哪个子任务（task_id） 多子任务时判断「哪个还没查到」
    对应哪些源（source_ids）   输出层据此分配引用编号
    可靠度（score / low_confidence）
    有没有放宽条件（relaxed / filter_rejected）

有它，校验层才有东西可判、输出层才有依据编号、评测才能做归因。
这不是封装，是整条链路的地基。

为什么存 dict 而不是 dataclass 实例：
    LangGraph 的 state 要能序列化（将来接 checkpointer 要落盘），
    dataclass 进去要么被 pickle、要么在 checkpoint 后变成 dict —— 两种都容易踩。
    统一用 dict，规则简单：进 state 的一切都必须是 JSON 可序列化的。
"""

from __future__ import annotations

import json
from typing import Any

# 证据正文在 state 里最多留多少字。只用于校验和归因，不用于生成，
# 所以可以比真正进上下文的 doc_context 短得多。
MAX_EVIDENCE_TEXT = 6000


def _clip(text: str, limit: int = MAX_EVIDENCE_TEXT) -> str:
    return text if len(text) <= limit else text[:limit] + "……"


# 工具返回里表示「一共几条」的键。各工具命名不统一，集中在这一处维护。
_COUNT_KEYS = ("数量", "推荐数量", "记录数", "count")
# 表示「结果列表」的键：没有 count 键时按列表长度判断
_LIST_KEYS = ("考试列表", "竞赛列表", "推荐列表", "文件清单", "记录", "items")


def payload_is_empty(payload: Any) -> bool | None:
    """判断一次工具返回是不是「查到了 0 条」。

    True=空 / False=有内容 / None=无法判断（应看「错误」字段，归到 error）

    为什么不能只看 `if not payload`：
        query_exam 查空时返回的是
        `{"今天日期": ..., "数量": 0, "考试列表": [], "说明": "确实没有记录"}`
        —— 它是一个**非空字典**，但语义上是空结果。用真假值判断会把
        「查空了」当成「查到了」，这正是本次要修的病根。

    为什么 error 不在这里归成 empty：
        「库连不上」和「库里没有」是两件事。混在一起会让回退逻辑在
        数据库故障时去跑一次毫无意义的文档检索，还掩盖了真正的问题。
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(payload, dict):
        return None
    if payload.get("错误"):
        return None

    for key in _COUNT_KEYS:
        if key in payload:
            try:
                return int(payload[key]) == 0
            except (TypeError, ValueError):
                return None
    for key in _LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return len(value) == 0

    # 单对象工具（get_student_profile 的 {"学生信息": {...}}）走到这里：
    # 能拿回内容就说明查到了
    return not bool(payload)


def from_doc_search(
    result: dict[str, Any],
    *,
    task_id: str,
    tool: str = "search_documents",
    text: str | None = None,
) -> dict[str, Any]:
    """把一次文件检索的结果转成一条证据。

    result 就是 search_documents 的返回值。text 不传时用 hits 拼一份精简版，
    只服务于校验（数字溯源），不进 LLM 上下文。
    """
    docs = result.get("documents") or []
    hits = result.get("hits") or []

    body = text
    if body is None:
        body = "\n\n".join(str(h.get("content") or "") for h in hits)

    return {
        "task_id": task_id,
        "kind": "document",
        "tool": tool,
        "titles": [d.get("title") for d in docs if d.get("title")],
        "source_ids": [d.get("id") for d in docs if d.get("id") is not None],
        "text": _clip(body),
        "score": result.get("max_score"),
        "low_confidence": bool(result.get("low_confidence")),
        "relaxed": bool(result.get("fallback")),
        "filter_rejected": [],
        "count": len(docs),
    }


def from_tool_records(
    records: list[dict[str, Any]],
    *,
    task_id: str,
) -> dict[str, Any]:
    """把 ReAct 循环里的工具调用记录汇总成一条证据。

    关键区分（这次修复的核心）：
        **调用过 ≠ 查到了。** 一次 `query_exam(keyword="体测")` 返回
        `{"数量": 0}`，调用是成功的，但事实结论是「没有这条数据」。
        如果把它当证据正文交给校验层，校验层会认为「有依据」，
        于是「未检索到」这种回答被判 pass —— 这就是静默失败的根源。

    所以这里按结果分成两种 kind：
        structured_record  真的取回了记录，可以证明事实
        tool_trace         只证明「调用发生过」，不能证明任何事实

    结构化数据没有「相关度」，可靠度取决于工具是否成功：
        只要有一半以上调用被标记为失败，整条证据就降级为不可靠。
    """
    if not records:
        return _empty(task_id, "tool_trace")

    parts: list[str] = []
    rejected: list[str] = []
    failed = 0
    has_results = False
    for record in records:
        payload = record.get("结果")
        if isinstance(payload, dict):
            # 工具自己报的结构化提示：被忽略的枚举值、参数调整说明
            for key in ("参数调整", "提示"):
                value = payload.get(key)
                if value:
                    rejected.extend(value if isinstance(value, list) else [str(value)])
            if payload.get("错误"):
                failed += 1
            elif payload_is_empty(payload) is False:
                has_results = True
        parts.append(
            f"{record.get('工具')}("
            f"{json.dumps(record.get('参数') or {}, ensure_ascii=False)}) -> "
            f"{json.dumps(payload, ensure_ascii=False, default=str)}"
        )

    return {
        "task_id": task_id,
        "kind": "structured_record" if has_results else "tool_trace",
        "tool": ",".join(dict.fromkeys(str(r.get("工具")) for r in records)),
        "titles": [],
        "source_ids": [],
        # 没有实际记录时不留正文。留着调用日志，校验层就会把空查询当成有依据。
        "text": _clip("\n".join(parts)) if has_results else "",
        "score": None,
        "low_confidence": failed * 2 >= len(records),
        "relaxed": False,
        "filter_rejected": list(dict.fromkeys(rejected)),
        "count": len(records),
        # 明确的是否取到记录，供 verifier 判断，不用再去猜 text 的字数
        "has_results": has_results,
    }


def _empty(task_id: str, kind: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "kind": kind,
        "tool": "",
        "titles": [],
        "source_ids": [],
        "text": "",
        "score": None,
        "low_confidence": False,
        "relaxed": False,
        "filter_rejected": [],
        "count": 0,
        "has_results": False,
    }


def merge_evidence(
    existing: list[dict[str, Any]] | None,
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 task_id 合并：同一子任务用新结果替换，不同子任务累积。

    为什么不用 operator.add：hybrid / planning 会跑两条分支，
    累加本身没错；但一次重试或重跑会让同一个 task_id 出现两条，
    下游按 task 判覆盖度时就会重复计数。所以合并规则必须是「按 key 替换」。

    为什么不用覆盖语义：retrieve_docs 和 query_data 都要写 evidence，
    后跑的会把先跑的直接抹掉 —— 这正是 sources 当初踩过的坑。
    """
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for item in list(existing or []) + list(incoming):
        key = str(item.get("task_id") or "")
        if not key:
            continue
        if key not in merged:
            order.append(key)
        merged[key] = item

    return [merged[key] for key in order]


def evidence_text(evidence: list[dict[str, Any]] | None) -> str:
    """把全部证据正文拼成一段，供校验层做数字溯源。"""
    return "\n\n".join(
        str(item.get("text") or "") for item in (evidence or []) if item.get("text")
    )


def source_count(evidence: list[dict[str, Any]] | None) -> int:
    """证据里一共提到多少份文件（用于校验引用编号范围）。"""
    titles: list[str] = []
    for item in evidence or []:
        for title in item.get("titles") or []:
            if title and title not in titles:
                titles.append(title)
    return len(titles)
