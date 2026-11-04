"""管理端工具组：只读地看整个知识库。

和学生端工具（DOC_TOOLS）的差别只有三点，也正是「同一套 RAG、不同视角」的全部实现：

    1. 默认**不**过滤状态 —— 已废止的、原件缺失的都要能看见，否则管理员没法管；
    2. 多返回内部字段（id / status / file_missing_at / 切片数），这些不该给学生看；
    3. 多了聚合统计 —— 管理端最常问的是「库里现在什么情况」。

这里全部是只读工具。改元数据的写工具（update_document_meta / move_document）
在 P1 单独加，并且要走 dry-run + 确认 + 审计，不和读工具混在一组里。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app import config, db
from app.admin_change import EDITABLE_FIELDS, create_change
from app.rag.retriever import format_context, search_documents_admin

_columns_cache: set[str] | None = None


def _columns() -> set[str]:
    global _columns_cache
    if _columns_cache is None:
        try:
            _columns_cache = {
                str(row.get("Field") or "")
                for row in db.fetch_all("SHOW COLUMNS FROM `document`")
            }
        except Exception:
            _columns_cache = set()
    return _columns_cache


def _has_lifecycle() -> bool:
    return {"status", "file_missing_at"} <= _columns()


def _select_columns() -> str:
    base = (
        "id, doc_no, title, department, doc_type, college, publish_date, year, "
        "file_path, url, summary"
    )
    if _has_lifecycle():
        base += ", status, superseded_by, superseded_at, file_missing_at"
    if "content_hash" in _columns():
        base += ", content_hash"
    return base


def _status_of(row: dict[str, Any]) -> str:
    if not _has_lifecycle():
        return "active"
    if row.get("file_missing_at"):
        return "原件缺失"
    return "已废止" if row.get("status") == "superseded" else "现行"


def _file_exists(file_path: str) -> bool:
    rel = str(file_path or "").strip().replace("\\", "/")
    if not rel:
        return False
    candidate = Path(rel)
    target = candidate if candidate.is_absolute() else config.BASE_DIR / candidate
    return target.is_file()


def _shape(row: dict[str, Any]) -> dict[str, Any]:
    """给模型看的文件摘要：只留决策需要的字段，不要把整行原样吐出去。"""
    return {
        "id": row.get("id"),
        "title": row.get("title"),
        "doc_type": row.get("doc_type"),
        "college": row.get("college"),
        "department": row.get("department"),
        "publish_date": str(row.get("publish_date") or ""),
        "状态": _status_of(row),
        "file_path": row.get("file_path"),
        "原件在磁盘": _file_exists(str(row.get("file_path") or "")),
    }


@tool
def admin_knowledge_overview() -> str:
    """知识库总体情况：文件总数、现行/废止/原件缺失的分布、按类型和学院的统计、向量片段数。

    管理员问「现在库里有多少文件」「哪些类型积压最多」「有没有原件丢了」
    「最近入库正常吗」这类**全局性**问题时先调这个，先看整体再看局部。
    它不返回具体文件清单，要清单改用 admin_list_documents。
    """
    payload: dict[str, Any] = {}
    try:
        payload["文件总数"] = int(
            (db.fetch_one("SELECT COUNT(*) AS c FROM `document`") or {}).get("c") or 0
        )
    except Exception as exc:
        return json.dumps({"错误": f"读取文件表失败：{exc}"}, ensure_ascii=False)

    if _has_lifecycle():
        try:
            status_rows = db.fetch_all(
                "SELECT status, COUNT(*) AS c FROM `document` GROUP BY status"
            )
            payload["按状态"] = {str(r["status"]): int(r["c"]) for r in status_rows}
            missing = db.fetch_one(
                "SELECT COUNT(*) AS c FROM `document` WHERE file_missing_at IS NOT NULL"
            )
            payload["原件缺失"] = int((missing or {}).get("c") or 0)
        except Exception as exc:
            payload["状态统计失败"] = str(exc)
    else:
        payload["提示"] = "数据库缺少生命周期字段，无法统计废止/缺失"

    for label, sql, key in (
        ("按类型", "SELECT doc_type, COUNT(*) AS c FROM `document` "
                   "GROUP BY doc_type ORDER BY c DESC", "doc_type"),
        ("按学院", "SELECT college, COUNT(*) AS c FROM `document` "
                   "GROUP BY college ORDER BY c DESC", "college"),
        ("按部门", "SELECT department, COUNT(*) AS c FROM `document` "
                   "GROUP BY department ORDER BY c DESC", "department"),
    ):
        try:
            payload[label] = {
                str(r[key] or "未标注"): int(r["c"]) for r in db.fetch_all(sql)
            }
        except Exception as exc:
            payload[label] = f"统计失败：{exc}"

    try:
        from app.rag.vectorstore import count as vector_count

        payload["向量片段数"] = int(vector_count())
    except Exception as exc:
        payload["向量片段数"] = f"读取失败：{exc}"

    return json.dumps(payload, ensure_ascii=False, default=str)


@tool
def admin_list_documents(
    doc_type: str = "",
    college: str = "",
    year: int = 0,
    department: str = "",
    keyword: str = "",
    status: str = "",
    limit: int = 20,
) -> str:
    """按精确条件列出文件清单（元数据，不含正文）。管理端默认能看见**全部**状态。

    什么时候用它、而不是 admin_search_knowledge：
    只要问题里出现了明确的筛选条件（类型、学院、年份、部门、标题关键词），就用这个。
    它是精确匹配 + 可翻页，结果可控；语义检索只适合「不知道是哪份文件、只能按内容找」。

    Args:
        doc_type: 文件类型，如「奖学金」「考试」「竞赛」。不确定传空字符串。
        college: 学院，如「计算机学院」。传空字符串表示不限学院。
        year: 年份，如 2024。不确定传 0。
        department: 发布部门，如「教务处」。不确定传空字符串。
        keyword: 标题关键词，如「四六级」。不确定传空字符串。
        status: 状态筛选。可选值："" 表示全部；"active" 只列现行；
            "superseded" 只列已废止；"missing" 只列原件缺失的；"present" 只列原件在的。
        limit: 最多返回多少条，默认 20，上限 50。
    """
    conditions: list[tuple[str, str, object]] = []
    if doc_type:
        conditions.append(("doc_type", "=", doc_type))
    if college:
        conditions.append(("college", "=", college))
    if year:
        conditions.append(("year", "=", int(year)))
    if department:
        conditions.append(("department", "=", department))
    if keyword:
        conditions.append(("title", "like", keyword))

    if _has_lifecycle():
        key = (status or "").strip().lower()
        if key == "active":
            conditions.append(("status", "=", "active"))
            conditions.append(("file_missing_at", "is_null", None))
        elif key == "superseded":
            conditions.append(("status", "=", "superseded"))
        elif key == "missing":
            conditions.append(("file_missing_at", "is_not_null", None))
        elif key == "present":
            conditions.append(("file_missing_at", "is_null", None))

    where, params = db.build_where(conditions)
    params["limit"] = max(1, min(int(limit), 50))
    sql = (
        f"SELECT {_select_columns()} FROM document{where} "
        "ORDER BY publish_date DESC, id DESC LIMIT :limit"
    )
    try:
        rows = db.fetch_all(sql, params)
    except Exception as exc:
        return json.dumps({"错误": f"查询失败：{exc}"}, ensure_ascii=False)

    return json.dumps(
        {
            "筛选条件": {k: v for k, v in params.items() if k != "limit"},
            "数量": len(rows),
            "文件清单": [_shape(r) for r in rows],
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def admin_get_document(doc_id: int) -> str:
    """按 id 看一份文件的完整档案：元数据、状态、原件是否在磁盘、切片数、向量库是否有它的片段。

    已经知道是哪份文件（从列表或检索结果拿到 id）时用它。它比列表多给内部字段，
    适合回答「这份文件为什么搜不到」「它的原件还在不在」这类问题。

    Args:
        doc_id: 文件编号，来自 admin_list_documents 或 admin_search_knowledge 的结果。
    """
    try:
        row = db.fetch_one(
            f"SELECT {_select_columns()} FROM document WHERE id = :i",
            {"i": int(doc_id)},
        )
    except Exception as exc:
        return json.dumps({"错误": f"查询失败：{exc}"}, ensure_ascii=False)
    if not row:
        return json.dumps(
            {"错误": f"没有 id={doc_id} 的文件", "提示": "先用 admin_list_documents 找到正确的 id"},
            ensure_ascii=False,
        )

    payload: dict[str, Any] = {"档案": _shape(row)}
    payload["档案"]["doc_no"] = row.get("doc_no") or ""
    payload["档案"]["url"] = row.get("url") or ""
    payload["档案"]["summary"] = row.get("summary") or ""
    if _has_lifecycle():
        payload["档案"]["superseded_by"] = row.get("superseded_by")
        payload["档案"]["superseded_at"] = str(row.get("superseded_at") or "") or None
        payload["档案"]["file_missing_at"] = str(row.get("file_missing_at") or "") or None

    try:
        from app.rag.vectorstore import get_vectorstore

        found = get_vectorstore().get(
            where={"file_path": str(row.get("file_path") or "")}, include=[]
        )
        payload["切片数"] = len(found.get("ids") or [])
    except Exception as exc:
        payload["切片数"] = f"读取失败：{exc}"

    if payload.get("切片数") == 0:
        payload["提示"] = "向量库里没有这份文件的片段，学生端检索不到它，需要重新灌库"
    return json.dumps(payload, ensure_ascii=False, default=str)


@tool
def admin_search_knowledge(
    query: str,
    doc_type: str = "",
    college: str = "",
    year: int = 0,
    k: int = 5,
) -> str:
    """在知识库里做语义检索，返回最相关的原文片段。管理端默认包含已废止和原件缺失的文件。

    只在**不知道是哪份文件**、必须按内容找时用它，例如
    「哪些通知提到助学金额度」「关于转专业笔试的原文怎么写的」。
    如果已经知道类型/年份/学院，先用 admin_list_documents，更快更准。

    Args:
        query: 自然语言描述，例如「国家奖学金金额」。
        doc_type: 文件类型过滤，不确定传空字符串。
        college: 学院过滤，不确定传空字符串。
        year: 年份过滤，不确定传 0。
        k: 返回多少个片段，默认 5。
    """
    try:
        result = search_documents_admin(
            query,
            k=k,
            doc_type=doc_type or None,
            college=college or None,
            year=year or None,
        )
    except Exception as exc:
        return json.dumps({"错误": f"检索失败：{exc}"}, ensure_ascii=False)

    return json.dumps(
        {
            "查询": query,
            "命中文件数": len(result["documents"]),
            "文件列表": result["documents"],
            "是否放宽了过滤条件": result["fallback"],
            "最高相关度": result.get("max_score"),
            "提示": result.get("notice") or None,
            "原文片段": format_context(result["hits"], notice=result.get("notice") or ""),
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def admin_propose_metadata_change(
    doc_id: int,
    title: str = "",
    doc_type: str = "",
    college: str = "",
    department: str = "",
    year: int = 0,
) -> str:
    """提出一份文件的元数据修改建议 —— **只生成待确认的变更单，不会真的改**。

    管理员要改文件的标题、分类、归属学院、发布部门、年份时用它。
    它返回「原值 → 新值」的 diff 和一个 change_id，管理员必须在管理端
    核对并点击确认之后才会真正生效。所以你可以放心地提，但不要声称
    「已经改好了」——你只能提出，执行权在人手里。

    改不了的东西：正文内容、文件位置（file_path）。这两类要另外的流程。

    Args:
        doc_id: 要修改的文件 id，必须先用 admin_list_documents /
            admin_search_knowledge 拿到的真实 id，不能猜。
        title: 新的标题。不改就传空字符串。
        doc_type: 新的文件类型，如「奖学金」「考试」。不改就传空字符串。
        college: 新的适用学院，如「计算机学院」。不改就传空字符串。
        department: 新的发布部门，如「教务处」。不改就传空字符串。
        year: 新的年份，如 2024。不改就传 0。
    """
    changes: dict[str, Any] = {}
    if title.strip():
        changes["title"] = title.strip()
    if doc_type.strip():
        changes["doc_type"] = doc_type.strip()
    if college.strip():
        changes["college"] = college.strip()
    if department.strip():
        changes["department"] = department.strip()
    if year:
        changes["year"] = int(year)

    if not changes:
        return json.dumps(
            {"错误": "没有提供任何要修改的字段", "可改字段": list(EDITABLE_FIELDS)},
            ensure_ascii=False,
        )

    try:
        result = create_change(int(doc_id), changes)
    except Exception as exc:
        return json.dumps({"错误": f"生成变更单失败：{exc}"}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False, default=str)


@tool
def admin_propose_move(doc_id: int, new_filename: str) -> str:
    """给一份文件**改名**（同目录内），只生成待确认的变更单，不会真的动磁盘。

    管理员说「把这份文件的名字改成 xxx」「文件名不规范，统一一下」时用它。
    只改文件名，不改正文。后缀必须和原文件一致（改后缀等于换解析器，不是改名）；
    不允许换目录 —— 知识库只扫 data/raw 一层，挪进子目录等于让它从库里消失。

    同样只是**提出**，需要管理员在管理端确认后才会真的改。

    Args:
        doc_id: 文件 id，必须先用 admin_list_documents 拿到的真实 id。
        new_filename: 新的文件名（含后缀），例如「2025年国家奖学金评审通知.md」。
            只传文件名，不要带目录。
    """
    from app.admin_change import create_move_change

    try:
        result = create_move_change(int(doc_id), new_filename)
    except Exception as exc:
        return json.dumps({"错误": f"生成改名单失败：{exc}"}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False, default=str)


def _session_id(config: RunnableConfig | None) -> str | None:
    """从 LangGraph 的 config 里取本次请求的会话 id。

    为什么不给工具加一个 session_id 参数让模型填：模型不该知道、也不该决定
    自己读哪个会话的目录，那等于把越权入口开给提示词。config 是框架层的
    请求上下文，模型看不到它，注入最安全。
    """
    try:
        return ((config or {}).get("configurable") or {}).get("session_id")
    except Exception:
        return None


@tool
def admin_list_attachments(config: RunnableConfig) -> str:
    """列出本次对话里管理员上传的附件（文件名、字数、标题）。

    用户说「这份文件」「我传的那个」时先用它确认到底是哪几份附件。
    附件只属于当前会话，不参与知识库检索 —— 它们还没入库。
    """
    from app.chat_attachment import list_attachments, supported_suffixes

    items = list_attachments(_session_id(config))
    if not items:
        return json.dumps(
            {
                "附件数": 0,
                "说明": "本次对话还没有上传附件",
                "支持格式": supported_suffixes(),
            },
            ensure_ascii=False,
        )
    return json.dumps({"附件数": len(items), "附件": items}, ensure_ascii=False, default=str)


@tool
def admin_read_attachment(
    name: str,
    offset: int = 0,
    max_chars: int = 6000,
    config: RunnableConfig = None,
) -> str:
    """读取一个附件解析出来的正文。**附件正文不会自动进入你的上下文**，必须调这个工具才看得到。

    附件很长时用 offset 分段读（返回里会告诉你下一次的 offset）。
    要通读全文，就按返回的 offset 继续调，直到「还有更多」为 false。

    Args:
        name: 附件文件名，来自 admin_list_attachments 或用户消息里的附件列表。
        offset: 从第几个字符开始读，默认 0。
        max_chars: 本次读多少字，默认 6000，最大 20000。
    """
    from app.chat_attachment import read_attachment

    try:
        result = read_attachment(
            name, offset=offset, max_chars=max_chars, session_id=_session_id(config)
        )
    except Exception as exc:
        return json.dumps({"错误": f"读取附件失败：{exc}"}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False, default=str)


ADMIN_TOOLS = [
    admin_knowledge_overview,
    admin_list_documents,
    admin_get_document,
    admin_search_knowledge,
    admin_propose_metadata_change,
    admin_propose_move,
    admin_list_attachments,
    admin_read_attachment,
]
