"""学校文件相关工具：RAG 检索 + 元数据查询。

这两个工具正好代表两种完全不同的取数方式：
    search_school_document  → 走向量库（非结构化，模糊语义）
    list_school_documents   → 走 MySQL（结构化，精确条件）
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from app import db
from app.rag.retriever import format_context, search_documents

# 数据库有没有生命周期字段（探测一次后缓存），见 _supports_lifecycle
_has_lifecycle_columns: bool | None = None


@tool
def search_school_document(
    query: str,
    doc_type: str = "",
    college: str = "",
    year: int = 0,
    include_superseded: bool = False,
) -> str:
    """在学校历年发布的通知、公告、规章制度中做语义检索，返回最相关的原文片段。

    当学生想「找某份文件」「问某个政策怎么规定」「某件事的办理流程」时使用这个工具。
    它返回的是文件正文片段，适合用来回答「内容是什么」类问题。

    默认只检索**现行**文件。如果学生明确在问往年政策（如「2023 年的奖学金
    怎么评的」「去年那份通知写了什么」），把 include_superseded 设为 True，
    才会把已废止的旧版本、以及原件已不在库中的留档一并纳入检索。

    Args:
        query: 学生的自然语言问题或关键词，例如「国家奖学金评选条件」。
        doc_type: 文件类型过滤。可选：奖学金、助学金、考试、竞赛、学籍、教学、
            规章制度、活动、后勤。不确定就传空字符串。
        college: 学院过滤，例如「计算机学院」。不确定就传空字符串。
        year: 年份过滤，例如 2024。不确定就传 0。
        include_superseded: 是否把已废止 / 原件已不在库中的旧文件也纳入检索，
            默认 False。
    """
    result = search_documents(
        query,
        doc_type=doc_type or None,
        college=college or None,
        year=year or None,
        include_superseded=include_superseded,
    )
    hits = result["hits"]

    payload = {
        "查询": query,
        "命中文件数": len(result["documents"]),
        "文件列表": result["documents"],
        "是否放宽了过滤条件": result["fallback"],
        "原文片段": format_context(hits),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _supports_lifecycle() -> bool:
    """数据库里有没有生命周期字段（status / file_missing_at）。

    与 app/ingest.py 里同样的探测：库可能比代码旧（没重跑 init_db.py）。
    缺列时退化成「不按状态过滤」，让学生端至少还能用，
    而不是整个文件列表接口直接 500。

    要求**两列都在**才算数：只查 status 的话，一个「有 status、没
    file_missing_at」的半吊子库会让下面拼出的 WHERE 撞上未知列。
    """
    global _has_lifecycle_columns
    if _has_lifecycle_columns is None:
        try:
            columns = {
                str(row.get("Field") or "")
                for row in db.fetch_all("SHOW COLUMNS FROM `document`")
            }
            _has_lifecycle_columns = {"status", "file_missing_at"} <= columns
        except Exception:
            _has_lifecycle_columns = False
    return _has_lifecycle_columns


@tool
def list_school_documents(
    doc_type: str = "",
    college: str = "",
    year: int = 0,
    department: str = "",
    keyword: str = "",
    limit: int = 10,
    include_superseded: bool = False,
) -> str:
    """按精确条件从数据库列出学校文件清单（只返回标题、部门、日期等元数据，不返回正文）。

    默认只列**现行**文件 —— 已经废止的旧版政策、以及原件已被移出库的文件
    都不会混进来。只有学生明确问「往年 / 去年 / 历史上」的规定时，
    才把 include_superseded 设为 True。

    适合回答「有哪些」「某年发过几份」这类统计性、列表性问题。
    如果要看文件的具体内容，请改用 search_school_document。

    Args:
        doc_type: 文件类型，例如「奖学金」「考试」「竞赛」。不确定传空字符串。
        college: 学院，例如「计算机学院」。不确定传空字符串。
        year: 年份，例如 2024。不确定传 0。
        department: 发布部门，例如「教务处」「学生工作处」。不确定传空字符串。
        keyword: 标题关键词，例如「四六级」。不确定传空字符串。
        limit: 最多返回多少条，默认 10。
        include_superseded: 是否把已废止 / 原件已不在库中的旧文件也列出来，
            默认 False。
    """
    conditions: list[tuple[str, str, object]] = []
    # 状态过滤与向量检索那边保持同一口径（见 app/rag/retriever.py 的
    # _baseline_filter）。否则会出现「列表里明明有这份文件，搜的时候却搜不到」
    # —— 这类不一致是最难排查的。
    #
    # 两条地基条件：这份文件现行，且它的原件还在磁盘上。后者同样重要 ——
    # 原件没了就没法「查看原文」，学生会点出一个 404。
    if not include_superseded and _supports_lifecycle():
        conditions.append(("status", "=", "active"))
        conditions.append(("file_missing_at", "is_null", None))
    if doc_type:
        conditions.append(("doc_type", "=", doc_type))
    if college:
        # 同样地，「全校」文件对所有学院都适用
        conditions.append(("college", "in", (college, "全校")))
    if year:
        conditions.append(("year", "=", int(year)))
    if department:
        conditions.append(("department", "=", department))
    if keyword:
        conditions.append(("title", "like", keyword))

    where, params = db.build_where(conditions)
    params["limit"] = max(1, min(int(limit), 50))
    sql = (
        "SELECT id, doc_no, title, department, doc_type, college, publish_date, year, "
        "url, summary "
        f"FROM document{where} ORDER BY publish_date DESC LIMIT :limit"
    )
    rows = db.fetch_all(sql, params)

    return json.dumps(
        {"条件": {k: v for k, v in params.items() if k != "limit"}, "数量": len(rows), "文件清单": rows},
        ensure_ascii=False,
        default=str,
    )


def _detail_columns() -> str:
    """详情查询要取的列。

    生命周期字段是后加进表里的，库比代码旧时带上它们会让整条查询报错，
    所以这里同样做一次探测再决定 —— 见 _supports_lifecycle。
    """
    base = (
        "id, doc_no, title, department, doc_type, college, publish_date, year, "
        "file_path, url, summary"
    )
    if _supports_lifecycle():
        return base + (
            ", status, doc_family, version, superseded_by, superseded_at, "
            "file_missing_at"
        )
    return base


@tool
def get_document_detail(doc_id: int) -> str:
    """按文件编号(id)获取一份学校文件的完整元数据和摘要。

    Args:
        doc_id: 文件在数据库中的 id，可从 list_school_documents 或
            search_school_document 的结果里拿到。
    """
    row = db.fetch_one(
        f"SELECT {_detail_columns()} FROM document WHERE id = :doc_id",
        {"doc_id": int(doc_id)},
    )
    if not row:
        return json.dumps({"错误": f"没有找到 id={doc_id} 的文件"}, ensure_ascii=False)
    return json.dumps(row, ensure_ascii=False, default=str)
