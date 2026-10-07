"""检索器：向量检索 + 元数据过滤。

真实场景里，学生的问题往往自带约束条件：
    「2024 年 计算机学院 的 奖学金 通知」
纯向量检索很容易被「2025 年」或「其他学院」的相似文档污染。
所以这里采取 **两段式**：
    1) 用元数据先做硬过滤（缩小候选范围，保证不跑偏）
    2) 再在候选里做向量语义排序（找到最相关的那几段）
"""

from __future__ import annotations

import re
from typing import Any

from app import config
from app.logging_config import get_logger
from app.rag.vectorstore import get_vectorstore

log = get_logger(__name__)

# 检索默认只看「现行」文件。已废止的文件并没有被删掉 ——
# 它们只是不再进入默认候选集，学生明确问往年政策时仍然搜得到。
STATUS_ACTIVE = "active"


def _merge(*conditions: dict[str, Any] | None) -> dict[str, Any] | None:
    """把多个 where 条件合成一个。

    顺手把 `{"$and": [...]}` 拆开摊平：Chroma 的过滤条件嵌套几层是合法的，
    但 `{"$and": [{"$and": [a, b]}, c]}` 这种结构在日志里几乎读不懂，
    排查「为什么这份文件没被过滤掉」时会非常难受。
    """
    items: list[dict[str, Any]] = []
    for condition in conditions:
        if not condition:
            continue
        if len(condition) == 1 and "$and" in condition:
            items.extend(condition["$and"])
        else:
            items.append(condition)
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    return {"$and": items}


def _baseline_filter() -> dict[str, Any]:
    """任何检索都必须成立的「地基」条件。

    两条，各自挡掉一种「不该出现在默认结果里」的文件：

        status = active     这份文件已不再现行（被新版本替代）
        file_missing = 0    这份文件的原件已从磁盘上消失

    第二条为什么也要挡：原件没了，学生在来源卡片上点「查看原文」会直接 404。
    但它**没有被删除** —— 片段和正文都还在库里，问历史时仍然查得到。
    """
    return {
        "$and": [
            {"status": {"$eq": STATUS_ACTIVE}},
            {"file_missing": {"$eq": 0}},
        ]
    }


def build_filter(
    doc_type: str | None = None,
    college: str | None = None,
    year: int | None = None,
    department: str | None = None,
) -> dict[str, Any] | None:
    """把过滤条件拼成 Chroma 的 where 语法。

    注意 college 的处理：写「全校」的文件对任何学院都适用，
    所以用 $in 把目标学院和「全校」一起放进候选，而不是精确等于。
    """
    conditions: list[dict[str, Any]] = []

    if doc_type:
        conditions.append({"doc_type": {"$eq": doc_type}})
    if department:
        conditions.append({"department": {"$eq": department}})
    if college:
        conditions.append({"college": {"$in": [college, "全校"]}})
    if year:
        conditions.append({"year": {"$eq": int(year)}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _to_hit(doc, score: float) -> dict[str, Any]:
    meta = doc.metadata or {}
    return {
        "title": meta.get("title", ""),
        "doc_no": meta.get("doc_no", ""),
        "department": meta.get("department", ""),
        "doc_type": meta.get("doc_type", ""),
        "college": meta.get("college", ""),
        "publish_date": meta.get("publish_date", ""),
        "year": meta.get("year", 0),
        "file_path": meta.get("file_path", ""),
        "url": meta.get("url", ""),
        "doc_family": meta.get("doc_family", ""),
        "status": meta.get("status", STATUS_ACTIVE),
        # 原件是否已从磁盘消失。默认检索本来就把它挡在外面了，带上是为了
        # 「问历史」这条路径 —— 前端拿到 1 就知道「这份点不开原件」，
        # 可以提前把「查看原文」置灰，而不是让用户点出一个 404。
        "file_missing": int(meta.get("file_missing") or 0),
        "chunk_index": meta.get("chunk_index", 0),
        "score": round(float(score), 4),
        "content": doc.page_content,
    }


def search_documents(
    query: str,
    k: int | None = None,
    doc_type: str | None = None,
    college: str | None = None,
    year: int | None = None,
    department: str | None = None,
    include_superseded: bool = False,
) -> dict[str, Any]:
    """检索学校文件，返回 {hits, documents, filter_used, fallback}。

    hits：chunk 级别（给 LLM 当上下文）
    documents：按标题去重后的文件级列表（给前端展示卡片）

    include_superseded=False（默认）只搜现行文件。学生明确问「往年政策」
    「去年的通知怎么规定的」时传 True，把已废止的文件一并纳入候选。
    """
    store = get_vectorstore()
    top_k = k or config.RETRIEVE_TOP_K

    # 状态过滤是所有检索的**地基**，业务条件只是可松可紧的上层。
    # 两者分开构造，才有办法在「放宽」时只放开上层。
    baseline_where = None if include_superseded else _baseline_filter()
    business_where = build_filter(doc_type, college, year, department)
    where = _merge(baseline_where, business_where)

    fallback = False
    try:
        pairs = _search_with_score(store, query, top_k, where)
        if not pairs and business_where:
            # 带业务条件查不到 → 放开业务条件重查一次，避免「一个条件卡死后端」。
            # 但状态过滤**必须保留**：否则这一「放宽」就会把已经废止的旧政策、
            # 或者原件已经被删掉的留档重新召回上来，而那正是它要防的事。
            fallback = True
            pairs = _search_with_score(store, query, top_k, baseline_where)
        if not pairs and not include_superseded and _library_lacks_state_fields(store):
            # 仍然为空，而且这次不是「真的没有」—— 是向量库还没按新结构重灌、
            # chunk 上根本没有 status / file_missing 字段。这是数据迁移问题，
            # 不该表现成「什么都搜不到」，所以降级为不按状态过滤，并留下日志。
            log.warning(
                "向量库中的 chunk 缺少 status / file_missing 字段，本次检索暂不按"
                "文件状态过滤；重新执行 python run_ingest.py 可修复。"
            )
            fallback = True
            pairs = _search_with_score(store, query, top_k, business_where)
    except Exception as exc:  # 过滤语法不被支持时也要能兜住
        fallback = True
        pairs = _search_with_score(store, query, top_k, baseline_where)
        if not pairs:
            raise exc

    hits = [_to_hit(doc, score) for doc, score in pairs]

    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in sorted(hits, key=lambda h: h["score"], reverse=True):
        key = hit["title"]
        if key in seen:
            continue
        seen.add(key)
        merged = {k_: v for k_, v in hit.items() if k_ != "content"}
        # 带上命中片段，前端卡片直接展示「为什么这份文件被召回」。
        # 去掉切分时加上的【标题】前缀，避免卡片里标题出现两次。
        snippet = str(hit["content"])
        if snippet.startswith("【"):
            snippet = snippet.split("】", 1)[-1]
        merged["snippet"] = re.sub(r"\s+", " ", snippet).strip()[:180]
        documents.append(merged)

    return {
        "hits": hits,
        "documents": documents,
        "filter_used": where,
        "fallback": fallback,
    }


def _library_lacks_state_fields(store) -> bool:
    """向量库里是否还存在「缺少状态字段」的 chunk。

    两种情况会这样：库是在加 status / file_missing 之前灌的、还没重灌；
    或者灌到一半中断了。抽成独立函数是为了让 search_documents 的逻辑
    读起来仍然是一句话。
    """
    try:
        found = store.get(limit=1, include=["metadatas"])
    except Exception:
        return False
    metadatas = found.get("metadatas") or []
    if not metadatas:
        return False
    meta = metadatas[0] or {}
    return "status" not in meta or "file_missing" not in meta


def _search_with_score(store, query: str, k: int, where: dict | None):
    try:
        return store.similarity_search_with_relevance_scores(query, k=k, filter=where)
    except Exception:
        return store.similarity_search_with_score(query, k=k, filter=where)


def format_context(
    hits: list[dict[str, Any]],
    limit_docs: int = 5,
    max_chars_per_doc: int = 1400,
) -> str:
    """把检索结果整理成给 LLM 看的上下文。

    注意这里是**按文件编号**（[资料1]、[资料2]），不是按 chunk 编号。
    同一份文件的多个 chunk 会合并到一个编号下 —— 因为：
      1. 前端展示的来源卡片是按文件去重的，编号一致用户才不会困惑；
      2. 模型引用 [资料2] 时，能明确对应到「第二份文件的原文」。
    """
    if not hits:
        return "（未检索到相关文件）"

    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for hit in hits:
        title = hit["title"]
        if title not in grouped:
            grouped[title] = {"meta": hit, "chunks": []}
            order.append(title)
        grouped[title]["chunks"].append(str(hit["content"]))

    blocks: list[str] = []
    for index, title in enumerate(order[:limit_docs], start=1):
        item = grouped[title]
        meta = item["meta"]
        body = "\n".join(item["chunks"])[:max_chars_per_doc]
        # 已废止 / 原件已消失的文件只在「学生主动问历史」时才会进上下文。
        # 标出来是为了让模型说清这是哪个版本，而不是把旧政策当成现行规定来答。
        if meta.get("status") == "superseded":
            flag = "｜状态：已废止（往年版本，非现行）"
        elif meta.get("file_missing"):
            flag = "｜状态：原始文件已不在库中，以下内容来自留档片段"
        else:
            flag = ""
        blocks.append(
            f"[资料{index}] 标题：{meta['title']}｜发布部门：{meta['department']}"
            f"｜类型：{meta['doc_type']}｜适用学院：{meta['college']}"
            f"｜发布日期：{meta['publish_date']}{flag}\n{body}"
        )
    return "\n\n".join(blocks)


if __name__ == "__main__":
    for question, filters in [
        ("国家奖学金评选条件", {"doc_type": "奖学金"}),
        ("转专业笔试考什么", {"college": "计算机学院"}),
        ("四六级什么时候报名", {"doc_type": "考试"}),
    ]:
        result = search_documents(question, **filters)
        print(f"\n=== {question}  (filter={filters}) ===")
        for d in result["documents"][:3]:
            print(f"  {d['score']:.3f}  {d['publish_date']}  {d['title']}")
        if not result["documents"]:
            print("  （无结果）")
