"""向量库：Embedding + Chroma。

Embedding 用阿里云 DashScope 的 text-embedding-v4（1024 维）。
为什么不用 DeepSeek 做 embedding？—— DeepSeek 目前没有开放 embedding 接口，
对话和向量由不同供应商提供，这在工程上很常见（「模型编排」的一部分）。
"""

from __future__ import annotations

from typing import Any, Iterable

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

from app import config
from app.logging_config import get_logger

log = get_logger(__name__)

_embeddings: DashScopeEmbeddings | None = None
_store: Chroma | None = None


def get_embeddings() -> DashScopeEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = DashScopeEmbeddings(model=config.EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore() -> Chroma:
    """取向量库单例。

    两种模式（见 config.CHROMA_MODE）：

    - embedded（默认）：直接开 var/chroma 目录。快，但索引是**进程内快照**，
      别的进程写进去的数据本进程 query 看不到 —— 所以只能单 worker。
    - server：连独立 Chroma 服务，客户端不持有索引，多 worker 共享同一份数据。
      代价是多一个常驻进程，收益是「入库后不用重启 web 服务」。
    """
    global _store
    if _store is not None:
        return _store

    if config.chroma_is_server():
        import chromadb

        client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
        # 传 client 时 langchain 内部会 get_collection，集合不存在会直接抛错，
        # 所以先确保它在（幂等，已存在时什么都不改）。
        client.get_or_create_collection(config.COLLECTION_NAME)
        _store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=get_embeddings(),
            client=client,
        )
        log.info("向量库：server 模式 %s:%s", config.CHROMA_HOST, config.CHROMA_PORT)
        return _store

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
    log.info("向量库：embedded 模式 %s", config.CHROMA_DIR)
    return _store


def reset_vectorstore() -> None:
    """清空集合，重新灌库时用。

    server 模式下 `reset_collection` 需要服务端开 `--allow-reset`，
    没开就只能删除整个集合再重建 —— 两条路都试一遍。
    """
    global _store
    store = get_vectorstore()
    try:
        store.reset_collection()
        return
    except Exception as exc:
        log.warning("reset_collection 失败（%s），改为删除集合", exc)

    try:
        store.delete_collection()
    except Exception as exc:
        raise RuntimeError(
            "无法清空向量库：reset 与 delete 都失败。"
            "server 模式请确认 chroma 服务启动时带了 --allow-reset"
        ) from exc
    _store = None


def add_documents(chunks: list[Document], batch_size: int = 10, verbose: bool = True) -> int:
    """分批写入，避免单次请求 embedding 文本过多被接口拒绝。

    verbose 只控制要不要往控制台打印进度；批量灌库时想看进度，
    上传接口里则交给日志，不往 HTTP 响应里塞噪音。
    """
    store = get_vectorstore()
    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        store.add_documents(batch)
        total += len(batch)
        if verbose:
            print(f"      已写入 {total}/{len(chunks)} 个 chunk")
    log.info("向量库写入 %d 个 chunk", total)
    return total


def delete_by_file_path(file_path: str) -> int:
    """删掉某个文件的全部 chunk，返回删除条数。

    这是「增量写入」能做到幂等的关键：同一份文件重复处理多少次，
    库里都只有一份 —— 先把它的旧 chunk 删干净再写新的，不会越积越多。

    注意必须**按 metadata 过滤**而不是按 id 删：一份文件对应几十个
    chunk，它们各自的 id 是向量库内部生成的，业务侧并不掌握。
    """
    store = get_vectorstore()
    try:
        found = store.get(where={"file_path": file_path})
        ids = list(found.get("ids") or [])
    except Exception as exc:
        log.warning("按路径查询旧 chunk 失败 %s：%s", file_path, exc)
        return 0
    if ids:
        store.delete(ids=ids)
        log.info("清理旧 chunk %d 个（%s）", len(ids), file_path)
    return len(ids)


def delete_many(file_paths: Iterable[str]) -> int:
    """批量按路径清理，**一次删除调用**搞定。

    原来是逐份循环调用 delete_by_file_path。改成一次 `$in` 不是为了快，
    是为了别把库写坏：实测（`var/_exp_chroma_churn2.py`，805 条 / 1024 维）
    一百多次零碎删除之后，chroma 的 HNSW 索引会变成读不出来的状态，
    换进程打开直接 `Error loading hnsw index`。老链路只是"出问题退回去"的开关，
    而退回去的那一刻往往正是要重灌全库的时候 —— 恰恰最容易踩这个坑。
    """
    paths = [str(p) for p in dict.fromkeys(file_paths) if p]
    if not paths:
        return 0
    store = get_vectorstore()
    try:
        found = store.get(where={"file_path": {"$in": paths}})
        ids = list(found.get("ids") or [])
    except Exception as exc:
        log.warning("按路径批量查询旧 chunk 失败：%s", exc)
        return 0
    if ids:
        store.delete(ids=ids)
        log.info("批量清理旧 chunk %d 个（%d 份文件）", len(ids), len(paths))
    return len(ids)


def update_metadata_by_file_path(file_path: str, changes: dict[str, Any]) -> int:
    """只改某个文件全部 chunk 的 metadata，返回受影响条数。

    为什么必须走底层 `_collection.update` 而不是官方的
    `update_document` / `update_documents`：后两者接收的是 Document
    （含 page_content），会连正文一起重写，从而**重新计算 embedding** ——
    也就是重新花钱。而「废止一份文件」改变的只有 status 这一个字段，
    正文一个字都没动，没有任何理由重算向量。

    注意 update 是**整份 metadata 替换**而不是合并，所以这里必须先把旧
    metadata 读出来，改完再整体写回。只传 changes 会把其他字段抹掉。
    """
    store = get_vectorstore()
    try:
        found = store.get(where={"file_path": file_path}, include=["metadatas"])
    except Exception as exc:
        log.warning("按路径查询 chunk 失败 %s：%s", file_path, exc)
        return 0

    ids = list(found.get("ids") or [])
    if not ids:
        return 0
    metadatas = list(found.get("metadatas") or [])
    merged = [dict(meta or {}, **changes) for meta in metadatas]
    # 条数对不上说明返回结构异常，宁可不动，也不要写坏 metadata
    if len(merged) != len(ids):
        log.warning("chunk 的 id 与 metadata 数量不一致（%d vs %d），已放弃更新 %s",
                    len(ids), len(merged), file_path)
        return 0

    try:
        store._collection.update(ids=ids, metadatas=merged)
    except Exception as exc:
        log.warning("更新 chunk metadata 失败 %s：%s", file_path, exc)
        return 0
    log.info("更新 chunk metadata %d 个（%s）：%s", len(ids), file_path, changes)
    return len(ids)


def list_file_paths(page_size: int = 1000, max_chunks: int = 500_000) -> set[str]:
    """列出向量库里所有 chunk 归属的文件路径（已去重）。

    对账要用：只从元数据表这一侧查，只能查出「表里有、库里没有」；
    反过来「库里有、表里没有」的残留 chunk 是查不出来的 —— 而它恰恰是
    「元数据没写成、向量却写进去了」这类半途失败留下的痕迹。

    分页读取。一次把整个集合的 metadata 拉进内存，在几万个 chunk 时就是
    几十 MB，没必要。max_chunks 是防呆上限：真到了那个量级，
    该换向量库了（Chroma 是嵌入式单机方案），而不是继续在这里硬扛。
    """
    store = get_vectorstore()
    paths: set[str] = set()
    offset = 0
    while offset < max_chunks:
        try:
            found = store.get(limit=page_size, offset=offset, include=["metadatas"])
        except Exception as exc:
            log.warning("分页读取向量库失败（offset=%d）：%s", offset, exc)
            break
        metadatas = list(found.get("metadatas") or [])
        if not metadatas:
            break
        for meta in metadatas:
            value = (meta or {}).get("file_path")
            if value:
                paths.add(str(value))
        offset += len(metadatas)
        if len(metadatas) < page_size:
            break
    return paths


def count() -> int:
    return get_vectorstore()._collection.count()
