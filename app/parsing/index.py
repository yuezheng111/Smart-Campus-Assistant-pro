"""Chroma 适配：把 Chunk 写进向量库（阶段五）。

不直接用 `app/rag/vectorstore.py` 的原因
----------------------------------------
那一层是围绕「一份文件 = 一堆 LangChain Document」设计的：id 交给 Chroma 生成
UUID、按 file_path 过滤、没有 document_id 的概念。阶段五要的是另一套约束：

    · **稳定的 chunk_id**（`<document_id>-c0003`），而不是随机 UUID ——
      否则同一份文件重跑一次就是几十条新记录，还查不出是哪次写的；
    · **按 document_id 查/删**，而不是按路径字符串 —— 路径一改就断链；
    · **能写进一个独立集合**，生产集合一个字节都不动。

所以这里直接对 chromadb 编程，而不是复用 LangChain 的 Chroma 包装。
两套并存：生产读写仍走 `app/rag/vectorstore.py`，新链路先写 staging。

三条硬约束
----------
1. **禁止写生产集合**：`collection_name` 等于 `config.COLLECTION_NAME` 时直接抛错，
   而不是"提醒一下"。这个检查放在构造函数里，写不进去比写错了强。
2. **先删后写**：写之前按 document_id 清掉旧 chunk。chunk_id 稳定时其实
   upsert 也能覆盖，但**文件变短**时旧的高序号 chunk 不会被覆盖 —— 它们会变成
   幽灵片段，检索时冒出来，点开原文却找不到对应段落。
3. **只存标量 metadata**：嵌套结构进不了 Chroma，转换在 `Chunk.to_chroma_metadata()`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import chromadb
from chromadb.api.models.Collection import Collection

from app import config
from app.logging_config import get_logger
from app.parsing.chunker import Chunk

log = get_logger(__name__)

#: 新链路的默认集合。带 staging 后缀，一眼能看出「这不是生产库」。
DEFAULT_COLLECTION = "school_documents_staging"

#: 一次 add 送多少条去算向量。DashScope 的批量 embedding 单次上限是 25 条，
#: 取 10 和现有 `add_documents` 保持一致 —— 批量太大会整批失败重试，太小则慢。
DEFAULT_BATCH_SIZE = 10


@dataclass
class IndexResult:
    """一次写入的结果。失败不抛异常，让上层能记录「写了几条、错在哪」。"""

    document_id: str
    written: int = 0
    deleted: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "written": self.written,
            "deleted": self.deleted,
            "skipped": self.skipped,
            "errors": list(self.errors),
        }


class LangChainEmbeddingFunction:
    """把 LangChain 的 `Embeddings` 适配成 chromadb 的 EmbeddingFunction。

    为什么不让 chromadb 用它自带的默认 embedding：那会去下载一个 ONNX 模型，
    而且和现有 1024 维的 text-embedding-v4 不是一个向量空间 —— 同一个集合里
    混两种向量，检索结果会毫无意义地漂。

    text-embedding-v4 对「文档」和「查询」用同一套编码（`text_type` 只是提示），
    所以 `embed_query` 直接复用 `embed_documents` 的实现，不需要再分叉。

    **刻意不注册到 chroma 的全局注册表**（`register_embedding_function`）。
    试过一次，副作用很隐蔽：注册之后，别的进程打开集合时能凭名字把这个函数重建出来，
    于是 LangChain 的 `similarity_search_with_relevance_scores` 从"抛异常"变成"可用"，
    检索返回的 `score` 含义当场从「距离」翻成「1-距离」（实测同一问 0.1905 ↔ 0.8095），
    下游的 `low_confidence` 闸门和前端展示跟着变，而这跟入库一点关系都没有。
    注册带来的唯一好处是"临时脚本不传 embedding 函数也能查"，不值得。
    """

    def __init__(self, embeddings: Any = None, model_name: str = "") -> None:
        self._embeddings = embeddings
        self._model = model_name or getattr(embeddings, "model", "") or "unknown"

    @property
    def embeddings(self) -> Any:
        """真正的向量函数。没传就按 config 现造一个 —— 延迟到第一次调用。

        chromadb 建集合时会用 `build_from_config` 重新造一个函数对象做校验，
        如果那里就把 DashScope 客户端建起来，等于每次开集合都多一个网络相关对象。
        所以这里把"用哪个模型"和"什么时候建客户端"分开。
        """
        if self._embeddings is None:
            from app.rag.vectorstore import get_embeddings

            self._embeddings = get_embeddings()
            self._model = getattr(self._embeddings, "model", "") or self._model
        return self._embeddings

    def __call__(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002
        vectors = self.embeddings.embed_documents(list(input))
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        """集合会把这个名字记下来，换模型时能一眼看出向量空间变了。

        必须是静态方法：chromadb 注册时是拿**类**在调 `cls.name()`，
        写成实例方法会报 `name() missing 1 required positional argument`。
        因此这个名字只能标识"用哪个适配器"，具体模型放在 get_config 里。
        """
        return "campus-agent-dashscope"

    def get_config(self) -> dict[str, Any]:
        return {"model": self._model}

    @staticmethod
    def build_from_config(config_dict: dict[str, Any]) -> "LangChainEmbeddingFunction":
        """从配置重建。只在"另一个进程读同一个集合"时才会被调用。

        这里**不建客户端**（见 `embeddings` 属性）：建集合的路径上会被调用一次，
        在那里连 DashScope 纯属浪费。顺手记下模型名，将来对不上了一眼能看出来。
        """
        return LangChainEmbeddingFunction(None, str(config_dict.get("model") or ""))

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def is_legacy(self) -> bool:
        """告诉 chromadb「我不是旧式函数」，别再走已废弃的配置序列化路径。

        旧路径会尝试把函数本身 pickle 进集合配置 —— 我们的函数里握着一个
        DashScope 客户端，pickle 出来既没用也不安全。
        """
        return False


class ChromaChunkIndex:
    """一个集合的写入 / 删除 / 查询封装。

    用法：

        index = ChromaChunkIndex()                 # staging
        index.upsert_document(chunks)              # 幂等写入
        index.chunk_ids_for(document_id)           # 追溯这份文件的全部片段
        index.delete_document(document_id)         # 下架
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        *,
        persist_dir: Path | str | None = None,
        embeddings: Any = None,
        client: Any = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        allow_production: bool = False,
    ) -> None:
        # 这一条是保险丝，不是提示。生产集合一旦被新链路写脏，
        # 学生的检索结果会混进没校验过的 chunk，回滚很麻烦。
        #
        # 阶段六切生产时必须显式传 allow_production=True —— 也就是说，
        # 「写生产」只能是一个**有人主动按下去**的动作，
        # 不能是路径上顺手走到的默认值。
        if collection_name == config.COLLECTION_NAME and not allow_production:
            raise ValueError(
                f"拒绝写入生产集合 {config.COLLECTION_NAME!r}。"
                f"默认只允许写独立集合（{DEFAULT_COLLECTION!r}）；"
                "确实要切生产，请显式传 allow_production=True。"
            )

        self.collection_name = collection_name
        self.batch_size = max(1, batch_size)
        self._persist_dir = Path(persist_dir) if persist_dir else config.CHROMA_DIR
        self._embeddings = embeddings
        self._client = client
        self._collection: Collection | None = None
        self._embedding_function_cache: LangChainEmbeddingFunction | None = None

    # ---- 基础设施 ----

    def _embedding_function(self) -> LangChainEmbeddingFunction:
        """向量函数单例。

        缓存它不是为了省一次构造，而是为了**只建一个 DashScope 客户端**：
        批量入库时会反复调用，每次都新建一个客户端等于每批都重新握手。
        """
        if self._embedding_function_cache is None:
            # 不再在这里抢着建客户端：真正写入时才需要它，
            # 而「打开一个已有集合看一眼」不该产生任何网络相关的副作用。
            self._embedding_function_cache = LangChainEmbeddingFunction(
                self._embeddings, getattr(config, "EMBEDDING_MODEL", "")
            )
        return self._embedding_function_cache

    def warmup(self) -> None:
        """把向量函数（含底下的 HTTP 客户端）建出来，**不发请求**。

        并发入库前在主线程调一次：否则 4 个 worker 会同时去初始化客户端。
        """
        self._embedding_function()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把文本算成向量，**不写库**。

        单独拆出来是为了并发：向量化是最慢的一步（公网往返），可以四个线程一起算；
        而写 Chroma（SQLite 单写者）必须串行。拆开之后，慢的部分并行、
        危险的部分串行，各归各位。
        """
        if not texts:
            return []
        return self._embedding_function()(list(texts))

    @property
    def collection(self) -> Collection:
        if self._collection is not None:
            return self._collection

        if self._client is None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function(),
            # 显式指定 cosine：Chroma 默认是 l2，而 text-embedding-v4 是
            # 归一化向量，两者在排序上等价但不完全一致，统一按余弦更稳。
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    # ---- 写 ----

    def upsert_document(
        self,
        chunks: Sequence[Chunk],
        *,
        replace: bool = True,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> IndexResult:
        """把一份文件的全部 chunk 写进集合。

        replace=True（默认）时先按 document_id 删除旧片段再写，这是
        「重复入库不会翻倍」和「文件变短不留下幽灵片段」的共同保证。

        embeddings 是**预先算好的向量**（见 `embed()`）。传了就用它，
        不再让 Chroma 内部去调向量函数 —— 这样并发流程才能"线程里算向量、
        主线程里按顺序写库"。
        """
        if not chunks:
            return IndexResult(document_id="", skipped=0, errors=["没有可写入的 chunk"])

        document_id = chunks[0].document_id
        result = IndexResult(document_id=document_id)

        mixed = {chunk.document_id for chunk in chunks}
        if len(mixed) > 1:
            result.errors.append(f"一次只能写一份文件，收到 {len(mixed)} 个 document_id")
            return result

        if embeddings is not None and len(embeddings) != len(chunks):
            result.errors.append(
                f"向量条数与 chunk 数不一致（{len(embeddings)} vs {len(chunks)}），已放弃写入"
            )
            return result

        if replace:
            result.deleted = self.delete_document(document_id)
            # 再按 file_path 清一遍：老链路（阶段五之前）写进库的 chunk **没有
            # document_id 字段**，它们的主键是 Chroma 随机 UUID。只按 document_id
            # 删的话，这些老片段会原地不动，和新片段一起被检索到 ——
            # 学生看到的「参考资料」里就有同一份文件的两个版本。
            result.deleted += self.delete_by_file_path(
                str(chunks[0].metadata.get("file_path") or "")
            )

        collection = self.collection
        for start in range(0, len(chunks), self.batch_size):
            batch = list(chunks[start:start + self.batch_size])
            try:
                payload: dict[str, Any] = {
                    "ids": [chunk.chunk_id for chunk in batch],
                    "documents": [chunk.text for chunk in batch],
                    "metadatas": [chunk.to_chroma_metadata() for chunk in batch],
                }
                if embeddings is not None:
                    payload["embeddings"] = [
                        list(vector) for vector in embeddings[start:start + len(batch)]
                    ]
                # 用 upsert 而不是 add：chunk_id 稳定，同样的 id 重写本来就该是
                # "覆盖"而不是"撞车报错"。批量删除在前（见 _clear），这里不做兜底
                # 删除 —— 逐份删正是把索引写坏的那个操作。
                collection.upsert(**payload)
            except Exception as exc:  # noqa: BLE001 - 单批失败不能中断整份文件
                message = (
                    f"第 {start // self.batch_size + 1} 批（{len(batch)} 条）写入失败："
                    f"{type(exc).__name__}: {exc}"
                )
                log.warning("向量写入失败 | %s | %s", document_id, message)
                result.errors.append(message)
                result.skipped += len(batch)
                continue
            result.written += len(batch)

        log.info(
            "写入集合 %s | %s | 新增 %d 条，清理旧片段 %d 条，失败 %d 条",
            self.collection_name, document_id, result.written, result.deleted, result.skipped,
        )
        return result

    # ---- 删 ----

    def delete_document(self, document_id: str) -> int:
        """删掉一份文件的全部 chunk，返回删除条数。"""
        try:
            existing = self.chunk_ids_for(document_id)
            if not existing:
                return 0
            self.collection.delete(where={"document_id": document_id})
        except Exception as exc:  # noqa: BLE001
            log.warning("删除 chunk 失败 | %s：%s", document_id, exc)
            return 0
        return len(existing)

    def delete_documents(self, document_ids: Iterable[str]) -> int:
        return sum(self.delete_document(document_id) for document_id in dict.fromkeys(document_ids))

    def delete_documents_bulk(
        self, document_ids: Iterable[str], file_paths: Iterable[str] = ()
    ) -> int:
        """一次删掉一批文件的旧片段，**总共只发两次删除**。

        这个方法存在的理由不是性能，是正确性。实测（`var/_exp_chroma_churn2.py`，
        805 条、1024 维、115 份文件）：

            · 逐份「删一次 → 写一批」交替      → 索引写坏，新进程报 Error loading hnsw index
            · 先逐份删 115 次、再统一写（分相） → 同样写坏
            · 一次删空、再统一写              → 正常
            · 删集合重写                     → 正常

        也就是说，**删除操作的次数**才是变量（115 次 vs 1 次），不是删除与写入
        是否交替、也不是收尾有没有 flush（关之前等 10 秒照样坏）。
        chroma 1.5 的每次 delete 都会在段里留一条变更记录，上百条零碎删除之后，
        HNSW 索引文件就不再有可读的持久化形态，读的时候想现场重建也失败。

        所以把删除收到整批一次：`$in` 一把删完，写的时候用 upsert。
        单份上传时它退化成"删两次"，和原来一样，本来也不会出问题。
        """
        ids = [str(i) for i in dict.fromkeys(document_ids or []) if i]
        paths = [str(p) for p in dict.fromkeys(file_paths or []) if p]
        total = 0
        for where in (
            {"document_id": {"$in": ids}} if ids else None,
            {"file_path": {"$in": paths}} if paths else None,
        ):
            if where is None:
                continue
            try:
                found = self.collection.get(where=where, include=[])
                chunk_ids = list(found.get("ids") or [])
                if not chunk_ids:
                    continue
                self.collection.delete(ids=chunk_ids)
                total += len(chunk_ids)
            except Exception as exc:  # noqa: BLE001
                log.warning("批量删除旧片段失败 %s：%s", list(where)[0], exc)
        if total:
            log.info("批量清理旧片段 %d 个（%d 份文件）", total, len(ids))
        return total

    def delete_by_file_path(self, file_path: str) -> int:
        """按文件路径删掉 chunk，返回条数。

        存在的理由只有一个：**兼容老链路留下的数据**。老链路写的 chunk 没有
        `document_id` 字段，只有 `file_path`。迁移期间两种数据混在一个集合里，
        只按 document_id 清理会留下幽灵片段。
        """
        if not file_path:
            return 0
        try:
            found = self.collection.get(where={"file_path": file_path}, include=[])
            ids = list(found.get("ids") or [])
            if not ids:
                return 0
            self.collection.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001
            log.warning("按路径删除 chunk 失败 %s：%s", file_path, exc)
            return 0
        log.info("按路径清理旧 chunk %d 个（%s）", len(ids), file_path)
        return len(ids)

    def update_metadata_for_document(
        self, document_id: str, changes: dict[str, Any]
    ) -> int:
        """只改某份文件全部 chunk 的 metadata，**不重算向量**。

        废止、标记原件缺失这类操作改的都是状态字段，正文一个字没动，
        没有任何理由重新花钱算一遍 embedding。

        必须走底层 `collection.update` 而不是"删了重写"：后者会重算向量。
        注意 update 是**整份 metadata 替换**而非合并，所以要先读旧值再整体写回。
        """
        return self._patch({"document_id": document_id}, changes, f"document_id={document_id}")

    def update_metadata_by_file_path(self, file_path: str, changes: dict[str, Any]) -> int:
        """按路径改 metadata。和上面那个的区别：能命中**老链路**写的 chunk。

        「原件缺失」这类标记必须两种数据都覆盖到，否则迁移期间有的片段退出检索、
        有的还在，学生搜到的结果会时有时无。
        """
        if not file_path:
            return 0
        return self._patch({"file_path": file_path}, changes, file_path)

    def _patch(self, where: dict[str, Any], changes: dict[str, Any], label: str) -> int:
        try:
            found = self.collection.get(where=where, include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            log.warning("按 %s 查询 chunk 失败：%s", label, exc)
            return 0

        ids = list(found.get("ids") or [])
        metadatas = list(found.get("metadatas") or [])
        if not ids:
            return 0
        if len(metadatas) != len(ids):
            log.warning(
                "chunk 的 id 与 metadata 数量不一致（%d vs %d），已放弃更新 %s",
                len(ids), len(metadatas), label,
            )
            return 0

        merged = [dict(meta or {}, **changes) for meta in metadatas]
        try:
            self.collection.update(ids=ids, metadatas=merged)
        except Exception as exc:  # noqa: BLE001
            log.warning("更新 chunk metadata 失败 %s：%s", label, exc)
            return 0
        log.info("更新 chunk metadata %d 个（%s）：%s", len(ids), label, changes)
        return len(ids)

    def drop_collection(self) -> None:
        """整个集合删掉。只在「换 embedding 模型、要重灌」时用。"""
        if self._client is None:
            # 记住这个客户端，交给 close() 关 —— 顺带建的客户端如果没人关，
            # 后面那次重建就会踩到和 close() 注释里一样的坑。
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        client = self._client
        try:
            client.delete_collection(self.collection_name)
        except Exception as exc:  # noqa: BLE001
            log.warning("删除集合 %s 失败：%s", self.collection_name, exc)
        self._collection = None

    def close(self) -> None:
        """关掉底层客户端，**把还没合并的写入刷进磁盘**。

        为什么必须显式关（这一条是踩出来的，不是讲究）：
        chromadb 1.5 的写入先落 SQLite 的变更日志，再由后台 compactor
        合并进 HNSW 索引文件。进程退出时如果没人调 `close()`，这次合并
        可能还没做完，磁盘上就留下「元数据说有 485 条、索引文件里还是旧的
        204 条」的半截状态 —— 下一个进程打开时连 `get()` 都报
        `Error loading hnsw index`，整个知识库对学生**完全不可检索**。

        实测：115 份文件的全量重建跑完后，生产集合在另一个进程里一条都读不出来。

        `close()` 是引用计数的：同一进程里还有别的客户端用着同一个存储目录
        （web 服务的 `vectorstore` 就是）时，这里不会把那个一起关掉。
        """
        client, self._client = self._client, None
        self._collection = None
        if client is None:
            return
        close = getattr(client, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception as exc:  # noqa: BLE001 - 关不掉也不该让入库流程报错
            log.warning("关闭 Chroma 客户端失败：%s", exc)

    # ---- 读 ----

    def chunk_ids_for(self, document_id: str, limit: int = 10_000) -> list[str]:
        """一份文件在库里的全部 chunk id，按序号排好。

        「追踪单个文件对应的全部 Chunk」就是靠它 —— 入库后拿这份清单和切分结果
        对一遍，就能发现「切出来了但没写进去」的漏网之鱼。
        """
        try:
            found = self.collection.get(
                where={"document_id": document_id}, limit=limit, include=[]
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("查询 chunk 失败 | %s：%s", document_id, exc)
            return []
        ids = list(found.get("ids") or [])
        return sorted(ids)

    def document_ids(self, limit: int = 100_000) -> set[str]:
        """集合里出现过的所有 document_id。对账用。"""
        try:
            found = self.collection.get(limit=limit, include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            log.warning("列出 document_id 失败：%s", exc)
            return set()
        return {
            str(metadata.get("document_id"))
            for metadata in (found.get("metadatas") or [])
            if metadata and metadata.get("document_id")
        }

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """按 chunk_id 取回一条，含正文和元数据。"""
        try:
            found = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001
            log.warning("读取 chunk 失败 | %s：%s", chunk_id, exc)
            return None
        ids = list(found.get("ids") or [])
        if not ids:
            return None
        documents = list(found.get("documents") or [])
        metadatas = list(found.get("metadatas") or [])
        return {
            "chunk_id": ids[0],
            "text": documents[0] if documents else "",
            "metadata": metadatas[0] if metadatas else {},
        }

    def query(
        self, text: str, top_k: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """语义检索。阶段五用它验证「新切分的 chunk 确实检索得到」。"""
        try:
            found = self.collection.query(
                query_texts=[text],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("检索失败：%s", exc)
            return []

        def first(key: str) -> list[Any]:
            value = found.get(key) or []
            return list(value[0]) if value else []

        ids = first("ids")
        documents = first("documents")
        metadatas = first("metadatas")
        distances = first("distances")
        results: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else None
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": documents[index] if index < len(documents) else "",
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    # 余弦距离 → 相似度，和现有 retriever 的口径保持一致
                    "score": round(1 - float(distance), 4) if distance is not None else None,
                }
            )
        return results

    def stats(self) -> dict[str, Any]:
        return {
            "collection": self.collection_name,
            "chunk数": self.collection.count(),
            "文件数": len(self.document_ids()),
        }
