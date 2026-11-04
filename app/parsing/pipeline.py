"""入库流水线（阶段六）：一份文件从磁盘走到「可被检索」。

状态机
------
    PENDING → VALIDATING → PARSING → NORMALIZING → CHUNKING
            → EMBEDDING → INDEXING → PERSISTING → COMPLETED
    任一步失败 → FAILED（记下**在哪一步**失败）
    内容有损但可用（页里坏了几页、几张表没抽出来）→ PARTIAL
    不该入库（扫描件没有文本层、切不出内容）→ SKIPPED（带原因，不是静默丢弃）

**PERSISTING 是阶段六新增的一步。** 规格里的状态机只到 INDEXING，但写向量和
写元数据是两个独立的失败点：合并成一步的话，出问题时日志只会说"入库失败"，
你得翻三处才知道到底是向量没写进去（搜不到）还是元数据没写进去（列表里没有）。

为什么向量先写、元数据后写
--------------------------
元数据表里的那一行是「这份文件正式在库里」的凭证。先写向量、再写凭证，
中途挂掉的结果是「磁盘上有、表里没有」—— 对账工具里叫「未登记」，
管理员一眼能看见该补录。反过来先写凭证再写向量，挂掉会变成
「表里有、搜不到」，学生点进去发现搜不到还查不出原因。

三层防护，各管一件事
--------------------
    · **单文件隔离**：任何一份文件炸掉都只影响它自己（`_prepare` 里全包住）
    · **重试**：只重试"可能是网络抖动"的步骤（向量化），不重试确定性失败
      （文件损坏、格式不支持 —— 重试一百次也是同一个结果，只是白等）
    · **回滚**：重试仍失败时，把已写入的半份 chunk 删掉，不留半成品
"""

from __future__ import annotations

import datetime as dt
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from app.logging_config import get_logger
from app.parsing.chunker import Chunk, ChunkingOptions, chunk_document
from app.parsing.factory import SUPPORTED_TYPES, detect_file_type, parse_file
from app.parsing.base import ParseOptions, relative_path
from app.parsing.models import ParsedDocument, ParseStatus, document_id_for
from app.parsing.normalize import NormalizeOptions, normalize
# 侧车文件名的命名契约在 `app/rag/loader`。这里只引用那一个常量，
# 不复制一份字符串 —— 两处各写一次，迟早一边改成 .meta.yml 另一边没改。
from app.rag.loader import SIDECAR_SUFFIX

log = get_logger(__name__)

_DATE_IN_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------


class IngestStage(str, Enum):
    """入库状态。值直接进日志和报表，所以用英文短词。"""

    PENDING = "pending"
    VALIDATING = "validating"
    PARSING = "parsing"
    NORMALIZING = "normalizing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    PERSISTING = "persisting"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


TERMINAL_STAGES = frozenset(
    {IngestStage.COMPLETED, IngestStage.PARTIAL, IngestStage.FAILED, IngestStage.SKIPPED}
)

#: 允许的迁移。写成显式的表，而不是靠 if/else 到处判断 ——
#: 状态机散落在流程里，最后一定出现"某个字段跳了两级"这种查不出来的问题。
_TRANSITIONS: dict[IngestStage, frozenset[IngestStage]] = {
    IngestStage.PENDING: frozenset(
        {IngestStage.VALIDATING, IngestStage.SKIPPED, IngestStage.FAILED}
    ),
    IngestStage.VALIDATING: frozenset(
        {IngestStage.PARSING, IngestStage.SKIPPED, IngestStage.FAILED}
    ),
    IngestStage.PARSING: frozenset({IngestStage.NORMALIZING, IngestStage.SKIPPED, IngestStage.FAILED}),
    IngestStage.NORMALIZING: frozenset({IngestStage.CHUNKING, IngestStage.FAILED}),
    IngestStage.CHUNKING: frozenset({IngestStage.EMBEDDING, IngestStage.SKIPPED, IngestStage.FAILED}),
    IngestStage.EMBEDDING: frozenset({IngestStage.INDEXING, IngestStage.FAILED}),
    IngestStage.INDEXING: frozenset({IngestStage.PERSISTING, IngestStage.PARTIAL, IngestStage.FAILED}),
    IngestStage.PERSISTING: frozenset(
        {IngestStage.COMPLETED, IngestStage.PARTIAL, IngestStage.FAILED}
    ),
    IngestStage.COMPLETED: frozenset(),
    IngestStage.PARTIAL: frozenset(),
    IngestStage.FAILED: frozenset(),
    IngestStage.SKIPPED: frozenset(),
}


class IllegalTransition(RuntimeError):
    """状态迁移非法。属于代码 bug，不是数据问题，所以要吵一点。"""


class StageFailure(RuntimeError):
    """某个阶段失败。带上是哪一步、能不能重试。"""

    def __init__(self, stage: IngestStage, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.stage = stage
        self.retryable = retryable


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class IngestOptions:
    """一批文件的处理参数。"""

    #: 解析并切分的并发度。向量化在 worker 里做（公网往返，慢），
    #: 写 Chroma 始终在主线程做（SQLite 单写者，并发写只会互相锁）。
    workers: int = 4
    #: 只用于"可能是网络抖动"的步骤（向量化/写索引）。
    retries: int = 2
    #: 重试前的等待秒数，翻倍递增。
    retry_backoff: float = 0.5
    full: bool = False
    purge: bool = False
    chunking: ChunkingOptions = field(default_factory=ChunkingOptions)
    normalize: NormalizeOptions = field(default_factory=NormalizeOptions)


@dataclass
class DocumentRecord:
    """写进元数据表的一行。**内容侧**字段，状态侧由调用方回填。

    刻意不用 `loader.LoadedDoc`：那个结构是给"正文拼成一个大字符串"的老链路用的，
    新链路手里只有 `ParsedDocument`，硬转一次反而多一层容易漂的映射。
    """

    file_path: str
    document_id: str
    title: str
    doc_no: str = ""
    department: str = ""
    doc_type: str = "其他"
    college: str = "全校"
    publish_date: dt.date | None = None
    year: int | None = None
    url: str = ""
    doc_family: str = ""
    version: int | None = None
    effective_from: dt.date | None = None
    effective_to: dt.date | None = None
    summary: str = ""
    content_hash: str = ""
    parse_status: str = "completed"


@dataclass
class PreparedDocument:
    """解析、清洗、切分都做完，但还没落库的一份文件。"""

    path: Path
    record: DocumentRecord
    doc: ParsedDocument
    chunks: list[Chunk]

    @property
    def document_id(self) -> str:
        return self.record.document_id


@dataclass
class FileOutcome:
    """一份文件的最终结果。报表和测试都以它为准。"""

    file_path: str
    document_id: str = ""
    title: str = ""
    stage: IngestStage = IngestStage.PENDING
    chunks: int = 0
    blocks: int = 0
    attempts: int = 0
    duration: float = 0.0
    reason: str = ""
    errors: list[str] = field(default_factory=list)
    transitions: list[IngestStage] = field(default_factory=list)

    def advance(self, stage: IngestStage) -> None:
        """迁移到下一状态，非法迁移直接抛。

        原地重入（stage 相同）是允许的：写库失败重试时会重跑同一阶段，
        这时状态本来就该停在原地，不该报"非法迁移"。
        """
        if stage is self.stage:
            return
        if stage not in _TRANSITIONS[self.stage]:
            raise IllegalTransition(
                f"{self.file_path}：不允许 {self.stage.value} → {stage.value}"
            )
        self.stage = stage
        self.transitions.append(stage)

    @property
    def ok(self) -> bool:
        return self.stage in (IngestStage.COMPLETED, IngestStage.PARTIAL)

    def as_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "document_id": self.document_id,
            "title": self.title,
            "stage": self.stage.value,
            "chunks": self.chunks,
            "blocks": self.blocks,
            "attempts": self.attempts,
            "duration": round(self.duration, 3),
            "reason": self.reason,
            "errors": list(self.errors),
        }


@dataclass
class BatchReport:
    """一批文件的结果。"""

    total: int = 0
    completed: int = 0
    partial: int = 0
    failed: int = 0
    skipped: int = 0
    chunks: int = 0
    duration: float = 0.0
    outcomes: list[FileOutcome] = field(default_factory=list)
    #: content_hash → 多个 file_path。同一份内容存了两遍时才出现。
    duplicates: dict[str, list[str]] = field(default_factory=dict)

    @property
    def indexed(self) -> int:
        """真正入库成功的文件数（完成 + 部分完成）。"""
        return self.completed + self.partial

    def failures(self) -> list[FileOutcome]:
        return [o for o in self.outcomes if o.stage is IngestStage.FAILED]

    def skipped_files(self) -> list[FileOutcome]:
        return [o for o in self.outcomes if o.stage is IngestStage.SKIPPED]

    def as_dict(self) -> dict[str, Any]:
        return {
            "总数": self.total,
            "完成": self.completed,
            "部分完成": self.partial,
            "失败": self.failed,
            "跳过": self.skipped,
            "chunk数": self.chunks,
            "耗时": round(self.duration, 2),
            "重复内容": {h: paths for h, paths in self.duplicates.items()},
            "明细": [o.as_dict() for o in self.outcomes],
        }

    def summary_line(self) -> str:
        return (
            f"{self.total} 份 → 完成 {self.completed} / 部分 {self.partial} / "
            f"跳过 {self.skipped} / 失败 {self.failed} | {self.chunks} chunk | "
            f"{self.duration:.1f}s"
        )


# ---------------------------------------------------------------------------
# 元数据：侧车 + 文件名兜底
# ---------------------------------------------------------------------------


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + SIDECAR_SUFFIX)


def _read_sidecar(path: Path) -> dict[str, Any]:
    """读同名侧车元数据。读不到就当没有 —— 一份文件坏掉不该影响别的。"""
    target = sidecar_path(path)
    if not target.is_file():
        return {}
    try:
        import yaml

        meta = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("侧车元数据读取失败 %s：%s", target.name, exc)
        return {}
    return meta if isinstance(meta, dict) else {}


def _date_from_filename(name: str) -> dt.date | None:
    match = _DATE_IN_NAME.search(name)
    if not match:
        return None
    try:
        return dt.date(*(int(group) for group in match.groups()))
    except ValueError:
        return None


def _summary_of(doc: ParsedDocument, limit: int = 140) -> str:
    """一句话摘要：第一段正文的开头。

    刻意不调 LLM —— 灌库阶段为 100 份文件调 100 次模型，又慢又贵，
    而元数据表里的 summary 只用于列表展示。
    """
    for block in doc.blocks:
        text = block.text.strip()
        if text and not block.extra.get("is_boilerplate"):
            cleaned = re.sub(r"\s+", " ", text)
            return cleaned[:limit] + ("…" if len(cleaned) > limit else "")
    return ""


def _build_record(doc: ParsedDocument, path: Path, content_hash: str) -> DocumentRecord:
    meta = doc.metadata
    publish_date = meta.publish_date or _date_from_filename(path.name)
    return DocumentRecord(
        file_path=meta.file_path or path.as_posix(),
        document_id=doc.document_id or document_id_for(path.as_posix()),
        title=doc.title or path.stem,
        doc_no=meta.doc_no,
        department=meta.department,
        doc_type=meta.doc_type or "其他",
        college=meta.college or "全校",
        publish_date=publish_date,
        year=meta.year or (publish_date.year if publish_date else None),
        url=meta.url,
        doc_family=meta.doc_family,
        version=meta.version,
        effective_from=meta.effective_from,
        effective_to=meta.effective_to,
        summary=_summary_of(doc) or meta.get("summary", ""),
        content_hash=content_hash,
        parse_status=doc.parse_status.value,
    )


def _sha256(path: Path, block_size: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------


class IngestPipeline:
    """把一批文件跑完「解析 → 切分 → 向量化 → 落库」。

    三个外部依赖都是注入的，方便单测和替换：

        index            有 embed / upsert_document / delete_document 的对象
        write_metadata   把 DocumentRecord 写进元数据表（回调）
        read_states      （可选）读这份文件在库里的状态侧字段
    """

    def __init__(
        self,
        index: Any,
        write_metadata: Callable[[DocumentRecord, dict[str, Any]], None],
        *,
        read_states: Callable[[str], dict[str, Any]] | None = None,
        unchanged_check: Callable[[DocumentRecord], bool] | None = None,
        options: IngestOptions | None = None,
    ) -> None:
        self.index = index
        self.write_metadata = write_metadata
        self.read_states = read_states
        #: 判断"这份文件和库里已有的那份没差别"，用来省掉一次 embedding。
        #: 判断依据必须是**内容 + 内容侧元数据**：只比内容哈希会漏掉
        #: 「改了学院/文号但文件字节没变」的情况，那正是管理员修错的时候。
        self.unchanged_check = unchanged_check
        self.options = options or IngestOptions()

    # ---- 对外入口 ----

    def run(self, paths: Sequence[Path]) -> BatchReport:
        started = time.time()
        path_list = [Path(p) for p in paths]
        report = BatchReport(total=len(path_list))

        if not path_list:
            report.duration = time.time() - started
            return report

        # 先把向量函数预热在主线程里，避免 4 个 worker 同时去建客户端。
        # 注意预热**不是**拿一句话去算向量（那会白花一次接口调用，还会让
        # 单测里的调用计数对不上），只是把客户端建出来。
        try:
            warmup = getattr(self.index, "warmup", None)
            if callable(warmup):
                warmup()
        except Exception as exc:  # noqa: BLE001
            log.warning("向量函数预热失败（不影响后续逐文件重试）：%s", exc)

        records: list[DocumentRecord] = []
        workers = max(1, int(self.options.workers or 1))
        #: 先全部准备完、再统一删、最后逐份写。三步的顺序是刻意的：
        #: 删除必须**整批一次**，理由见 `_clear()` —— 逐份删会把向量库写坏。
        prepared_items: list[tuple[PreparedDocument, FileOutcome]] = []
        outcomes: list[FileOutcome] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._prepare_safely, path) for path in path_list]
            for future in futures:
                prepared, outcome = future.result()
                outcomes.append(outcome)
                if prepared is not None:
                    prepared_items.append((prepared, outcome))

        self._clear([prepared for prepared, _ in prepared_items])

        for prepared, outcome in prepared_items:
            # 写库在主线程串行做：Chroma 是 SQLite 单写者，
            # 多线程并发写只会互相等锁，还会让失败回滚变复杂。
            self._write_with_retry(prepared, outcome)
            if outcome.ok:
                records.append(prepared.record)

        for outcome in outcomes:
            report.outcomes.append(outcome)
            if outcome.stage is IngestStage.COMPLETED:
                report.completed += 1
            elif outcome.stage is IngestStage.PARTIAL:
                report.partial += 1
            elif outcome.stage is IngestStage.SKIPPED:
                report.skipped += 1
            else:
                report.failed += 1
            report.chunks += outcome.chunks

        report.duplicates = group_by_hash(records)
        report.duration = time.time() - started
        log.info("批处理完成 | %s", report.summary_line())
        return report

    # ---- 阶段：清旧片段（整批一次） ----

    def _clear(self, prepared: list[PreparedDocument]) -> None:
        """把这批文件在向量库里的旧片段**一次性**清掉。

        为什么不是"每份文件写之前删掉它自己的"：实测（`var/_exp_chroma_churn2.py`，
        805 条 / 1024 维 / 115 份文件）逐份删+写会把 chroma 的 HNSW 索引写坏，
        换进程打开直接 `Error loading hnsw index`，整个知识库不可检索 ——
        而且分相（先删 115 次、再统一写）同样会坏，说明是**删除次数**的问题。
        一次删完再写就正常。

        索引对象没有批量删能力的（单测里的假索引）退化成逐份删：
        那种规模（个位数文件）从来不会触发这个问题。
        """
        if not prepared:
            return
        bulk = getattr(self.index, "delete_documents_bulk", None)
        try:
            if callable(bulk):
                bulk(
                    [doc.document_id for doc in prepared],
                    [doc.record.file_path for doc in prepared],
                )
                return
            for doc in prepared:
                self.index.delete_document(doc.document_id)
        except Exception as exc:  # noqa: BLE001 - 清理失败不该让整批停摆
            log.warning("清理旧片段失败（按无旧片段继续）：%s", exc)

    # ---- 阶段：解析到切分（可并发） ----

    def _prepare_safely(self, path: Path) -> tuple[PreparedDocument | None, FileOutcome]:
        """跑完 prepare 并保证**绝不抛异常**。"""
        # 报表里的路径一律用「相对项目根的 POSIX 路径」，和元数据表、向量库、
        # 前端「查看原文」保持同一口径。存绝对路径的话，同一个文件在报表里
        # 写 D:\...、在数据库里写 data/raw/...，对账时两边永远对不上。
        outcome = FileOutcome(file_path=relative_path(path))
        started = time.time()
        try:
            prepared = self._prepare(path, outcome)
        except StageFailure as failure:
            _mark_failed(outcome, failure.stage, str(failure))
            prepared = None
        except Exception as exc:  # noqa: BLE001 - 单文件隔离的最后一层兜底
            log.exception("入库异常 | %s", path)
            _mark_failed(outcome, outcome.stage, f"{type(exc).__name__}: {exc}")
            prepared = None
        outcome.duration = time.time() - started
        outcome.attempts = max(1, outcome.attempts)
        return prepared, outcome

    def _prepare(self, path: Path, outcome: FileOutcome) -> PreparedDocument | None:
        opts = self.options

        # --- VALIDATING ---
        outcome.advance(IngestStage.VALIDATING)
        if not path.is_file():
            raise StageFailure(IngestStage.VALIDATING, f"文件不存在：{path}")
        if path.stat().st_size == 0:
            raise StageFailure(IngestStage.VALIDATING, "文件是空的（0 字节）")

        file_type = detect_file_type(path)
        if file_type not in SUPPORTED_TYPES:
            raise StageFailure(
                IngestStage.VALIDATING,
                f"不支持的格式：{file_type.value}（可能是 .doc/.xls/.ppt 老格式）",
            )

        # --- PARSING ---
        outcome.advance(IngestStage.PARSING)
        sidecar = _read_sidecar(path)
        doc = parse_file(
            path,
            ParseOptions(include_images=True, assets_dir=None, sidecar=sidecar),
        )
        outcome.document_id = doc.document_id
        outcome.title = doc.title
        outcome.blocks = len(doc.blocks)

        if doc.parse_status is ParseStatus.FAILED:
            message = doc.errors[0].message if doc.errors else "解析失败"
            raise StageFailure(IngestStage.PARSING, message)
        if doc.parse_status is ParseStatus.EMPTY:
            # 扫描件、纯图表文件走这条。**不是失败，但要让人看见** ——
            # 老链路在这里只打一行 WARNING 然后 continue，文件就这么无声消失了。
            reason = doc.errors[0].message if doc.errors else "没有可提取的文本"
            outcome.advance(IngestStage.SKIPPED)
            outcome.reason = f"没有文本层，需要 OCR：{reason}"
            return None

        # --- NORMALIZING ---
        outcome.advance(IngestStage.NORMALIZING)
        clean_doc, _ = normalize(doc, opts.normalize)

        # --- CHUNKING ---
        outcome.advance(IngestStage.CHUNKING)
        chunks = chunk_document(clean_doc, opts.chunking)
        if not chunks:
            outcome.advance(IngestStage.SKIPPED)
            outcome.reason = "切分后没有任何片段（内容可能全被判定为页眉页脚）"
            return None

        content_hash = _sha256(path)
        record = _build_record(clean_doc, path, content_hash)
        outcome.document_id = record.document_id
        outcome.title = record.title

        # 增量场景：内容和档案都没变 → 整个跳过。
        # 放在切分之后判断，是因为判断要用的就是"这份文件的档案信息"，
        # 而它得先解析出来。解析很快（毫秒级），花钱的是后面的 embedding。
        if self.unchanged_check is not None and not opts.full:
            try:
                if self.unchanged_check(record):
                    outcome.advance(IngestStage.SKIPPED)
                    outcome.reason = "内容与档案都没变，跳过（不重算向量）"
                    return None
            except Exception as exc:  # noqa: BLE001
                log.warning("判断文件是否有变化失败（按有变化处理）%s：%s", path, exc)

        outcome.chunks = len(chunks)
        return PreparedDocument(path=path, record=record, doc=clean_doc, chunks=chunks)

    # ---- 阶段：向量化到落库（必须串行） ----

    def _write_with_retry(self, prepared: PreparedDocument, outcome: FileOutcome) -> None:
        opts = self.options
        attempts = opts.retries + 1
        for attempt in range(1, attempts + 1):
            outcome.attempts = attempt
            try:
                self._write(prepared, outcome)
                return
            except StageFailure as failure:
                if not failure.retryable or attempt >= attempts:
                    log.error(
                        "入库失败 | %s | 阶段=%s | %s",
                        prepared.record.file_path, failure.stage.value, failure,
                    )
                    _mark_failed(outcome, failure.stage, str(failure))
                    self._rollback(prepared, outcome)
                    return
                log.warning(
                    "入库重试 | %s | 阶段=%s | 第 %d/%d 次 | %s",
                    prepared.record.file_path, failure.stage.value, attempt, attempts, failure,
                )
                time.sleep(opts.retry_backoff * attempt)
            except Exception as exc:  # noqa: BLE001
                log.exception("入库异常 | %s", prepared.record.file_path)
                _mark_failed(outcome, outcome.stage, f"{type(exc).__name__}: {exc}")
                self._rollback(prepared, outcome)
                return

    def _write(self, prepared: PreparedDocument, outcome: FileOutcome) -> None:
        record = prepared.record
        chunks = prepared.chunks

        # 状态侧字段：现行/废止只有数据库里有，向量库那边是副本。
        # 副本必须和正本一致，否则检索时"已废止"的过滤会失效。
        states: dict[str, Any] = {}
        if self.read_states is not None:
            try:
                states = self.read_states(record.file_path) or {}
            except Exception as exc:  # noqa: BLE001
                log.warning("读取文件旧状态失败（按现行处理）%s：%s", record.file_path, exc)
        status = str(states.get("status") or "active")
        for chunk in chunks:
            chunk.metadata["status"] = status
            chunk.metadata["file_missing"] = 0

        # --- EMBEDDING ---
        if outcome.stage is IngestStage.CHUNKING:
            outcome.advance(IngestStage.EMBEDDING)
        try:
            vectors = self.index.embed([chunk.text for chunk in chunks])
        except Exception as exc:  # noqa: BLE001
            raise StageFailure(
                IngestStage.EMBEDDING,
                f"向量化失败：{type(exc).__name__}: {exc}",
                retryable=True,
            ) from exc

        # --- INDEXING ---
        outcome.advance(IngestStage.INDEXING)
        # replace=False：旧片段已经由 `_clear()` 整批删掉了。这里再让每份文件
        # 自己删一次，就退回"逐份删+写"，会把索引写坏（见 _clear 注释）。
        result = self.index.upsert_document(chunks, embeddings=vectors, replace=False)
        if result.errors and result.written == 0:
            raise StageFailure(
                IngestStage.INDEXING,
                f"向量写入全部失败：{result.errors[0]}",
                retryable=True,
            )
        partial_index = bool(result.errors)

        # --- PERSISTING ---
        outcome.advance(IngestStage.PERSISTING)
        try:
            self.write_metadata(record, states)
        except Exception as exc:  # noqa: BLE001
            raise StageFailure(
                IngestStage.PERSISTING,
                f"元数据写入失败：{type(exc).__name__}: {exc}",
                retryable=True,
            ) from exc

        outcome.chunks = result.written
        if partial_index or prepared.doc.parse_status is ParseStatus.PARTIAL:
            outcome.advance(IngestStage.PARTIAL)
            if partial_index:
                outcome.reason = f"部分片段写入失败：{result.errors[0]}"
            else:
                outcome.reason = (
                    prepared.doc.errors[0].message
                    if prepared.doc.errors
                    else "部分页面/表格解析失败"
                )
            return
        outcome.advance(IngestStage.COMPLETED)

    def _rollback(self, prepared: PreparedDocument, outcome: FileOutcome) -> None:
        """失败收尾：把这份文件已写入的半成品片段删掉。

        不留半份的理由：半份 chunk 在检索里是"好像有、其实不全"，
        比彻底搜不到更糟 —— 学生会得到缺条件的答案，而且没人会想到是入库断了。
        """
        try:
            removed = self.index.delete_document(prepared.document_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("回滚失败 %s：%s", prepared.record.file_path, exc)
            return
        if removed:
            log.info("回滚 | %s | 删除半成品片段 %d 个", prepared.record.file_path, removed)
            outcome.reason = (outcome.reason + "；" if outcome.reason else "") + (
                f"已回滚 {removed} 个片段"
            )

def _mark_failed(outcome: FileOutcome, stage: IngestStage, message: str) -> None:
    """把 outcome 置为 FAILED，并保留失败发生在哪一步。"""
    outcome.errors.append(message)
    if outcome.stage is not IngestStage.FAILED:
        outcome.stage = IngestStage.FAILED
        outcome.transitions.append(IngestStage.FAILED)
    outcome.reason = outcome.reason or f"{stage.value}: {message}"


def group_by_hash(records: Iterable[DocumentRecord]) -> dict[str, list[str]]:
    """内容哈希相同的文件路径。供调用方做「重复内容」报表。

    放在这里而不是 `_find_duplicates` 里：并发流程里 outcome 拿不到哈希，
    而 record 能拿到。硬塞进 outcome 只会让那个结构多一个只在报表时才有用的字段。
    """
    buckets: dict[str, list[str]] = {}
    for record in records:
        if record.content_hash:
            buckets.setdefault(record.content_hash, []).append(record.file_path)
    return {hash_: paths for hash_, paths in buckets.items() if len(paths) > 1}
