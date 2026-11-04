"""结构感知切分。

为什么不能直接上 RecursiveCharacterTextSplitter
------------------------------------------------
现有 `app/rag/splitter.py` 把整份文件拼成一个字符串再按 600 字切。实测的
后果是：`31_教学_...毕业论文.md` 的 9 行表格被切成 9 行 + 3 行两块，
**第二块的首行是数据行，表头丢了** —— 学生问「答辩时间」，模型拿到的是
一堆没有列名的日期。

这一层的做法：
    · 按 heading_path 分组，同一章节的段落才可能进同一个 chunk
    · 表格整体成块；超长按**行**分组，**每组都重新挂表头**
    · 图片 OCR 文本单独成块，带页码
    · PPT 按幻灯片、Excel 按 sheet + 数据区组织
    · 每个 chunk 的正文前缀带上 `标题 > 章节`，让向量模型知道这段在说什么

刻意不做全文 overlap：结构切分已经保证语义完整，再叠加字符级 overlap
只会让同一段内容在两个 chunk 里重复召回、互相挤占 top_k 名额。
只有**超长段落被强制切开**时才用 overlap（否则句子会断在中间）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import BaseModel, Field

from app.logging_config import get_logger
from app.parsing.models import (
    BlockType,
    DocumentBlock,
    ParsedDocument,
    SourceLocation,
    TableData,
)

log = get_logger(__name__)

# 句末标点：超长段落优先在这些位置断开
_SENTENCE_ENDS = ("。", "！", "？", "；", "\n")


@dataclass
class ChunkingOptions:
    """切分参数。

    默认值按中文校园文件调：一段通知通常在 200-500 字，700 字能装下
    「一条完整要求」，又不会大到让检索失去区分度。
    """

    max_chars: int = 700
    min_chars: int = 120
    # 只在超长段落被强制切开时使用
    overlap: int = 80
    table_max_chars: int = 1200
    include_image_ocr: bool = True
    drop_boilerplate: bool = True
    # 同一份文件里文字完全相同的 chunk 只留第一个。校园文件里模板页很多
    # （实习记录本一页一篇日记，40 页的模板会切出 40 个一字不差的 chunk），
    # 不去重就会让一次检索的 top_k 全被同一段文字占满。
    dedupe: bool = True


class Chunk(BaseModel):
    """一个待入库的片段。

    字段与阶段五要求对齐：document_id / chunk_id / file_name / file_type /
    content_type / chunk_index / source_location / heading_path /
    parser_version / document_version。
    """

    chunk_id: str
    document_id: str
    text: str
    content_type: str
    chunk_index: int
    heading_path: list[str] = Field(default_factory=list)
    source_location: SourceLocation = Field(default_factory=SourceLocation)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_chroma_metadata(self) -> dict[str, Any]:
        """转成 Chroma 能接受的 metadata（只允许 str / int / float / bool）。

        嵌套结构一律不进 Chroma：表格的原貌留在 MySQL 或解析产物里，
        向量库只留「能用来过滤」的标量。这是阶段五的硬约束。
        """
        flat: dict[str, Any] = {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "chunk_index": int(self.chunk_index),
            "content_type": self.content_type,
            "heading_path": " > ".join(self.heading_path),
        }
        flat.update(self.source_location.to_flat_dict())
        for key, value in self.metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                flat[key] = value
            elif isinstance(value, (list, tuple)):
                flat[key] = "、".join(str(item) for item in value)
            else:
                flat[key] = str(value)
        return flat


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """超长文本按句末标点切开。

    只在「单段就超过 max_chars」时用。切点尽量落在句号/分号/换行，
    避免把「报名截止时间为 2026 年 9 月 30 日」切成两半。
    """
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            window = text[start:end]
            cut = max(window.rfind(mark) for mark in _SENTENCE_ENDS)
            if cut >= max_chars * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return pieces


def _table_markdown(table: TableData, rows: list[list[str]], with_header: bool) -> str:
    """按 `rows` 渲染一张 Markdown 表。with_header=True 时重挂表头。"""
    lines: list[str] = []
    header_rows = table.header_row_count if with_header else 0
    grid = table.grid()
    if header_rows > 0:
        head = grid[:header_rows]
        if header_rows == 1:
            lines.append("| " + " | ".join(cell for cell in head[0]) + " |")
        else:
            merged: list[str] = []
            for col in range(table.n_cols):
                parts: list[str] = []
                for r in range(header_rows):
                    value = head[r][col].strip()
                    if value and (not parts or parts[-1] != value):
                        parts.append(value)
                merged.append(" / ".join(parts))
            lines.append("| " + " | ".join(merged) + " |")
        lines.append("| " + " | ".join("---" for _ in range(table.n_cols)) + " |")

    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def _chunk_table(
    table: TableData, doc: ParsedDocument, block: DocumentBlock,
    options: ChunkingOptions, start_index: int,
) -> list[Chunk]:
    """表格切分：能装下就整张，装不下按行分组，**每组都带表头**。"""
    grid = table.grid()
    header_rows = table.header_row_count
    body = grid[header_rows:]
    caption = table.caption.strip()
    # 表格 chunk 同样要带章节前缀。没有它，一张「名额/金额」表会丢掉
    # 「一、奖励标准」这个语境，向量只由表格内容决定 —— 学生问
    # 「国家励志奖学金多少钱」，它和一张课程表在向量空间里分不开。
    prefix = _prefix(block.heading_path, doc.title)
    overhead = len(prefix) + len(caption) + 4

    def make(text_body: str, index: int, row_range: tuple[int, int] | None) -> Chunk:
        head = f"{prefix}\n{caption}" if caption else prefix
        metadata = _base_metadata(doc, block)
        if row_range:
            metadata["table_row_start"] = row_range[0]
            metadata["table_row_end"] = row_range[1]
        metadata["table_rows"] = table.n_rows
        metadata["table_cols"] = table.n_cols
        metadata["table_header_rows"] = header_rows
        return Chunk(
            chunk_id=_chunk_id(doc.document_id, index),
            document_id=doc.document_id,
            text=f"{head}\n{text_body}",
            content_type="table",
            chunk_index=index,
            heading_path=list(block.heading_path),
            source_location=block.source_location,
            metadata=metadata,
        )

    whole = _table_markdown(table, body, with_header=True)
    if overhead + len(whole) <= options.table_max_chars or not body:
        return [make(_table_markdown(table, body, with_header=True), start_index, None)]

    # 按行分组：逐行累加，超过上限就断开。表头由 _table_markdown 每组重挂。
    # 预算要扣掉前缀和标题占的位置，否则加完前缀实际会超出 table_max_chars。
    budget = max(options.table_max_chars - overhead, options.table_max_chars // 2)
    chunks: list[Chunk] = []
    group: list[list[str]] = []
    group_start = 0
    size = 0
    for offset, row in enumerate(body):
        row_len = sum(len(str(cell)) for cell in row) + table.n_cols * 3
        if group and size + row_len > budget:
            chunks.append(
                make(
                    _table_markdown(table, group, with_header=True),
                    start_index + len(chunks),
                    (group_start + header_rows, offset - 1 + header_rows),
                )
            )
            group = []
            group_start = offset
            size = 0
        group.append(row)
        size += row_len
    if group:
        chunks.append(
            make(
                _table_markdown(table, group, with_header=True),
                start_index + len(chunks),
                (group_start + header_rows, len(body) - 1 + header_rows),
            )
        )
    return chunks


def _base_metadata(doc: ParsedDocument, block: DocumentBlock | None = None) -> dict[str, Any]:
    """档案字段 + 解析版本。每个 chunk 都要带，检索时用来硬过滤。"""
    meta = doc.metadata
    return {
        "file_name": doc.file_name,
        "file_type": doc.file_type.value,
        "file_path": meta.file_path,
        "title": doc.title,
        "doc_no": meta.doc_no,
        "department": meta.department,
        "doc_type": meta.doc_type,
        "college": meta.college,
        "publish_date": meta.publish_date.isoformat() if meta.publish_date else "",
        "year": int(meta.year or 0),
        "url": meta.url,
        "doc_family": meta.doc_family,
        "version": int(meta.version or 0),
        "document_version": int(meta.version or 0),
        "parser_name": doc.parser_name,
        "parser_version": doc.parser_version,
        # 检索的两个**地基字段**必须随 chunk 一起写进向量库，且不能缺省：
        # `retriever._baseline_filter()` 的 where 条件是 `status=active AND
        # file_missing=0`，Chroma 对「字段不存在」的匹配行为不明确 ——
        # 缺字段的 chunk 会在过滤时被漏掉，表现为「文件明明入库了却搜不到」。
        # 这里给的是默认值，真正的状态由 ingest 从数据库读出后覆盖。
        "status": "active",
        "file_missing": 0,
    }


def _chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}-c{index:04d}"


def _prefix(heading_path: list[str], title: str) -> str:
    """chunk 正文前缀。

    只写标题不够：一份通知里「一、申请条件」和「五、附则」的措辞完全不同，
    没有章节名，向量模型分不清这段在讲什么。
    """
    parts = [part for part in heading_path if part.strip()]
    if not parts:
        return f"【{title}】"
    if parts[0] == title:
        return "【" + " > ".join(parts) + "】"
    return "【" + " > ".join([title] + parts) + "】"


_CHUNK_PREFIX = re.compile(r"^【[^】]*】\n")


def _strip_prefix(text: str) -> str:
    """去掉 chunk 开头的【标题 > 章节】。合并时只保留最前面那一个。"""
    return _CHUNK_PREFIX.sub("", text, count=1)


def _union_location(left: SourceLocation, right: SourceLocation) -> SourceLocation:
    """两个位置取并集：页码/段落号取首尾范围，其余字段保留 left 的。

    bbox 只保留 left 的 —— 一个跨块 chunk 的外接矩形没有意义
    （可能横跨整页），留着反而会让前端高亮框错位。
    """
    def _span(attr: str, pick: Any) -> int | None:
        values = [
            getattr(location, attr)
            for location in (left, right)
            if getattr(location, attr) is not None
        ]
        return pick(values) if values else None

    return left.model_copy(
        update={
            "page_start": _span("page_start", min),
            "page_end": _span("page_end", max),
            "paragraph_index": _span("paragraph_index", min),
            "char_offset": _span("char_offset", min),
        }
    )


def _absorb(head: Chunk, tail: Chunk) -> Chunk:
    """把 tail 并进 head：正文相接、位置取并集，前缀/类型/标题路径保留 head 的。"""
    body = _strip_prefix(tail.text).strip()
    if not body:
        return head

    metadata = dict(head.metadata)
    # 两块都是被切开的同一张表时，行范围要跟着扩，否则「本块是第 52-58 行」
    # 会和正文里实际出现的第 59 行对不上，追溯表格来源时就错了。
    if "table_row_start" in metadata and "table_row_start" in tail.metadata:
        metadata["table_row_start"] = min(
            metadata["table_row_start"], tail.metadata["table_row_start"]
        )
        metadata["table_row_end"] = max(
            metadata.get("table_row_end", 0), tail.metadata.get("table_row_end", 0)
        )

    return head.model_copy(
        update={
            "text": f"{head.text}\n\n{body}",
            "source_location": _union_location(head.source_location, tail.source_location),
            "metadata": metadata,
        }
    )


def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
    """同一份文件里文字完全相同的 chunk 只留第一个。

    两个要点：
    · **只在一份文件内部去重**。跨文件的重复（同一份通知发了两年）是有意义的，
      不能在这里吃掉 —— 它们的元数据不同，检索时要能分别命中。
    · 保留下来的那个携带它自己首次出现的位置。模板页之间的位置差异对检索
      没有价值，反而会让「打开原文」跳到一处随机的示例页。
    """
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        if chunk.text in seen:
            continue
        seen.add(chunk.text)
        unique.append(chunk)
    return unique


_MERGE_LOOKBACK = 2


def _can_merge(head: Chunk, tail: Chunk, max_chars: int) -> bool:
    """两个 chunk 能不能合。三个条件缺一不可，理由见 _coalesce。"""
    if head.content_type != tail.content_type:
        return False
    if head.heading_path != tail.heading_path:
        return False
    merged_len = len(head.text) + len(_strip_prefix(tail.text)) + 2
    return merged_len <= max_chars


def _coalesce(chunks: list[Chunk], min_chars: int, max_chars: int) -> list[Chunk]:
    """把过短的 chunk 并进邻居，避免 40 字的碎片占掉一个 top_k 名额。

    42 字的 chunk（「实 习 考 勤 表」这种孤零零一行）单独入库只是噪音：
    它的向量几乎没有区分度，却和别的 chunk 抢同一个召回名额。

    三个限制都不是随手加的：

    · **只并同类型**。段落并进表格，会让「这行字属于表里哪一格」彻底说不清。
    · **只并同章节**。跨章节合并会把前缀换成另一节的，学生看到的
      「参考章节」就和内容对不上 —— 一份通知里「一、奖励标准」的正文被标成
      「二、申请条件」，比短一点更糟。
    · **合完不超过 max_chars**。否则为了消化一个 40 字的碎片，造出一个
      1200 字的巨块，向量被稀释得更厉害。

    方向：先试后面（保留后一块的前缀，短块多半是上一节的尾巴，挂后一节更贴语境）；
    后面接不上时再回看 `_MERGE_LOOKBACK` 个，覆盖「同章节里被一张表隔开的两段」——
    那是同一节被表格切断，不是两个章节。
    """
    result: list[Chunk] = []
    for chunk in chunks:
        if (
            result
            and len(result[-1].text) < min_chars
            and _can_merge(result[-1], chunk, max_chars)
        ):
            result[-1] = _absorb(result[-1], chunk)
            continue

        if len(chunk.text) < min_chars:
            merged = False
            for offset in range(1, min(_MERGE_LOOKBACK, len(result)) + 1):
                target = result[-offset]
                if _can_merge(target, chunk, max_chars):
                    result[-offset] = _absorb(target, chunk)
                    merged = True
                    break
            if merged:
                continue

        result.append(chunk)
    return result


def _reindex(chunks: list[Chunk], document_id: str) -> list[Chunk]:
    """重排 chunk_index / chunk_id，保证从 0 连续。

    去重和合并都会留下空洞。旧切分器就有这个毛病：`len(text) < 20` 的碎片
    被丢弃但序号照加，chunk_index 出现 0,1,3,4 —— 前端按序号排序时看不出来，
    但「这个文件一共几块、第几块」这类追溯全乱了。
    """
    return [
        chunk.model_copy(
            update={"chunk_index": index, "chunk_id": _chunk_id(document_id, index)}
        )
        for index, chunk in enumerate(chunks)
    ]


def _finalize(chunks: list[Chunk], doc: ParsedDocument, opts: ChunkingOptions) -> list[Chunk]:
    """收尾：去重 → 合并过短块 → 连续编号。

    顺序不能反。先去重再合并：模板文档里 38 个一字不差的 42 字表格，如果先合并，
    它们会被串成一个 1600 字的巨块（38 × 42），比原来更糟。
    """
    if opts.dedupe:
        chunks = _dedupe(chunks)
    chunks = _coalesce(chunks, opts.min_chars, opts.max_chars)
    return _reindex(chunks, doc.document_id)


def chunk_document(
    doc: ParsedDocument, options: ChunkingOptions | None = None
) -> list[Chunk]:
    """把一份解析好的文档切成 Chunk 列表。

    不依赖任何具体解析器，只认 ParsedDocument —— 这正是统一中间结构的价值。
    """
    opts = options or ChunkingOptions()

    blocks: Iterable[DocumentBlock] = doc.blocks
    if opts.drop_boilerplate:
        blocks = (b for b in doc.blocks if not b.extra.get("is_boilerplate"))

    chunks: list[Chunk] = []
    pending: list[DocumentBlock] = []
    pending_path: list[str] = []

    def flush() -> None:
        """把积累的段落块合成 chunk。"""
        nonlocal pending, pending_path
        if not pending:
            return
        text_parts = [b.text.strip() for b in pending if b.text.strip()]
        if not text_parts:
            pending = []
            return

        merged = "\n\n".join(text_parts)
        prefix = _prefix(pending_path, doc.title)

        if len(merged) <= opts.max_chars:
            chunks.append(
                _make_paragraph_chunk(doc, pending, prefix, merged, len(chunks))
            )
        else:
            # 超长：按段落边界分组；单段超长再按句末切
            group: list[str] = []
            size = 0
            for part in text_parts:
                if len(part) > opts.max_chars:
                    if group:
                        chunks.append(_make_paragraph_chunk(
                            doc, pending, prefix, "\n\n".join(group), len(chunks)
                        ))
                        group, size = [], 0
                    for piece in _split_long_text(part, opts.max_chars, opts.overlap):
                        chunks.append(_make_paragraph_chunk(
                            doc, pending, prefix, piece, len(chunks)
                        ))
                    continue
                if group and size + len(part) > opts.max_chars:
                    chunks.append(_make_paragraph_chunk(
                        doc, pending, prefix, "\n\n".join(group), len(chunks)
                    ))
                    group, size = [], 0
                group.append(part)
                size += len(part)
            if group:
                chunks.append(_make_paragraph_chunk(
                    doc, pending, prefix, "\n\n".join(group), len(chunks)
                ))
        pending = []

    for block in blocks:
        if block.block_type is BlockType.HEADING:
            flush()
            pending_path = list(block.heading_path)
            continue

        if block.block_type is BlockType.TABLE and block.table_data is not None:
            flush()
            chunks.extend(_chunk_table(block.table_data, doc, block, opts, len(chunks)))
            continue

        if block.block_type is BlockType.IMAGE:
            flush()
            if not opts.include_image_ocr:
                continue
            ocr_text = str(block.extra.get("ocr_text") or "").strip()
            if not ocr_text:
                # 没有 OCR 文本的图片只有一句 `embedded://...`，进库就是噪音。
                # 它的价值要等阶段四的 OCR 通道真正接上。
                continue
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc.document_id, len(chunks)),
                    document_id=doc.document_id,
                    text=f"{_prefix(block.heading_path, doc.title)}\n{ocr_text}",
                    content_type="image_ocr",
                    chunk_index=len(chunks),
                    heading_path=list(block.heading_path),
                    source_location=block.source_location,
                    metadata=_base_metadata(doc, block),
                )
            )
            continue

        if not block.text.strip():
            continue
        if not pending:
            pending_path = list(block.heading_path) or pending_path
        pending.append(block)

    flush()
    return _finalize(chunks, doc, opts)


def _make_paragraph_chunk(
    doc: ParsedDocument,
    blocks: list[DocumentBlock],
    prefix: str,
    body: str,
    index: int,
) -> Chunk:
    first = blocks[0]
    location = first.source_location
    # 一个 chunk 可能跨页，页码取首块到末块的范围
    if len(blocks) > 1:
        starts = [b.source_location.page_start for b in blocks if b.source_location.page_start]
        ends = [b.source_location.page_end for b in blocks if b.source_location.page_end]
        location = location.model_copy(
            update={
                "page_start": min(starts) if starts else None,
                "page_end": max(ends) if ends else None,
            }
        )
    return Chunk(
        chunk_id=_chunk_id(doc.document_id, index),
        document_id=doc.document_id,
        text=f"{prefix}\n{body}",
        content_type="paragraph",
        chunk_index=index,
        heading_path=list(blocks[0].heading_path),
        source_location=location,
        metadata=_base_metadata(doc, first),
    )


def chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    """切分质量概览，阶段五验收用。"""
    if not chunks:
        return {"chunk数": 0}
    sizes = [len(chunk.text) for chunk in chunks]
    by_type: dict[str, int] = {}
    for chunk in chunks:
        by_type[chunk.content_type] = by_type.get(chunk.content_type, 0) + 1
    return {
        "chunk数": len(chunks),
        "按类型": dict(sorted(by_type.items())),
        "最短": min(sizes),
        "最长": max(sizes),
        "平均": round(sum(sizes) / len(sizes)),
        "去重后": len({chunk.text for chunk in chunks}),
    }
