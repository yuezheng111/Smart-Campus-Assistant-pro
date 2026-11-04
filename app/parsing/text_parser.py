"""Markdown / 纯文本解析器。

这两类文件在项目里占多数（data/raw 里 55 个 md + 20 个 txt），但一直
被当成"反正就是纯文本"处理。实际上 md 里的 `#` 标题和 `|` 表格就是
现成的结构信息，只是旧代码没去读。

两处对旧行为的修正
------------------
1. 旧 loader 用 `errors="ignore"` 读文件，GBK 编码的 txt 会被**静默丢字**。
   这里按 utf-8-sig → gb18030 顺序试，都不行才用 replace 并在 errors 里记账。
2. 表格不再跟着正文一起被字符切分器切碎，而是整体成块（表头随块走）。
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

import yaml

from app.logging_config import get_logger
from app.parsing._heuristics import (
    apply_header_flags,
    cn_heading_level,
    normalize_heading_paths,
)
from app.parsing.base import BaseParser, ParseOptions
from app.parsing.models import (
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    FileType,
    ParsedDocument,
    ParseError,
    ParseStatus,
    SourceLocation,
    TableCell,
    TableData,
    new_block_id,
)

log = get_logger(__name__)

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)]|[（(]\d+[)）])\s+")
_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_TABLE_CAPTION = re.compile(r"^\s*\**\s*(表|附表|表格)\s*[\d一二三四五六七八九十]+")


def read_text(path: Path) -> tuple[str, str, bool]:
    """读文本，返回 (内容, 编码, 是否有损)。

    顺序是 utf-8-sig → utf-8 → gb18030。gb18030 几乎能覆盖所有中文编码，
    放在最后是因为它什么字节都能解出字符，放前面会把 UTF-8 解成乱码。
    """
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding), encoding, False
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8(replace)", True


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONT_MATTER.match(text.lstrip())
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    return meta, text.lstrip()[match.end():]


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _parse_table(lines: list[str], start: int) -> tuple[TableData, int]:
    """从 start 开始解析 Markdown 表格，返回 (表格, 结束行下标)。"""
    raw_rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        if not _TABLE_SEPARATOR.match(lines[index]):
            raw_rows.append(_split_row(lines[index]))
        index += 1

    n_cols = max((len(row) for row in raw_rows), default=0)
    if not raw_rows or n_cols == 0:
        return TableData(n_rows=0, n_cols=0), index

    # 补齐短行：Markdown 允许行尾省略空单元格
    for row in raw_rows:
        row.extend([""] * (n_cols - len(row)))

    has_separator = any(
        _TABLE_SEPARATOR.match(line)
        for line in lines[start:index]
    )
    cells = [
        TableCell(text=row[col], row=r, col=col, is_header=has_separator and r == 0)
        for r, row in enumerate(raw_rows)
        for col in range(n_cols)
    ]
    table = TableData(
        n_rows=len(raw_rows),
        n_cols=n_cols,
        cells=cells,
        header_row_count=1 if has_separator else 0,
    )
    if not has_separator:
        apply_header_flags(table)
    return table, index


class TextParser(BaseParser):
    name = "text"
    version = "0.1.0"
    file_types = (FileType.MARKDOWN, FileType.TEXT)

    def _parse(
        self, path: Path, options: ParseOptions, document_id: str
    ) -> ParsedDocument:
        file_type = (
            FileType.MARKDOWN if path.suffix.lower() == ".md" else FileType.TEXT
        )
        text, encoding, lossy = read_text(path)
        front_meta, body = _split_front_matter(text)

        errors: list[ParseError] = []
        if lossy:
            errors.append(
                ParseError(
                    stage="decode",
                    message="文件编码无法识别，已用替换字符解码，可能有个别字损坏",
                )
            )

        blocks: list[DocumentBlock] = []
        heading_path: list[str] = []
        lines = body.splitlines()
        title = str(front_meta.get("title") or "").strip()
        last_paragraph = ""
        index = 0

        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            if not stripped:
                index += 1
                continue

            heading = _HEADING.match(stripped)
            if heading:
                level = len(heading.group(1))
                text_value = heading.group(2).strip()
                heading_path = heading_path[: level - 1] + [text_value]
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(document_id, len(blocks)),
                        block_type=BlockType.HEADING,
                        text=text_value,
                        heading_level=level,
                        heading_path=list(heading_path),
                        source_location=SourceLocation(char_offset=self._offset(body, index)),
                        extra={"heading_source": "markdown"},
                    )
                )
                index += 1
                continue

            if stripped.startswith("|") and index + 1 < len(lines) and (
                _TABLE_SEPARATOR.match(lines[index + 1])
                or lines[index + 1].strip().startswith("|")
            ):
                table, next_index = _parse_table(lines, index)
                if table.n_rows > 0:
                    caption = (
                        last_paragraph
                        if last_paragraph and _TABLE_CAPTION.match(last_paragraph)
                        else ""
                    )
                    blocks.append(
                        DocumentBlock(
                            block_id=new_block_id(document_id, len(blocks)),
                            block_type=BlockType.TABLE,
                            table_data=table.model_copy(update={"caption": caption}),
                            heading_path=list(heading_path),
                            source_location=SourceLocation(char_offset=self._offset(body, index)),
                            extra={"caption": caption} if caption else {},
                        )
                    )
                    index = next_index
                    continue

            # 公文式中文编号标题：「一、资助标准与名额」。
            #
            # 这条判据原来只给了 docx 和 pdf，md/txt 漏了。漏掉的后果不是"少一个
            # 标题"，而是**整篇退化成 1 个 chunk**：实测 36_奖学金 那份 md，
            # 老链路按 600 字硬切还有 3 块，新链路"更结构化"反而只剩 1 块，
            # 「国家奖学金 8000 元」被埋进整篇里，问金额时它排到了励志奖学金后面。
            # 三种格式共用同一条判据，切分粒度才不会随文件格式漂移。
            cn_level = cn_heading_level(stripped)
            if cn_level is not None:
                heading_path = heading_path[: cn_level - 1] + [stripped]
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(document_id, len(blocks)),
                        block_type=BlockType.HEADING,
                        text=stripped,
                        heading_level=cn_level,
                        heading_path=list(heading_path),
                        source_location=SourceLocation(
                            char_offset=self._offset(body, index)
                        ),
                        extra={"heading_source": "cn_number"},
                    )
                )
                index += 1
                continue

            if _BULLET.match(line):
                items: list[str] = []
                start_index = index
                while index < len(lines) and _BULLET.match(lines[index]):
                    items.append(_BULLET.sub("", lines[index]).strip())
                    index += 1
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(document_id, len(blocks)),
                        block_type=BlockType.LIST,
                        text="\n".join(f"- {item}" for item in items if item),
                        heading_path=list(heading_path),
                        source_location=SourceLocation(
                            char_offset=self._offset(body, start_index)
                        ),
                        extra={"ordered": bool(re.match(r"^\s*\d+[.)]", line))},
                    )
                )
                continue

            # 普通段落：连续非空、非标题、非表格、非列表的行
            start_index = index
            paragraph: list[str] = []
            while index < len(lines):
                current = lines[index]
                if not current.strip():
                    break
                if _HEADING.match(current.strip()):
                    break
                if current.strip().startswith("|"):
                    break
                if _BULLET.match(current):
                    break
                # 段落中途撞到「二、申请条件」也要断开：md 里经常整篇没有空行，
                # 不在这里断的话，下一节的标题会被吞进上一节的正文里。
                if cn_heading_level(current.strip()) is not None:
                    break
                paragraph.append(current.strip())
                index += 1
            value = "\n".join(paragraph).strip()
            if value:
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(document_id, len(blocks)),
                        block_type=BlockType.PARAGRAPH,
                        text=value,
                        heading_path=list(heading_path),
                        source_location=SourceLocation(
                            char_offset=self._offset(body, start_index)
                        ),
                    )
                )
                last_paragraph = value

        # 标题层级归一化：`##` 而没有 `#` 的文档里，原来的就地切片会把
        # 两个同级标题拼成父子关系。详见 normalize_heading_paths 的注释。
        normalize_heading_paths(blocks)

        metadata = DocumentMetadata(
            encoding=encoding,
            **{
                key: value
                for key, value in front_meta.items()
                if key in DocumentMetadata.model_fields and value not in (None, "")
            },
        )

        if not blocks:
            return ParsedDocument(
                document_id=document_id,
                file_name=path.name,
                file_type=file_type,
                title=title or path.stem,
                metadata=metadata,
                blocks=[],
                parser_name=self.name,
                parser_version=self.version,
                parse_status=ParseStatus.EMPTY,
                errors=errors
                + [ParseError(stage="extract", message="文件里没有提取到任何正文")],
            )

        return ParsedDocument(
            document_id=document_id,
            file_name=path.name,
            file_type=file_type,
            title=title or path.stem,
            metadata=metadata,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            parse_status=ParseStatus.PARTIAL if errors else ParseStatus.COMPLETED,
            errors=errors,
        )

    @staticmethod
    def _offset(body: str, line_index: int) -> int:
        """该行在全文中的字符下标，供前端高亮。按行累加，不做全文 index()——
        同一行文字出现多次时 index() 会定位到第一处。"""
        return sum(len(line) + 1 for line in body.splitlines()[:line_index])
