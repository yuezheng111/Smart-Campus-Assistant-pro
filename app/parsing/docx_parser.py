"""DOCX 解析器。

要解决的两个现有缺陷
--------------------
1. **表格整段丢失**。现在是 `"\n".join(p.text for p in document.paragraphs)`，
   `document.tables` 从来没被读过。实测 `tables-with-incomplete-rows.docx`
   有 6 张表、43 个非空单元格，旧代码只输出 75 个字符。
2. **段落与表格的顺序丢失**。`.paragraphs` 和 `.tables` 是两个独立视图，
   先取完段落再取表格，正文里"段落→表格→段落"的交错关系就没了。

所以这里按 **body 的 XML 子节点顺序**遍历，遇到 `w:p` 出段落、遇到 `w:tbl`
出表格；`w:sdt`（内容控件）要递归钻进去，否则公文里用控件包的正文会被整段漏掉。

合并单元格
----------
按 OOXML 的原始语义解析：`gridSpan` 是横向跨度，`vMerge=restart/continue`
是纵向合并。刻意不用 `table.rows[i].cells` —— 那个接口会把合并区域里的
同一个单元格重复返回多次，直接用会造出一堆重复内容。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.logging_config import get_logger
from app.parsing._heuristics import (
    apply_header_flags,
    cn_heading_level,
    normalize_heading_paths,
)
from app.parsing.base import BaseParser, ParseOptions, relative_path
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

# "Heading 1" / "标题 1" / "heading1" 都要认
_STYLE_LEVEL = re.compile(r"(?:heading|标题)\s*([1-9])", re.IGNORECASE)
# 公文里手写的章标题（不是 Word 样式）："第一章 总则"、"第二十三条 ..."
# 用 .{0,30} 而不是 \S{0,30}：真实公文常写成「第一章　总　则」，
# 中间有全角空格，用 \S 会漏掉这一类最常见的章标题。
_MANUAL_HEADING = re.compile(
    r"^第[一二三四五六七八九十百零]+(章|节).{0,30}$"
)
# 表格标题：表 1 / 附表2 / 表格三
_TABLE_CAPTION = re.compile(r"^\s*(表|附表|表格)\s*[\d一二三四五六七八九十]+")


def _iter_body_children(body: Any) -> Iterator[Any]:
    """按 XML 顺序产出 body 的直接子节点，穿透 w:sdt 内容控件。"""
    for child in body.iterchildren():
        if child.tag == qn("w:sdt"):
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                yield from _iter_body_children(content)
            continue
        yield child


def _paragraph_heading_level(paragraph: Paragraph) -> int | None:
    """从样式或 outlineLvl 读标题级别。"""
    style_name = ""
    try:
        style_name = paragraph.style.name or ""
    except Exception:  # 样式表损坏时不应影响正文提取
        style_name = ""
    match = _STYLE_LEVEL.search(style_name)
    if match:
        return int(match.group(1))

    ppr = paragraph._p.pPr
    if ppr is not None:
        outline = ppr.find(qn("w:outlineLvl"))
        if outline is not None:
            raw = outline.get(qn("w:val"))
            if raw is not None and str(raw).isdigit():
                return min(int(raw) + 1, 6)
    return None


def _is_list_paragraph(paragraph: Paragraph) -> bool:
    style_name = ""
    try:
        style_name = (paragraph.style.name or "").lower()
    except Exception:
        style_name = ""
    if "list" in style_name or "列表" in style_name:
        return True
    ppr = paragraph._p.pPr
    return ppr is not None and ppr.numPr is not None


def _has_styled_headings(document: Document) -> bool:
    """整份文档里有没有用 Word 标题样式。

    保留这个探测是为了让 `heading_source` 能标明「这个标题是猜的」。
    注意它**不再决定要不要启用手写标题识别**：真实公文经常是
    「第一章」用普通段落、后面的章用 Heading 1 —— 按整份文档开关的话，
    没套样式的那几章会集体漏掉，heading_path 跟着全错。
    """
    for paragraph in document.paragraphs:
        if _paragraph_heading_level(paragraph) is not None:
            return True
    return False


def _cell_text(tc: Any) -> str:
    """单元格文字：段内按行、段间按换行，保留手工换行 w:br。"""
    paragraphs: list[str] = []
    for p in tc.findall(qn("w:p")):
        pieces: list[str] = []
        for node in p.iter():
            if node.tag == qn("w:t") and node.text:
                pieces.append(node.text)
            elif node.tag == qn("w:br"):
                pieces.append("\n")
            elif node.tag == qn("w:tab"):
                pieces.append("\t")
        text = "".join(pieces).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _guess_header_rows(table: TableData) -> int:
    """保留本地别名，实际逻辑在 _heuristics，三种格式共用一套判据。"""
    return apply_header_flags(table)


def _table_from_tbl(tbl: Any, caption: str) -> TableData:
    """把 w:tbl 解析成 TableData（含 gridSpan / vMerge）。"""
    trs = tbl.findall(qn("w:tr"))
    logical_rows: list[list[tuple[Any, int, int | None]]] = []
    total_cols = 0

    for tr in trs:
        row_cells: list[tuple[Any, int, int | None]] = []
        width = 0
        for tc in tr.findall(qn("w:tc")):
            tcpr = tc.find(qn("w:tcPr"))
            span = 1
            vmerge: int | None = None  # None=不参与纵向合并, 1=restart, 0=continue
            if tcpr is not None:
                grid_span = tcpr.find(qn("w:gridSpan"))
                if grid_span is not None:
                    raw = grid_span.get(qn("w:val"))
                    if raw and str(raw).isdigit():
                        span = max(1, int(raw))
                vm = tcpr.find(qn("w:vMerge"))
                if vm is not None:
                    val = vm.get(qn("w:val"))
                    # 没有 val 属性 = continue（OOXML 的默认行为）
                    vmerge = 1 if val == "restart" else 0
            row_cells.append((tc, span, vmerge))
            width += span
        logical_rows.append(row_cells)
        total_cols = max(total_cols, width)

    n_rows = len(logical_rows)
    if n_rows == 0 or total_cols == 0:
        return TableData(n_rows=0, n_cols=0, caption=caption)

    cells: list[TableCell] = []
    open_vmerge: dict[int, TableCell] = {}
    placed: set[tuple[int, int]] = set()

    for row_index, row_cells in enumerate(logical_rows):
        col = 0
        covered: set[int] = set()
        for tc, span, vmerge in row_cells:
            text = _cell_text(tc)

            # 残缺表格（行里缺 tc）会让游标和实际列对不上。遇到已占用的位置
            # 就顺延到本行第一个空位，而不是硬塞 —— 硬塞出来的是重叠单元格，
            # TableData 会据此判定结构非法，**整张表**都进不了库。
            # 宁可让某一格错位，也不能丢掉整张表。
            while col < total_cols and (row_index, col) in placed:
                col += 1
            if col >= total_cols:
                log.warning("表格列数不足，丢弃多余单元格（第 %d 行）：%r",
                            row_index, text[:20])
                continue
            span = max(1, min(span, total_cols - col))

            if vmerge == 0:  # continue：并入上方的起点单元格
                origin = open_vmerge.get(col)
                # 冲突检测要按**起点单元格的实际覆盖范围**算，不能只看当前列：
                # 畸形表格里延续格会和起点错开一列（起点 col0-1，延续格在 col1-2），
                # 只检查 (row, col) 就会放行，然后 rowspan+1 把中间行已经放好的
                # 单元格盖掉，造出重叠 → 整张表被判非法。
                conflict = False
                if origin is not None:
                    conflict = any(
                        (row_index, c) in placed
                        for c in range(origin.col, origin.col + origin.colspan)
                    )
                if origin is not None and not conflict:
                    origin.rowspan += 1
                    for c in range(col, col + span):
                        placed.add((row_index, c))
                    covered.update(range(col, col + span))
                    col += span
                    continue
                # 找不到起点：这一格通常本来就没有文字（合并区域的延续格）。
                # 直接占位、不生成新单元格，避免造出重复内容。
                log.warning("vMerge 找不到起点，按空位跳过（第 %d 行第 %d 列）",
                            row_index, col)
                for c in range(col, col + span):
                    placed.add((row_index, c))
                covered.update(range(col, col + span))
                col += span
                continue

            cell = TableCell(
                text=text,
                row=row_index,
                col=col,
                colspan=span,
                rowspan=1,
            )
            cells.append(cell)
            for c in range(col, col + span):
                placed.add((row_index, c))
            if vmerge == 1:
                for c in range(col, col + span):
                    open_vmerge[c] = cell
            else:
                for c in range(col, col + span):
                    open_vmerge.pop(c, None)
            covered.update(range(col, col + span))
            col += span

        # 本行没有覆盖到的列，纵向合并应当断开
        for c in list(open_vmerge):
            if c not in covered:
                del open_vmerge[c]

    table = TableData(
        n_rows=n_rows,
        n_cols=total_cols,
        cells=cells,
        caption=caption,
        header_row_count=0,
    )
    table.header_row_count = _guess_header_rows(table)
    return table


class DOCXParser(BaseParser):
    name = "docx"
    version = "0.1.0"
    file_types = (FileType.DOCX,)

    def _parse(
        self, path: Path, options: ParseOptions, document_id: str
    ) -> ParsedDocument:
        document = Document(str(path))
        blocks: list[DocumentBlock] = []
        errors: list[ParseError] = []
        heading_path: list[str] = []
        has_styled = _has_styled_headings(document)
        title = ""
        last_paragraph_text = ""
        paragraph_index = -1
        table_index = -1

        for child in _iter_body_children(document.element.body):
            if child.tag == qn("w:p"):
                paragraph_index += 1
                paragraph = Paragraph(child, document)
                text = (paragraph.text or "").strip()

                level = _paragraph_heading_level(paragraph)
                source = "style"
                if level is None and text and _MANUAL_HEADING.match(text):
                    # 与样式标题**并存**，不是二选一：公文里两者混用是常态。
                    # 正则很窄（只认「第X章/第X节」且不超过 30 字），
                    # 「第一条 为维护……」这类条款不会被误判成标题。
                    level = 1
                    source = "heuristic" if has_styled else "heuristic_only"
                if level is None and text:
                    # 没有样式也没有「第X章」时，退回中文编号（「一、资助标准」）。
                    # 不加这一条，绝大多数公文 docx 会把整篇挤成一个 chunk ——
                    # 实测 `38_助学金_...通知.docx` 21 个块只切出 1 块。
                    # 判据与 pdf 共用 cn_heading_level，两种格式的切分粒度才对得上。
                    cn_level = cn_heading_level(text)
                    if cn_level is not None:
                        level = cn_level
                        source = "heuristic" if has_styled else "heuristic_only"

                image_blocks: list[DocumentBlock] = []

                if level is not None and text:
                    heading_path = heading_path[: level - 1] + [text]
                    # 文档标题只认"不像编号标题"的那个：否则「一、资助标准与名额」
                    # 会被当成整份文档的标题，学生看到的来源就全错了。
                    if not title and cn_heading_level(text) is None:
                        title = text
                    blocks.append(
                        DocumentBlock(
                            block_id=new_block_id(document_id, len(blocks)),
                            block_type=BlockType.HEADING,
                            text=text,
                            heading_level=level,
                            heading_path=list(heading_path),
                            source_location=SourceLocation(paragraph_index=paragraph_index),
                            extra={"heading_source": source},
                        )
                    )
                elif text:
                    block_type = (
                        BlockType.LIST if _is_list_paragraph(paragraph) else BlockType.PARAGRAPH
                    )
                    blocks.append(
                        DocumentBlock(
                            block_id=new_block_id(document_id, len(blocks)),
                            block_type=block_type,
                            text=text,
                            heading_path=list(heading_path),
                            source_location=SourceLocation(paragraph_index=paragraph_index),
                        )
                    )

                # 图片块必须**排在文本块之后**再生成 ID：两处都用
                # len(blocks) 作序号，同时计算就会撞出重复 block_id，
                # 而 block_id 重复会被 ParsedDocument 直接判为非法，
                # 整份文件解析失败。
                if options.include_images:
                    image_blocks = self._images_from_paragraph(
                        paragraph, options, document_id, heading_path,
                        paragraph_index, len(blocks),
                    )
                    blocks.extend(image_blocks)
                if text:
                    last_paragraph_text = text

            elif child.tag == qn("w:tbl"):
                table_index += 1
                caption = (
                    last_paragraph_text
                    if last_paragraph_text and _TABLE_CAPTION.match(last_paragraph_text)
                    else ""
                )
                try:
                    table_data = _table_from_tbl(child, caption)
                except Exception as exc:
                    # 一张表解析失败不该让整份文件失败：记下来，继续下一个块
                    log.warning("表格解析失败 | %s | 第 %d 张表：%s", path.name, table_index, exc)
                    errors.append(
                        ParseError(
                            stage="table",
                            message=f"第 {table_index + 1} 张表解析失败：{type(exc).__name__}: {exc}",
                            location=SourceLocation(table_index=table_index),
                        )
                    )
                    continue
                if table_data.n_rows == 0 or table_data.n_cols == 0:
                    continue
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(document_id, len(blocks)),
                        block_type=BlockType.TABLE,
                        table_data=table_data,
                        heading_path=list(heading_path),
                        source_location=SourceLocation(table_index=table_index),
                        extra={"caption": caption} if caption else {},
                    )
                )

        # 标题层级归一化：文档没有 Heading1、只有 Heading2 起步时，
        # 就地切片会让两个同级标题叠成父子。详见 normalize_heading_paths。
        normalize_heading_paths(blocks)

        if not blocks:
            return ParsedDocument(
                document_id=document_id,
                file_name=path.name,
                file_type=FileType.DOCX,
                title=title or path.stem,
                metadata=DocumentMetadata(paragraph_count=paragraph_index + 1),
                blocks=[],
                parser_name=self.name,
                parser_version=self.version,
                parse_status=ParseStatus.EMPTY,
                errors=errors
                + [ParseError(stage="extract", message="文档里没有提取到任何正文或表格")],
            )

        status = ParseStatus.PARTIAL if errors else ParseStatus.COMPLETED
        return ParsedDocument(
            document_id=document_id,
            file_name=path.name,
            file_type=FileType.DOCX,
            title=title or path.stem,
            metadata=DocumentMetadata(
                paragraph_count=paragraph_index + 1,
                table_count=table_index + 1,
            ),
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            parse_status=status,
            errors=errors,
        )

    # ---- 图片 ----

    def _images_from_paragraph(
        self,
        paragraph: Paragraph,
        options: ParseOptions,
        document_id: str,
        heading_path: list[str],
        paragraph_index: int,
        block_offset: int,
    ) -> list[DocumentBlock]:
        if not options.include_images:
            return []
        blocks: list[DocumentBlock] = []
        for blip in paragraph._p.iter(qn("a:blip")):
            rel_id = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
            if not rel_id:
                continue
            part = None
            try:
                part = paragraph.part.related_parts.get(rel_id)
            except Exception:
                part = None

            partname = str(getattr(part, "partname", "") or "")
            if part is not None and options.assets_dir is not None:
                image_path = self.save_asset(
                    part.blob, Path(partname).name or "image.bin", options, document_id
                )
                embedded = False
            else:
                image_ref = partname or rel_id
                image_path = f"embedded://docx/{image_ref.lstrip('/')}"
                embedded = True

            blocks.append(
                DocumentBlock(
                    block_id=new_block_id(document_id, block_offset + len(blocks)),
                    block_type=BlockType.IMAGE,
                    image_path=image_path,
                    heading_path=list(heading_path),
                    source_location=SourceLocation(paragraph_index=paragraph_index),
                    extra={"embedded": embedded, "relationship_id": rel_id},
                )
            )
        return blocks
