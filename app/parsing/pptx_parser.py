"""PPTX 解析器。

组织单位是**一张幻灯片**，不是整个文件。这一点和 PDF（按页）类似，
但 PPT 的内容天生更碎：标题、正文文本框、表格、图片、图表、备注，
每一块都要能回溯到"第几页第几个形状"。

几个刻意的取舍
--------------
- 标题单独成 heading，heading_path = [演示文稿标题, 本页标题]；
  一张幻灯片因此成为一个能独立成 Chunk 的语义单元。
- 文本框**整框一个块**，不按段落切：一页里 8 个短段落切成 8 个块，
  检索时会各自召回、互相挤占名额，反而丢上下文。
- 演讲者备注单独成 NOTE 块，**不混进正文**。备注里经常有"这里要强调"
  这种写给自己看的话，混进正文会污染检索结果；但丢掉又可惜（它常含
  正文没写的解释），所以留成独立类型，阶段五再决定索不索引。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from app.logging_config import get_logger
from app.parsing._heuristics import apply_header_flags
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


def _shape_sort_key(shape: Any) -> tuple[int, int]:
    """按版面位置排序：从上到下、从左到右。

    `slide.shapes` 给的是 z-order（叠放顺序），不是阅读顺序。直接按它输出，
    会出现"右下角的图注排在标题前面"。None 当作 0，避免排序时崩。
    """
    top = getattr(shape, "top", None)
    left = getattr(shape, "left", None)
    return (int(top) if top is not None else 0, int(left) if left is not None else 0)


def _iter_shapes(shapes: Any) -> Iterator[Any]:
    """递归展开组合形状。组合里的文本框不展开就会整块漏掉。"""
    for shape in shapes:
        try:
            is_group = shape.shape_type == MSO_SHAPE_TYPE.GROUP
        except Exception:
            is_group = False
        if is_group:
            yield from _iter_shapes(shape.shapes)
        else:
            yield shape


def _text_frame_text(shape: Any) -> str:
    paragraphs: list[str] = []
    for paragraph in shape.text_frame.paragraphs:
        text = (paragraph.text or "").strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _is_bullet_frame(shape: Any) -> bool:
    """整框都是项目符号时按 list 处理。

    判据用「所有非空段落的 level 都 > 0」或文本自带项目符号。
    不做逐段判断，是因为一个块只能有一种类型；把同一框拆成多个块
    又会让这一页的内容在检索时被打散。
    """
    paragraphs = [p for p in shape.text_frame.paragraphs if (p.text or "").strip()]
    if not paragraphs:
        return False
    if all((p.level or 0) > 0 for p in paragraphs):
        return True
    first = (paragraphs[0].text or "").strip()
    return first.startswith(("•", "·", "●", "-", "－"))


def _table_from_graphic_frame(shape: Any) -> TableData:
    table = shape.table
    n_rows = len(table.rows)
    n_cols = len(table.columns)
    cells: list[TableCell] = []

    for r in range(n_rows):
        for c in range(n_cols):
            cell = table.cell(r, c)
            try:
                if cell.is_spanned:
                    continue  # 被合并覆盖的格子，内容在起点格里
                rowspan = cell.span_height if cell.is_merge_origin else 1
                colspan = cell.span_width if cell.is_merge_origin else 1
            except Exception:
                rowspan, colspan = 1, 1
            cells.append(
                TableCell(
                    text=(cell.text or "").strip(),
                    row=r,
                    col=c,
                    rowspan=max(1, rowspan),
                    colspan=max(1, colspan),
                )
            )

    table_data = TableData(
        n_rows=n_rows,
        n_cols=n_cols,
        cells=cells,
        header_row_count=0,
    )
    table_data.header_row_count = apply_header_flags(table_data)
    return table_data


class PPTXParser(BaseParser):
    name = "pptx"
    version = "0.1.0"
    file_types = (FileType.PPTX,)

    def _parse(
        self, path: Path, options: ParseOptions, document_id: str
    ) -> ParsedDocument:
        presentation = Presentation(str(path))
        blocks: list[DocumentBlock] = []
        errors: list[ParseError] = []
        deck_title = ""

        for slide_number, slide in enumerate(presentation.slides, start=1):
            title_shape = None
            title_text = ""
            try:
                title_shape = slide.shapes.title
                if title_shape is not None:
                    title_text = (title_shape.text or "").strip()
            except Exception:
                title_shape, title_text = None, ""

            if not deck_title and title_text:
                deck_title = title_text

            if title_text:
                heading_path = (
                    [deck_title, title_text] if deck_title and title_text != deck_title
                    else [title_text]
                )
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(document_id, len(blocks)),
                        block_type=BlockType.HEADING,
                        text=title_text,
                        heading_level=2 if title_text != deck_title else 1,
                        heading_path=heading_path,
                        source_location=SourceLocation(slide_number=slide_number),
                    )
                )
            else:
                heading_path = [deck_title] if deck_title else []

            title_shape_id = getattr(title_shape, "shape_id", None)
            shapes = [
                shape
                for shape in _iter_shapes(slide.shapes)
                if getattr(shape, "shape_id", None) != title_shape_id
            ]
            for shape in sorted(shapes, key=_shape_sort_key):
                block = self._block_from_shape(
                    shape, options, document_id, slide_number, heading_path, len(blocks),
                    errors,
                )
                if block is not None:
                    blocks.append(block)

            notes = self._notes_text(slide)
            if notes:
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(document_id, len(blocks)),
                        block_type=BlockType.NOTE,
                        text=notes,
                        heading_path=list(heading_path),
                        source_location=SourceLocation(slide_number=slide_number),
                        extra={"notes": True},
                    )
                )

        metadata = DocumentMetadata(slide_count=len(presentation.slides))

        if not blocks:
            return ParsedDocument(
                document_id=document_id,
                file_name=path.name,
                file_type=FileType.PPTX,
                title=deck_title or path.stem,
                metadata=metadata,
                blocks=[],
                parser_name=self.name,
                parser_version=self.version,
                parse_status=ParseStatus.EMPTY,
                errors=errors
                + [ParseError(stage="extract", message="幻灯片里没有提取到任何内容")],
            )

        return ParsedDocument(
            document_id=document_id,
            file_name=path.name,
            file_type=FileType.PPTX,
            title=deck_title or path.stem,
            metadata=metadata,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            parse_status=ParseStatus.PARTIAL if errors else ParseStatus.COMPLETED,
            errors=errors,
        )

    # ---- 单个形状 ----

    def _block_from_shape(
        self,
        shape: Any,
        options: ParseOptions,
        document_id: str,
        slide_number: int,
        heading_path: list[str],
        block_index: int,
        errors: list[ParseError],
    ) -> DocumentBlock | None:
        location = SourceLocation(slide_number=slide_number)

        # 表格
        try:
            has_table = bool(getattr(shape, "has_table", False))
        except Exception:
            has_table = False
        if has_table:
            try:
                table_data = _table_from_graphic_frame(shape)
            except Exception as exc:
                log.warning("幻灯片表格解析失败 | slide=%d | %s", slide_number, exc)
                errors.append(
                    ParseError(
                        stage="table",
                        message=f"第 {slide_number} 页的表格解析失败：{type(exc).__name__}: {exc}",
                        location=location,
                    )
                )
                return None
            if table_data.n_rows == 0 or table_data.n_cols == 0:
                return None
            return DocumentBlock(
                block_id=new_block_id(document_id, block_index),
                block_type=BlockType.TABLE,
                table_data=table_data,
                heading_path=list(heading_path),
                source_location=location,
            )

        # 图表
        try:
            has_chart = bool(getattr(shape, "has_chart", False))
        except Exception:
            has_chart = False
        if has_chart:
            extra: dict[str, Any] = {}
            try:
                chart = shape.chart
                extra["chart_type"] = str(chart.chart_type)
                try:
                    extra["series"] = [str(series.name) for series in chart.series]
                except Exception as exc:
                    # 某些图表类型（如 treemap/sunburst）python-pptx 解析不了 series。
                    # 只丢元数据，CHART 块本身还在，不该把整份文件降级成 partial。
                    log.warning("图表 series 读取失败 | slide=%d | %s", slide_number, exc)
                    extra["series_error"] = str(exc)[:120]
                try:
                    extra["chart_title"] = (chart.chart_title.text_frame.text or "").strip()
                except Exception:
                    extra["chart_title"] = ""
            except Exception as exc:
                log.warning("图表元数据读取失败 | slide=%d | %s", slide_number, exc)
                extra["chart_error"] = str(exc)[:120]
            return DocumentBlock(
                block_id=new_block_id(document_id, block_index),
                block_type=BlockType.CHART,
                heading_path=list(heading_path),
                source_location=location,
                extra=extra or {"chart": True},
            )

        # 图片
        try:
            is_picture = shape.shape_type == MSO_SHAPE_TYPE.PICTURE
        except Exception:
            is_picture = False
        if is_picture and options.include_images:
            image_path = ""
            embedded = True
            try:
                image = shape.image
                if options.assets_dir is not None:
                    image_path = self.save_asset(
                        image.blob,
                        image.filename or f"slide{slide_number}.png",
                        options,
                        document_id,
                    )
                    embedded = False
            except Exception as exc:
                errors.append(
                    ParseError(
                        stage="image",
                        message=f"第 {slide_number} 页图片读取失败：{exc}",
                        location=location,
                    )
                )
            if not image_path:
                image_path = f"embedded://pptx/slide{slide_number}/shape{getattr(shape, 'shape_id', 0)}"
            return DocumentBlock(
                block_id=new_block_id(document_id, block_index),
                block_type=BlockType.IMAGE,
                image_path=image_path,
                heading_path=list(heading_path),
                source_location=location,
                extra={"embedded": embedded, "shape_name": getattr(shape, "name", "")},
            )

        # 文本框
        try:
            has_text_frame = bool(getattr(shape, "has_text_frame", False))
        except Exception:
            has_text_frame = False
        if has_text_frame:
            text = _text_frame_text(shape)
            if not text:
                return None
            return DocumentBlock(
                block_id=new_block_id(document_id, block_index),
                block_type=BlockType.LIST if _is_bullet_frame(shape) else BlockType.PARAGRAPH,
                text=text,
                heading_path=list(heading_path),
                source_location=location,
                extra={"shape_name": getattr(shape, "name", "")},
            )

        return None

    def _notes_text(self, slide: Any) -> str:
        try:
            if not slide.has_notes_slide:
                return ""
            frame = slide.notes_slide.notes_text_frame
            return (frame.text or "").strip() if frame is not None else ""
        except Exception as exc:
            log.warning("读取演讲者备注失败：%s", exc)
            return ""
