"""PDF 解析器。

为什么是两个库
--------------
- **pdfplumber** 能拿到每个字符的字号、坐标，以及表格的边框和单元格。
  标题识别、表格结构、页内阅读顺序都靠它。
- **pypdf** 能拿到页面内嵌图片的**原始字节**。pdfplumber 只给图片位置，
  不给你 blob，想 OCR 就必须用 pypdf 抠出来。

两个库各开一次文件，页码一一对应（都从 0 开始），成本可以接受。

现在的实现边界（如实说明）
--------------------------
已做：
    · 逐页提取，页码写进 SourceLocation
    · 按字号/加粗识别标题层级，建立 heading_path
    · 表格用 find_tables 提取，**表格区域内的文字从正文里剔除**，避免同一段
      内容既在正文 chunk 又在表格 chunk 里出现两遍
    · 页内按 y 坐标排序输出，保持「表格前文字 → 表格 → 表格后文字」的顺序
    · 图片按页抠出（可落盘），记录页号
    · 单页失败不影响其它页；整份没有文本层 → EMPTY 并提示需要 OCR

未做（阶段四/扩展点）：
    · 扫描件 OCR
    · 跨页表格合并（宁可分成两张，也不猜）
    · 双栏阅读顺序（现在按 y 排，双栏会串行）
    · 页眉页脚剔除（阶段四做，且只标记不删）
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader

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

# 行内字符的基线容差（pt）。同一行的字会有一两点的上下浮动，
# 用 0 分组会把一行拆成十几个"行"。
_LINE_TOLERANCE = 3.0
# 超过这个字数的行不可能是标题
_MAX_HEADING_CHARS = 60
# 标题字号至少要比正文大这么多倍
_HEADING_SIZE_RATIO = 1.15
# 表格标题最多距离表格这么远（pt）
_CAPTION_DISTANCE = 40.0


def _round_size(value: float) -> float:
    return round(float(value), 1)


def _line_text(chars: list[dict[str, Any]]) -> str:
    """把一行字符拼成文本，**按间距补回空格**。

    不能直接 `"".join(text)`：很多 PDF 不把空格存成字符，而是用坐标间距
    表示。直接拼会得到 `TheEvolutionoftheWordProcessor` 这种连体英文 ——
    中文看不出问题，英文分词和向量化会明显变差。

    阈值取 0.25 倍字号：中文的字间距通常远小于它，不会被插空格；
    英文的单词间距一般在 0.3-0.5 倍字号，能正确切开。
    """
    parts: list[str] = []
    previous: dict[str, Any] | None = None
    for char in chars:
        if previous is not None:
            gap = float(char["x0"]) - float(previous["x1"])
            size = float(previous.get("size") or 0)
            prev_text = str(previous.get("text") or "")
            cur_text = str(char.get("text") or "")
            # 拉丁字母之间用更敏感的阈值（0.12 倍字号）：这类 PDF 常把词间距
            # 压得很小，"The" 和 "Evolution" 之间可能只差 1pt。
            # 中文不参与这个判断 —— 中文字距本来就有浮动，按这个阈值会插满空格。
            latin_pair = (
                prev_text[-1:].isascii()
                and prev_text[-1:].isalnum()
                and cur_text[:1].isascii()
                and cur_text[:1].isalnum()
            )
            threshold = 0.12 if latin_pair else 0.25
            if gap > max(0.5, size * threshold):
                parts.append(" ")
        parts.append(str(char.get("text") or ""))
        previous = char
    text = "".join(parts)
    # 连续半角空格压成一个；全角空格（\u3000）是中文排版的一部分，不能动
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _join_wrapped(left: str, right: str) -> str:
    """拼接被换行切断的两段文字。

    中文换行不加空格（PDF 里换行只是排版，不是词边界）；英文两个单词之间
    必须补一个空格，否则会拼出 `wordprocessing` 这种检索不到的假词。
    """
    if not left:
        return right
    if not right:
        return left
    if left[-1:].isascii() and left[-1:].isalnum() and right[:1].isascii() and right[:1].isalnum():
        return f"{left} {right}"
    return left + right


def _merge_wrapped_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把同一段落内被换行切断的视觉行合并成一块。

    不合并的后果很实在：一份 13 页双栏论文会产出 983 个块，每个块只有一行字。
    后面无论怎么切 Chunk，都拼不回完整的句子。

    判据用**行距的相对值**而不是固定阈值：正文行距因字号、行高设置而异，
    但同一段内部的行距是稳定的，段落之间会明显大于它。
    """
    if not lines:
        return []

    gaps = []
    for previous, current in zip(lines, lines[1:]):
        gap = current["top"] - previous["bbox"][3]
        if 0 <= gap <= 40:
            gaps.append(gap)
    typical = sorted(gaps)[len(gaps) // 2] if gaps else 0.0

    merged: list[dict[str, Any]] = []
    for line in lines:
        if merged and line.get("level") is None and merged[-1].get("level") is None:
            previous = merged[-1]
            gap = line["top"] - previous["bbox"][3]
            same_indent = abs(line["bbox"][0] - previous["bbox"][0]) <= 24
            close_enough = -2 <= gap <= typical * 1.4 + 1.5
            if same_indent and close_enough:
                previous["text"] = _join_wrapped(previous["text"], line["text"])
                previous["bbox"] = (
                    min(previous["bbox"][0], line["bbox"][0]),
                    min(previous["bbox"][1], line["bbox"][1]),
                    max(previous["bbox"][2], line["bbox"][2]),
                    max(previous["bbox"][3], line["bbox"][3]),
                )
                previous["size"] = max(previous["size"], line["size"])
                continue
        merged.append(dict(line))
    return merged


class PDFParser(BaseParser):
    name = "pdf"
    version = "0.1.0"
    file_types = (FileType.PDF,)

    def _parse(
        self, path: Path, options: ParseOptions, document_id: str
    ) -> ParsedDocument:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise RuntimeError(f"PDF 已加密，无法解析：{exc}") from exc
            if reader.is_encrypted:
                raise RuntimeError("PDF 已加密（需要密码），无法解析")

        with pdfplumber.open(str(path)) as pdf:
            page_limit = len(pdf.pages)
            if options.max_pages and page_limit > options.max_pages:
                page_limit = options.max_pages
            body_size = self._detect_body_size(pdf.pages[:page_limit])
            heading_levels = self._heading_size_levels(pdf.pages[:page_limit], body_size)

            blocks: list[DocumentBlock] = []
            errors: list[ParseError] = []
            heading_path: list[str] = []
            pages_without_text = 0

            for page_index in range(page_limit):
                page = pdf.pages[page_index]
                page_number = page_index + 1
                try:
                    page_blocks, page_heading_path = self._parse_page(
                        page,
                        page_number,
                        body_size,
                        heading_levels,
                        heading_path,
                        reader,
                        options,
                        document_id,
                        len(blocks),
                        errors,
                    )
                except Exception as exc:
                    log.warning("PDF 单页解析失败 | %s | 第 %d 页：%s", path.name, page_number, exc)
                    errors.append(
                        ParseError(
                            stage="page",
                            message=f"第 {page_number} 页解析失败：{type(exc).__name__}: {exc}",
                            location=SourceLocation(page_start=page_number),
                        )
                    )
                    continue

                if not any(block.block_type is not BlockType.IMAGE for block in page_blocks):
                    pages_without_text += 1
                heading_path = page_heading_path
                blocks.extend(page_blocks)

        metadata = DocumentMetadata(page_count=len(reader.pages))
        title = self._guess_title(blocks)
        # 标题层级归一化必须放在猜标题**之后**：这份通知没有第 1 级标题，
        # 归一化会把「一、」抬成第 1 级，先归一化就会把「一、奖励标准与名额」
        # 当成文档标题。详见 normalize_heading_paths。
        normalize_heading_paths(blocks)

        if not any(block.block_type is not BlockType.IMAGE for block in blocks):
            # 整份文件只剩图片或什么都没有：这是扫描件。**不能当作成功**，
            # 否则它会以"空文档"的形式入库，学生搜不到、管理员也看不见。
            message = (
                f"{len(reader.pages)} 页都没有可提取的文本层"
                + (f"（其中 {pages_without_text} 页只有图片）" if pages_without_text else "")
                + "，需要 OCR 才能入库"
            )
            return ParsedDocument(
                document_id=document_id,
                file_name=path.name,
                file_type=FileType.PDF,
                title=title or path.stem,
                metadata=metadata,
                blocks=blocks,
                parser_name=self.name,
                parser_version=self.version,
                parse_status=ParseStatus.EMPTY,
                errors=errors
                + [ParseError(stage="ocr", message=message, fatal=False)],
            )

        return ParsedDocument(
            document_id=document_id,
            file_name=path.name,
            file_type=FileType.PDF,
            title=title or path.stem,
            metadata=metadata,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            parse_status=ParseStatus.PARTIAL if errors else ParseStatus.COMPLETED,
            errors=errors,
        )

    # ---- 字号分析 ----

    @staticmethod
    def _detect_body_size(pages: list[Any]) -> float:
        """正文字号 = 全文字符数最多的那个字号。

        比"取中位数"可靠：一份公文里标题只有几十个字，正文有几千个字，
        众数天然落在正文上。
        """
        counter: Counter[float] = Counter()
        for page in pages:
            for char in page.chars:
                size = char.get("size")
                if size:
                    counter[_round_size(size)] += 1
        if not counter:
            return 0.0
        return counter.most_common(1)[0][0]

    @staticmethod
    def _heading_size_levels(pages: list[Any], body_size: float) -> dict[float, int]:
        """字号 → 标题级别：字号越大级别越高（1 最高）。

        只认明显大于正文的字号（>=1.15 倍）。中文公文常见 22/16/14 三档，
        映射成 1/2/3 级；超过 3 档的按顺序继续排，最多到 6。
        """
        if body_size <= 0:
            return {}
        sizes: Counter[float] = Counter()
        for page in pages:
            for char in page.chars:
                size = char.get("size")
                if size and _round_size(size) >= body_size * _HEADING_SIZE_RATIO:
                    sizes[_round_size(size)] += 1
        ordered = [size for size, count in sizes.most_common() if count >= 2]
        ordered.sort(reverse=True)
        return {size: min(index + 1, 6) for index, size in enumerate(ordered)}

    # ---- 单页 ----

    def _parse_page(
        self,
        page: Any,
        page_number: int,
        body_size: float,
        heading_levels: dict[float, int],
        heading_path: list[str],
        reader: PdfReader,
        options: ParseOptions,
        document_id: str,
        block_offset: int,
        errors: list[ParseError],
    ) -> tuple[list[DocumentBlock], list[str]]:
        blocks: list[DocumentBlock] = []
        tables = self._extract_tables(page, page_number, errors)
        table_boxes = [table_info["bbox"] for table_info in tables]

        lines = self._extract_lines(page)
        # 表格区域内的文字不再进正文：否则同一段内容会同时出现在正文 chunk
        # 和表格 chunk 里，检索时重复命中、还会把名额数字读串。
        text_lines = [
            line for line in lines
            if not self._inside_any(line["center"], table_boxes)
        ]
        # 先判标题再合并：标题行不能被并进后面的正文段落里
        for line in text_lines:
            line["level"] = self._line_heading_level(line, body_size, heading_levels)
        text_lines = _merge_wrapped_lines(text_lines)

        events: list[dict[str, Any]] = []
        for table_info in tables:
            events.append({"kind": "table", "y": table_info["bbox"][1], "payload": table_info})
        for line in text_lines:
            events.append({"kind": "text", "y": line["top"], "payload": line})
        events.sort(key=lambda event: event["y"])

        current_path = list(heading_path)

        for event in events:
            if event["kind"] == "table":
                table_info = event["payload"]
                caption = self._find_caption(text_lines, table_info["bbox"])
                table_data = table_info["table"]
                if caption:
                    table_data = table_data.model_copy(update={"caption": caption})
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(document_id, block_offset + len(blocks)),
                        block_type=BlockType.TABLE,
                        table_data=table_data,
                        heading_path=list(current_path),
                        source_location=SourceLocation(
                            page_start=page_number,
                            page_end=page_number,
                            bbox=tuple(round(v, 1) for v in table_info["bbox"]),
                        ),
                        extra={"caption": caption} if caption else {},
                    )
                )
                continue

            line = event["payload"]
            text = line["text"]
            if not text:
                continue
            level = line.get("level")
            if level is not None:
                current_path = current_path[: level - 1] + [text]
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(document_id, block_offset + len(blocks)),
                        block_type=BlockType.HEADING,
                        text=text,
                        heading_level=level,
                        heading_path=list(current_path),
                        source_location=SourceLocation(
                            page_start=page_number,
                            page_end=page_number,
                            bbox=tuple(round(v, 1) for v in line["bbox"]),
                            char_offset=line["char_offset"],
                        ),
                        extra={"font_size": line["size"], "bold": line["bold"]},
                    )
                )
            else:
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(document_id, block_offset + len(blocks)),
                        block_type=BlockType.PARAGRAPH,
                        text=text,
                        heading_path=list(current_path),
                        source_location=SourceLocation(
                            page_start=page_number,
                            page_end=page_number,
                            bbox=tuple(round(v, 1) for v in line["bbox"]),
                            char_offset=line["char_offset"],
                        ),
                    )
                )

        blocks.extend(
            self._page_images(
                reader, page_number, options, document_id,
                block_offset + len(blocks), current_path,
            )
        )
        # 记下页面高度：阶段四判断「这一块是不是在页眉/页脚区域」要用它。
        # 放在块上而不是 metadata 里，是因为一个块自己就能回答"我离页顶多远"，
        # 不需要再回头查页面表，Normalizer 也不用依赖 PDF 的页序结构。
        page_height = round(float(getattr(page, "height", 0) or 0), 1)
        if page_height > 0:
            for block in blocks:
                block.extra.setdefault("page_height", page_height)
        return blocks, current_path

    # ---- 行与标题 ----

    @staticmethod
    def _extract_lines(page: Any) -> list[dict[str, Any]]:
        """把字符按 top 聚成"视觉行"，按 y 从上到下输出。

        不用 `page.extract_text()`：它给的是字符串，字号和坐标都丢了，
        标题层级就没法判断。
        """
        chars = [char for char in page.chars if str(char.get("text") or "").strip()]
        if not chars:
            return []

        chars.sort(key=lambda char: (round(float(char["top"]), 1), float(char["x0"])))
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_top: float | None = None

        for char in chars:
            top = float(char["top"])
            if current_top is None or abs(top - current_top) <= _LINE_TOLERANCE:
                current.append(char)
                if current_top is None:
                    current_top = top
            else:
                groups.append(current)
                current = [char]
                current_top = top
        if current:
            groups.append(current)

        lines: list[dict[str, Any]] = []
        offset = 0
        for group in groups:
            group.sort(key=lambda char: float(char["x0"]))
            text = _line_text(group)
            if not text:
                continue
            sizes = [float(char.get("size") or 0) for char in group]
            average_size = _round_size(sum(sizes) / len(sizes)) if sizes else 0.0
            bold = any(
                "bold" in str(char.get("fontname") or "").lower()
                or "black" in str(char.get("fontname") or "").lower()
                for char in group
            )
            lines.append(
                {
                    "text": text,
                    "top": min(float(char["top"]) for char in group),
                    "size": average_size,
                    "bold": bold,
                    "bbox": (
                        min(float(char["x0"]) for char in group),
                        min(float(char["top"]) for char in group),
                        max(float(char["x1"]) for char in group),
                        max(float(char["bottom"]) for char in group),
                    ),
                    "center": (
                        (min(float(char["x0"]) for char in group) + max(float(char["x1"]) for char in group)) / 2,
                        (min(float(char["top"]) for char in group) + max(float(char["bottom"]) for char in group)) / 2,
                    ),
                    "char_offset": offset,
                }
            )
            offset += len(text) + 1
        return lines

    @staticmethod
    def _line_heading_level(
        line: dict[str, Any], body_size: float, heading_levels: dict[float, int]
    ) -> int | None:
        text = line["text"]
        if len(text) > _MAX_HEADING_CHARS:
            return None
        size = _round_size(line["size"])
        if size in heading_levels:
            return heading_levels[size]
        # 字号没区分度时（大量公文全文一个字号），退回中文编号启发式。
        # 只认「第X章/节」和「一、」，不认「第一条 为……」——
        # 条款正文如果被当成标题，heading_path 会被几十条条款撑爆。
        cn_level = cn_heading_level(text)
        if cn_level is not None:
            return cn_level
        # 加粗但字号只比正文大一点点：公文里的小标题常这样
        if (
            line["bold"]
            and body_size > 0
            and size >= body_size * 1.02
            and len(text) <= 40
            and not text.endswith(("。", "；", "，", "："))
        ):
            return 6  # 最低一级，避免和真正的章标题抢层级
        return None

    # ---- 表格 ----

    def _extract_tables(
        self, page: Any, page_number: int, errors: list[ParseError]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        # 先看这页有没有表格线。find_tables 是全页扫描，对纯文字页是纯浪费 ——
        # 实测一份 33 页的公文因此要跑 31 秒，而它一页表都没有。
        # 没有线条/矩形时，pdfplumber 的默认策略本来也找不到表格，跳过等价。
        try:
            if not (page.lines or page.rects or page.curves):
                return results
        except Exception:
            pass

        try:
            found = page.find_tables()
        except Exception as exc:
            log.warning("PDF 表格检测失败 | 第 %d 页：%s", page_number, exc)
            errors.append(
                ParseError(
                    stage="table",
                    message=f"第 {page_number} 页表格检测失败：{exc}",
                    location=SourceLocation(page_start=page_number),
                )
            )
            return results

        for order, table in enumerate(found):
            try:
                raw = table.extract()
            except Exception as exc:
                errors.append(
                    ParseError(
                        stage="table",
                        message=f"第 {page_number} 页第 {order + 1} 张表提取失败：{exc}",
                        location=SourceLocation(page_start=page_number),
                    )
                )
                continue
            if not raw:
                continue

            n_rows = len(raw)
            n_cols = max(len(row) for row in raw)
            if n_cols == 0:
                continue
            cells = []
            for r, row in enumerate(raw):
                for c in range(n_cols):
                    value = row[c] if c < len(row) else None
                    cells.append(
                        TableCell(
                            text=str(value or "").replace("\n", " ").strip(),
                            row=r,
                            col=c,
                        )
                    )
            table_data = TableData(n_rows=n_rows, n_cols=n_cols, cells=cells, header_row_count=0)
            apply_header_flags(table_data)
            results.append({"table": table_data, "bbox": table.bbox})
        return results

    @staticmethod
    def _inside_any(point: tuple[float, float], boxes: list[Any]) -> bool:
        x, y = point
        for box in boxes:
            x0, top, x1, bottom = box
            if x0 <= x <= x1 and top <= y <= bottom:
                return True
        return False

    @staticmethod
    def _find_caption(lines: list[dict[str, Any]], bbox: Any) -> str:
        """表格正上方、距离 40pt 以内、以「表 X」开头的一行，作为表标题。"""
        import re

        pattern = re.compile(r"^\s*(表|附表|表格)\s*[\d一二三四五六七八九十]+")
        candidates = [
            line for line in lines
            if line["bbox"][3] <= bbox[1] and bbox[1] - line["bbox"][3] <= _CAPTION_DISTANCE
        ]
        if not candidates:
            return ""
        nearest = max(candidates, key=lambda line: line["bbox"][3])
        return nearest["text"] if pattern.match(nearest["text"]) else ""

    # ---- 图片 ----

    def _page_images(
        self,
        reader: PdfReader,
        page_number: int,
        options: ParseOptions,
        document_id: str,
        block_offset: int,
        heading_path: list[str],
    ) -> list[DocumentBlock]:
        if not options.include_images:
            return []
        try:
            page = reader.pages[page_number - 1]
            images = list(page.images)
        except Exception as exc:
            log.warning("读取 PDF 图片失败 | 第 %d 页：%s", page_number, exc)
            return []

        blocks: list[DocumentBlock] = []
        for index, image in enumerate(images):
            name = str(getattr(image, "name", "") or f"image{index + 1}.png")
            data = getattr(image, "data", None)
            if data and options.assets_dir is not None:
                image_path = self.save_asset(
                    data, f"p{page_number}_{Path(name).name}", options, document_id
                )
                embedded = False
            else:
                image_path = f"embedded://pdf/page{page_number}/{Path(name).name}"
                embedded = True
            blocks.append(
                DocumentBlock(
                    block_id=new_block_id(document_id, block_offset + len(blocks)),
                    block_type=BlockType.IMAGE,
                    image_path=image_path,
                    heading_path=list(heading_path),
                    source_location=SourceLocation(page_start=page_number, page_end=page_number),
                    extra={"embedded": embedded, "image_index": index},
                )
            )
        return blocks

    # ---- 标题 ----

    @staticmethod
    def _guess_title(blocks: list[DocumentBlock]) -> str:
        """第一个一级标题就是文档标题。没有就返回空，由 BaseParser 回落成文件名。"""
        for block in blocks:
            if block.block_type is BlockType.HEADING and block.heading_level == 1:
                return block.text
        return ""
