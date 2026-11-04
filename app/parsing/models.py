"""统一中间文档结构（Canonical Document）。

这一层存在的理由
----------------
现在 `app/rag/loader.py` 把任意文件压成一个 `content: str`。信息一旦降维成
字符串就**不可逆**：页码、块类型、单元格边界没有任何地方能再推导回来。
所以先定结构，再写解析器 —— 这是顺序问题，不是代码洁癖。

下游（Chunker / Chroma / 检索）只认识这里的模型，不认识 pypdf、python-docx、
python-pptx、openpyxl。将来把某个解析器换成 Docling，只要它还能产出
`ParsedDocument`，后面的代码一行都不用改。

两个刻意的设计
--------------
1. **表格保留合并信息**。`TableData.cells` 存 row/col/rowspan/colspan 的原始
   单元格，而不是拍平成 `list[list[str]]`。校园文件的表头普遍是合并的
   （「国家奖学金」横跨三列），拍平之后那个层级关系就永久丢了。
   需要二维网格时调 `grid()`，它会把合并区域展开 —— 但是**读的时候才展开**，
   存的时候不丢。
2. **状态和内容分开**。`parse_status` 只描述「这次解析成不成功」，
   不掺业务状态（现行/废止），也不掺入库状态机（PENDING→COMPLETED）。
   后者属于阶段六，混进来的话，一个字段要同时回答两个问题。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class FileType(str, Enum):
    """文件格式。用枚举而不是裸字符串，是为了让「拼错后缀」变成类型错误。"""

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    MARKDOWN = "md"
    TEXT = "txt"
    UNKNOWN = "unknown"

    @classmethod
    def from_suffix(cls, suffix: str) -> "FileType":
        """按后缀推断格式。注意：**后缀只用于初判**，真实类型要在 ParserFactory
        里用文件头（magic bytes）复核 —— 改个后缀就能骗过后缀判断。"""
        key = str(suffix or "").lower().lstrip(".")
        for member in cls:
            if member.value == key:
                return member
        return cls.UNKNOWN


class BlockType(str, Enum):
    """内容块类型。

    前五种是六个阶段里明确要处理的；后几种是**预留**：解析器现在可能不产出，
    但类型先占住，将来加图表/公式/附件识别时不用改 Chunker 的匹配逻辑。
    """

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    LIST = "list"
    # ---- 预留 ----
    CHART = "chart"
    FORMULA = "formula"
    ATTACHMENT = "attachment"
    CODE = "code"
    NOTE = "note"


class ParseStatus(str, Enum):
    """单份文件的解析结果。

    COMPLETED  全部内容都拿到了
    PARTIAL    拿到了部分，但有块失败（比如表格解析失败、图片没读出来）
    EMPTY      文件能打开，但**一个字都没拿到**（扫描件、纯图片/pdfplumber 无文本层）
    FAILED     解析过程直接抛错，没有可用内容

    EMPTY 单独列出来，是因为它最容易被误判：现在的代码把空内容当成
    「格式不支持」直接跳过，文件就在库里消失了，管理员完全看不见。
    """

    COMPLETED = "completed"
    PARTIAL = "partial"
    EMPTY = "empty"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# 来源位置
# ---------------------------------------------------------------------------

_CELL_RANGE = re.compile(r"^[A-Z]{1,3}[1-9]\d*(?::[A-Z]{1,3}[1-9]\d*)?$")


class SourceLocation(BaseModel):
    """内容在原文件里的位置。

    不同格式填不同字段，**允许同时为空**（比如人工构造的块）。
    不做「PDF 必须有页码」这种跨字段硬校验：那会把校验规则和格式绑死，
    将来加一种格式就要回来改这里。

    字段含义：
        page_start/page_end   PDF 页码，1 起，闭区间
        bbox                  PDF 坐标 (x0, y0, x1, y1)，单位 pt，原点左下
        paragraph_index       Word 正文第几个段落（0 起）
        table_index           Word 第几张表（0 起）
        slide_number          PPT 第几页幻灯片（1 起）
        sheet_name            Excel 工作表名
        cell_range            Excel 单元格范围，如 "A1:D9"
        char_offset           该块在文件纯文本流中的起始字符下标，前端高亮用
    """

    model_config = ConfigDict(extra="forbid")

    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    paragraph_index: int | None = Field(default=None, ge=0)
    table_index: int | None = Field(default=None, ge=0)
    slide_number: int | None = Field(default=None, ge=1)
    sheet_name: str | None = None
    cell_range: str | None = None
    char_offset: int | None = Field(default=None, ge=0)

    @field_validator("sheet_name")
    @classmethod
    def _strip_sheet(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("cell_range")
    @classmethod
    def _check_range(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().upper().replace("$", "")
        if not _CELL_RANGE.match(cleaned):
            raise ValueError(f"cell_range 格式不合法：{value!r}（应形如 A1:D9）")
        return cleaned

    @model_validator(mode="after")
    def _check_page_and_bbox(self) -> "SourceLocation":
        if self.page_end is not None and self.page_start is None:
            raise ValueError("给了 page_end 就必须给 page_start")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError(
                f"page_end({self.page_end}) 不能小于 page_start({self.page_start})"
            )
        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            if x1 < x0 or y1 < y0:
                raise ValueError(f"bbox 左上/右下角顺序反了：{self.bbox}")
        return self

    def is_empty(self) -> bool:
        """是否一点定位信息都没有 —— 阶段三验收时用来找「来源丢失」的块。"""
        return not any(
            value is not None
            for value in (
                self.page_start,
                self.paragraph_index,
                self.table_index,
                self.slide_number,
                self.sheet_name,
                self.char_offset,
            )
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """拍成 Chroma 能存的标量字典（str / int / float / bool）。

        为什么要有这个方法：Chroma 的 metadata 装不下嵌套结构，也不接受 None。
        把「挑哪些字段、怎么转字符串」这件事收在模型里，阶段五就不会出现
        「这个字段忘了转、Chroma 直接拒写」的线上问题。
        """
        flat: dict[str, Any] = {}
        if self.page_start is not None:
            flat["page_start"] = int(self.page_start)
        if self.page_end is not None:
            flat["page_end"] = int(self.page_end)
        if self.slide_number is not None:
            flat["slide_number"] = int(self.slide_number)
        if self.sheet_name:
            flat["sheet_name"] = self.sheet_name
        if self.cell_range:
            flat["cell_range"] = self.cell_range
        if self.paragraph_index is not None:
            flat["paragraph_index"] = int(self.paragraph_index)
        if self.char_offset is not None:
            flat["char_offset"] = int(self.char_offset)
        return flat


# ---------------------------------------------------------------------------
# 表格
# ---------------------------------------------------------------------------


class TableCell(BaseModel):
    """一个原始单元格（未展开合并）。

    row/col 是左上角坐标，rowspan/colspan 是跨行跨列数。
    用「左上角 + 跨度」而不是「起止坐标」，是因为它跟 HTML 的
    rowspan/colspan 语义一致，从 docx（gridSpan/vMerge）、pptx、xlsx
    读出来时不用做一次坐标换算，少一层出错的机会。
    """

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    is_header: bool = False


class TableData(BaseModel):
    """一张表：尺寸 + 原始单元格 + 表头行数 + 表格标题。

    header_row_count 支持**多级表头**（校园表格里「国家奖学金 / 金额」这种
    两行表头很常见），不是布尔值。
    caption 是表格上方的标题文字（如「表 1 各学院名额分配」），它经常是
    理解表格的前提，丢了下游就不知道这张表在说什么。
    """

    model_config = ConfigDict(extra="forbid")

    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    cells: list[TableCell] = Field(default_factory=list)
    header_row_count: int = Field(default=0, ge=0)
    caption: str = ""

    @model_validator(mode="after")
    def _check_shape(self) -> "TableData":
        if self.header_row_count > self.n_rows:
            raise ValueError(
                f"表头行数({self.header_row_count})不能超过总行数({self.n_rows})"
            )

        occupied: dict[tuple[int, int], int] = {}
        for index, cell in enumerate(self.cells):
            if cell.row + cell.rowspan > self.n_rows:
                raise ValueError(
                    f"单元格#{index} 纵向越界：row={cell.row} + rowspan={cell.rowspan} "
                    f"> n_rows={self.n_rows}"
                )
            if cell.col + cell.colspan > self.n_cols:
                raise ValueError(
                    f"单元格#{index} 横向越界：col={cell.col} + colspan={cell.colspan} "
                    f"> n_cols={self.n_cols}"
                )
            for r in range(cell.row, cell.row + cell.rowspan):
                for c in range(cell.col, cell.col + cell.colspan):
                    if (r, c) in occupied:
                        raise ValueError(
                            f"单元格重叠：第 {r} 行第 {c} 列被 cell#{occupied[(r, c)]} "
                            f"和 cell#{index} 同时占用"
                        )
                    occupied[(r, c)] = index
        return self

    def grid(self) -> list[list[str]]:
        """展开合并单元格，返回 n_rows × n_cols 的二维文本。

        合并区域内的每个格子都填同一个文本 —— 这是有损的（丢失「这是一次合并」
        这件事），所以**只用于读取**，永远不要拿它反序列化回 TableData。
        """
        grid = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for cell in self.cells:
            for r in range(cell.row, min(cell.row + cell.rowspan, self.n_rows)):
                for c in range(cell.col, min(cell.col + cell.colspan, self.n_cols)):
                    grid[r][c] = cell.text
        return grid

    @staticmethod
    def _escape(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").strip()

    def to_markdown(self, include_caption: bool = True) -> str:
        """转成 Markdown 表格，供 Chunk 的正文使用。

        多级表头会被压成一行，单元格内容用 " / " 连接（如 "国家奖学金 / 金额"）：
        Markdown 语法本身表达不了跨行表头，与其输出一张语法非法的表，
        不如保留可读的层级文字。**原始结构仍在 cells 里，没丢。**
        """
        if self.n_rows == 0 or self.n_cols == 0:
            return ""

        grid = self.grid()
        header_rows = self.header_row_count
        if header_rows <= 0:
            # Markdown 要求第一行必须是表头。没有表头信息时借用第一行，
            # 但把它当作数据行也没什么可做 —— 只能这样，并保持内容完整。
            header_rows = 1

        if header_rows == 1:
            head = grid[0]
            body = grid[1:]
        else:
            head = []
            for col in range(self.n_cols):
                parts: list[str] = []
                for r in range(header_rows):
                    value = grid[r][col].strip()
                    if value and (not parts or parts[-1] != value):
                        parts.append(value)
                head.append(" / ".join(parts))
            body = grid[header_rows:]

        lines = [
            "| " + " | ".join(self._escape(x) for x in head) + " |",
            "| " + " | ".join("---" for _ in range(self.n_cols)) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(self._escape(x) for x in row) + " |")

        table = "\n".join(lines)
        if include_caption and self.caption.strip():
            return f"{self.caption.strip()}\n\n{table}"
        return table


# ---------------------------------------------------------------------------
# 内容块
# ---------------------------------------------------------------------------


class DocumentBlock(BaseModel):
    """文档里的一个结构化内容单元。

    heading_path 的约定：表示**这个块所处的章节路径**，包含各级标题文字。
    对 heading 块，路径的最后一项就是它自己（如 ["第三章 评审", "一、申请条件"]）。
    这样 Chunk 拿到的 heading_path 直接可用，不用回头找它的父标题。
    """

    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(min_length=1)
    block_type: BlockType
    text: str = ""
    heading_level: int | None = Field(default=None, ge=1, le=6)
    heading_path: list[str] = Field(default_factory=list)
    table_data: TableData | None = None
    image_path: str | None = None
    source_location: SourceLocation = Field(default_factory=SourceLocation)
    # 业务扩展：list 的有序性、图片的 ocr_text、图表的系列数据等。
    # 刻意不做成 Pydantic 的 extra="allow"：那样拼错字段名会静默通过，
    # 这里要求显式放进 extra，拼错就报错。
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_payload(self) -> "DocumentBlock":
        if self.block_type is BlockType.HEADING:
            if self.heading_level is None:
                raise ValueError("heading 块必须给 heading_level（1-6）")
            if not self.text.strip():
                raise ValueError("heading 块的 text 不能为空")
            if not self.heading_path:
                raise ValueError("heading 块必须给 heading_path（至少包含自身标题）")
            if self.heading_path[-1] != self.text.strip():
                raise ValueError(
                    f"heading 块 heading_path 的最后一项应与 text 一致："
                    f"{self.heading_path[-1]!r} != {self.text.strip()!r}"
                )
        if self.block_type is BlockType.TABLE and self.table_data is None:
            raise ValueError("table 块必须给 table_data")
        if self.block_type is BlockType.IMAGE and not (self.image_path or "").strip():
            raise ValueError("image 块必须给 image_path")

        has_payload = bool(
            self.text.strip()
            or self.table_data is not None
            or (self.image_path or "").strip()
            or self.extra
        )
        if not has_payload:
            raise ValueError(f"{self.block_type.value} 块没有任何内容载体")
        return self

    def render_text(self) -> str:
        """渲染成文本，供 `ParsedDocument.to_text()` 与 Chunk 使用。

        渲染规则刻意跟着 Markdown 走（标题加 #），因为下游要做的是「给人读、
        给向量模型读」，不是回写 Word。
        """
        if self.block_type is BlockType.TABLE and self.table_data is not None:
            markdown = self.table_data.to_markdown(include_caption=True)
            return markdown or self.text

        if self.block_type is BlockType.IMAGE:
            ocr = str(self.extra.get("ocr_text") or "").strip()
            label = f"[图片] {self.image_path}" + (f"：{ocr}" if ocr else "")
            return "\n".join(part for part in (self.text.strip(), label) if part)

        if self.block_type is BlockType.HEADING and self.heading_level:
            return f"{'#' * self.heading_level} {self.text.strip()}"

        return self.text.strip()


class ParseError(BaseModel):
    """一条解析错误。fatal=True 表示「这份文件没有可用内容」。"""

    model_config = ConfigDict(extra="forbid")

    stage: str = Field(min_length=1)
    message: str = Field(min_length=1)
    location: SourceLocation | None = None
    fatal: bool = False


# ---------------------------------------------------------------------------
# 文档级元数据
# ---------------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    """文件的档案信息。

    extra="allow"：这里就是「将来会加字段」的地方（作者、密级、附件清单……），
    每加一个都要改模型的话，解析器就没法往下走了。而块级结构用 forbid，
    因为那里的字段名是契约，拼错必须报错。
    """

    model_config = ConfigDict(extra="allow")

    file_path: str = ""
    file_size: int | None = Field(default=None, ge=0)
    sha256: str | None = None
    mime_type: str = ""
    page_count: int | None = Field(default=None, ge=0)
    slide_count: int | None = Field(default=None, ge=0)
    sheet_count: int | None = Field(default=None, ge=0)

    # ---- 校园档案字段（与现有 document 表、LoadedDoc 对齐）----
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

    @field_validator("sha256")
    @classmethod
    def _check_sha(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", cleaned):
            raise ValueError("sha256 必须是 64 位十六进制")
        return cleaned

    @model_validator(mode="after")
    def _fill_year(self) -> "DocumentMetadata":
        # year 是 publish_date 的派生值。让人手填两遍，迟早出现「日期 2025、
        # 年份 2026」这种自相矛盾的记录，检索按年份过滤就会漏文件。
        if self.year is None and self.publish_date is not None:
            self.year = self.publish_date.year
        return self


# ---------------------------------------------------------------------------
# 文档
# ---------------------------------------------------------------------------


def document_id_for(file_path: str) -> str:
    """由 file_path 派生稳定的 document_id。

    为什么不用自增 id：`document` 表是 DELETE + INSERT 写入的，重建后
    自增 id 会漂（实测 1 → 22）。任何跨重建的引用（Chunk、变更记录）
    只要挂在自增 id 上就会断。

    为什么不用文件内容哈希：内容一改 id 就变，同一份文件的 Chunk 会被
    当成两个文档，旧 Chunk 清理不掉。**身份跟路径走，版本跟哈希走**，
    这两件事必须分开。
    """
    normalized = str(file_path or "").strip().replace("\\", "/")
    if not normalized:
        raise ValueError("file_path 不能为空")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def new_block_id(document_id: str, index: int) -> str:
    """块 ID：文档 ID + 序号。

    稳定、可读、可追溯 —— 日志里看到 `a1b2c3d4-b0007` 就知道是这份文件的
    第 8 个块。用 uuid4 的话，同一个文件解析两次会得到两套 ID，
    「重新入库后旧 Chunk 没删干净」这类问题就查不出来了。
    """
    return f"{document_id}-b{int(index):04d}"


class ParsedDocument(BaseModel):
    """一份解析完成的文件。

    这是 Parser 的**唯一出口**，也是 Chunker 的**唯一入口**。
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    file_type: FileType
    title: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    blocks: list[DocumentBlock] = Field(default_factory=list)
    parser_name: str = Field(min_length=1)
    parser_version: str = "0.1.0"
    parse_status: ParseStatus = ParseStatus.COMPLETED
    errors: list[ParseError] = Field(default_factory=list)
    parsed_at: dt.datetime = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc)
    )

    @field_validator("title")
    @classmethod
    def _fallback_title(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _check_consistency(self) -> "ParsedDocument":
        seen: set[str] = set()
        for block in self.blocks:
            if block.block_id in seen:
                raise ValueError(f"block_id 重复：{block.block_id}")
            seen.add(block.block_id)

        fatal = [err for err in self.errors if err.fatal]
        if self.parse_status is ParseStatus.EMPTY:
            # 允许只有图片块：扫描件正是"一个字都没拿到，但图片抠出来了"。
            # 这些图片块是阶段四做 OCR 的原料，扔掉就再也找不回来了。
            content_blocks = [
                block for block in self.blocks if block.block_type is not BlockType.IMAGE
            ]
            if content_blocks:
                raise ValueError(
                    f"parse_status=empty 但有 {len(content_blocks)} 个文本/表格块，"
                    f"状态与内容矛盾"
                )
        if self.parse_status is ParseStatus.COMPLETED:
            if not self.blocks:
                raise ValueError("parse_status=completed 但没有任何内容块")
            if fatal:
                raise ValueError("parse_status=completed 但存在 fatal 错误")
        if self.parse_status is ParseStatus.PARTIAL:
            if not self.blocks:
                raise ValueError("parse_status=partial 但没有任何内容块")
            if not self.errors:
                raise ValueError("parse_status=partial 但没有任何错误记录，无法解释缺了什么")
        if self.parse_status is ParseStatus.FAILED and not fatal:
            raise ValueError("parse_status=failed 但没有任何 fatal 错误")
        return self

    # ---- 便捷方法：阶段三的验收、阶段五的 Chunk 都要用 ----

    def block_counts(self) -> dict[str, int]:
        """各类块的数量，按类型名排序。"""
        counts: dict[str, int] = {}
        for block in self.blocks:
            key = block.block_type.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def to_text(self) -> str:
        """按原顺序拼回纯文本。

        **兼容用的**：现有 `splitter` 只认一段字符串。改造期间它让新解析器
        能直接喂给旧流水线，等结构感知 Chunker 上线后再退居备份。
        """
        parts = [block.render_text().strip() for block in self.blocks]
        return "\n\n".join(part for part in parts if part)

    def locality_report(self) -> dict[str, Any]:
        """来源定位的覆盖情况。

        不放进校验器，是因为「块没有页码」在不同阶段是不同性质的问题：
        阶段三解析器可能还没做完，不该直接报错；但验收时必须能看见。
        """
        total = len(self.blocks)
        located = sum(1 for block in self.blocks if not block.source_location.is_empty())
        return {
            "总块数": total,
            "有位置信息": located,
            "缺位置信息": total - located,
            "覆盖率": round(located / total, 4) if total else 0.0,
        }

    def summary(self) -> dict[str, Any]:
        """阶段三验收表格的一行：文件名、块数、各类型数、状态、错误数。"""
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "file_type": self.file_type.value,
            "title": self.title,
            "块数": len(self.blocks),
            "各类块": self.block_counts(),
            "解析状态": self.parse_status.value,
            "错误数": len(self.errors),
            "致命错误数": sum(1 for err in self.errors if err.fatal),
            "定位覆盖率": self.locality_report()["覆盖率"],
            "parser": f"{self.parser_name}@{self.parser_version}",
        }
