"""内容标准化、清洗与质量检查。

这一层解决的是「不同解析器吐出来的东西脏法不一样」：PDF 抽出来的文字带
控制字符和断行、Word 段尾有尾随空格、Excel 单元格里有零宽字符。

清洗的三条铁律
--------------
1. **只做可逆或无害的改动**。删控制字符、压多余空格，这些不改变语义；
   改标点、转繁简、删"看起来像页眉"的段落，这些都会让原文失真。
2. **页眉页脚只标记，不删除**。真实公文里页眉经常写着发文机关（「XX大学
   文件」），那本身就是有效信息。标记出来让切分层决定要不要进 Chunk，
   原始数据永远留着。
3. **清洗前后必须可追溯**。每一条规则命中多少次、哪个块被改了，都进
   `NormalizeReport`。没有报告就等于没有证据。

关于页眉页脚判定
----------------
用四个信号叠加，而不是"页面最上方几行一律删"：
    · 位置：靠近页面顶部 12% 或底部 12%（要 PDF 页高，从 block.extra 取）
    · 重复：去掉数字后的文本在多页出现，页数 ≥ max(3, 总页数×30%)
    · 页码模式：「第 3 页」「- 12 -」「3 / 20」
    · 长度：超过 80 字的不当页眉（那多半是正文第一段）
单页、两页文档直接跳过 —— 没有"重复"就没有证据，删了就是瞎猜。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from app import config
from app.logging_config import get_logger
from app.parsing.models import (
    BlockType,
    DocumentBlock,
    ParsedDocument,
    ParseStatus,
    SourceLocation,
)

log = get_logger(__name__)

# 控制字符：保留 \t(\x09) 和 \n(\x0a)，其余全删
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 零宽字符与双向控制符：复制粘贴和 PDF 抽取的重灾区
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
# 连续 3 个以上半角空格压成 1 个（全角空格 \u3000 不动）
_MULTI_SPACE = re.compile(r"[ \t]{3,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# 页码的几种常见写法
_PAGE_NUMBER = re.compile(
    r"^[\s\-—–]*(第\s*)?\d+(\s*页)?[\s\-—–]*$"
    r"|^\d+\s*/\s*\d+$"
    r"|^[Pp]age\s+\d+(\s+of\s+\d+)?$"
)


# ---------------------------------------------------------------------------
# 配置与报告
# ---------------------------------------------------------------------------


@dataclass
class NormalizeOptions:
    """清洗选项。默认值偏保守：能不动就不动。"""

    clean_text: bool = True
    detect_boilerplate: bool = True
    # 少于这么多页就不做页眉页脚判定：没有跨页重复，就没有证据
    boilerplate_min_pages: int = 3
    # 文本出现在这么多比例的页面上，才算"重复"
    boilerplate_min_ratio: float = 0.3
    # 页面顶部/底部这么高的区域算"边缘"
    edge_ratio: float = 0.12
    # 超过这个字数的块不当页眉页脚
    boilerplate_max_chars: int = 80
    # OCR 引擎，默认不接（要下模型，且不是每份文件都需要）
    ocr_engine: "OcrEngine | None" = None
    ocr_max_images: int = 200


@dataclass
class NormalizeReport:
    """清洗台账。验收时要能看到"改了什么、改了多少"。"""

    blocks_total: int = 0
    blocks_changed: int = 0
    control_chars_removed: int = 0
    whitespace_collapsed: int = 0
    boilerplate_blocks: int = 0
    boilerplate_texts: list[str] = field(default_factory=list)
    ocr_attempted: int = 0
    ocr_succeeded: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "总块数": self.blocks_total,
            "被改写块数": self.blocks_changed,
            "删除控制字符": self.control_chars_removed,
            "压缩空白处数": self.whitespace_collapsed,
            "标记为页眉页脚": self.boilerplate_blocks,
            "页眉页脚样例": self.boilerplate_texts[:10],
            "OCR 尝试": self.ocr_attempted,
            "OCR 成功": self.ocr_succeeded,
            "警告": self.warnings[:20],
        }


# ---------------------------------------------------------------------------
# OCR 接口
# ---------------------------------------------------------------------------


@dataclass
class OcrResult:
    text: str
    engine: str = "unknown"
    confidence: float | None = None


class OcrEngine(Protocol):
    """OCR 引擎协议。

    刻意只定义 `recognize` 一个方法：PaddleOCR、云服务、本地模型都能实现，
    Normalizer 不关心用的是哪个。阶段四只提供通道，**默认不启用** ——
    接 OCR 意味着多几百 MB 模型依赖，不该因为"别人都在用"就默认打开。
    """

    name: str

    def recognize(
        self, image_path: str, *, location: SourceLocation | None = None
    ) -> OcrResult | None:
        ...


# ---------------------------------------------------------------------------
# 文本清洗
# ---------------------------------------------------------------------------


def clean_text(value: str, report: NormalizeReport | None = None) -> str:
    """清洗一段正文。

    做什么：
        · 统一换行（\\r\\n / \\r → \\n）
        · 删零宽字符、控制字符
        · 去每行行尾空白、压缩 3 个以上连续半角空格
        · 合并 3 个以上连续换行
    不做什么：
        · 不动全角空格（「第一章　总　则」里的排版）
        · 不改标点、不做繁简转换、不删任何"看起来没用"的句子
    """
    if not value:
        return value

    before = value
    text = value.replace("\r\n", "\n").replace("\r", "\n")

    zero_width = len(_ZERO_WIDTH.findall(text))
    text = _ZERO_WIDTH.sub("", text)

    control = len(_CONTROL.findall(text))
    text = _CONTROL.sub("", text)

    text = "\n".join(line.rstrip() for line in text.split("\n"))

    spaces = len(_MULTI_SPACE.findall(text))
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = text.strip()

    if report is not None:
        report.control_chars_removed += zero_width + control
        report.whitespace_collapsed += spaces

    return text if text or not before.strip() else before.strip()


def clean_cell(value: str) -> str:
    """单元格清洗：只去脏字符，**不压缩空格**。

    表格里连续空格常用来对齐数字，压掉会让 `8 000` 这类写法更难读。
    """
    if not value:
        return value
    text = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 页眉页脚
# ---------------------------------------------------------------------------


def _fingerprint(text: str) -> str:
    """去掉数字和标点后的指纹。

    页眉里的年份、文号、页码每页都不同，但去掉数字后骨架一样
    （「XX大学文件 2026 年第 3 号」→「XX大学文件第号」），
    这样才能跨页比对。
    """
    return re.sub(r"[\d\s\W_]+", "", text, flags=re.UNICODE)


def _is_edge(block: DocumentBlock, edge_ratio: float) -> bool:
    """块是否落在页面顶部/底部边缘。需要 PDF 写进来的 page_height。"""
    height = block.extra.get("page_height")
    bbox = block.source_location.bbox
    if not height or not bbox:
        return False
    try:
        page_height = float(height)
        top, bottom = float(bbox[1]), float(bbox[3])
    except (TypeError, ValueError):
        return False
    if page_height <= 0:
        return False
    return top <= page_height * edge_ratio or bottom >= page_height * (1 - edge_ratio)


def _mark_boilerplate(
    doc: ParsedDocument, options: NormalizeOptions, report: NormalizeReport
) -> None:
    pages = {
        block.source_location.page_start
        for block in doc.blocks
        if block.source_location.page_start
    }
    if len(pages) < options.boilerplate_min_pages:
        return

    threshold = max(
        options.boilerplate_min_pages,
        math.ceil(len(pages) * options.boilerplate_min_ratio),
    )

    occurrences: dict[str, set[int]] = {}
    for block in doc.blocks:
        page = block.source_location.page_start
        if not page:
            continue
        fingerprint = _fingerprint(block.text)
        if fingerprint:
            occurrences.setdefault(fingerprint, set()).add(page)

    for block in doc.blocks:
        if block.block_type not in (
            BlockType.PARAGRAPH, BlockType.HEADING, BlockType.LIST
        ):
            continue
        text = block.text.strip()
        if not text or len(text) > options.boilerplate_max_chars:
            continue
        if not _is_edge(block, options.edge_ratio):
            continue

        fingerprint = _fingerprint(text)
        repeated = bool(fingerprint) and len(occurrences.get(fingerprint, ())) >= threshold
        looks_like_page_number = bool(_PAGE_NUMBER.match(text))
        if not (repeated or looks_like_page_number):
            continue

        block.extra["is_boilerplate"] = True
        block.extra["boilerplate_reason"] = (
            f"跨页重复({len(occurrences.get(fingerprint, ()))}页)"
            if repeated
            else "页码样式"
        )
        report.boilerplate_blocks += 1
        if text not in report.boilerplate_texts and len(report.boilerplate_texts) < 20:
            report.boilerplate_texts.append(text)


def drop_boilerplate(doc: ParsedDocument) -> ParsedDocument:
    """返回一个去掉页眉页脚的副本，供**切分**使用。

    原始 doc 不受影响。这是"只标记不删除"的落点：
    入库/追溯用原 doc，生成 Chunk 时用这个副本。
    """
    kept = [
        block for block in doc.blocks
        if not block.extra.get("is_boilerplate")
    ]
    if len(kept) == len(doc.blocks):
        return doc
    clone = doc.model_copy(deep=True)
    clone.blocks = [block for block in clone.blocks if not block.extra.get("is_boilerplate")]
    return clone


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def _apply_ocr(
    doc: ParsedDocument, options: NormalizeOptions, report: NormalizeReport
) -> None:
    engine = options.ocr_engine
    if engine is None:
        return

    attempted = 0
    for block in doc.blocks:
        if block.block_type is not BlockType.IMAGE:
            continue
        if attempted >= options.ocr_max_images:
            report.warnings.append(
                f"图片数超过 OCR 上限 {options.ocr_max_images}，其余未识别"
            )
            break

        image_path = str(block.image_path or "")
        if not image_path:
            continue
        if image_path.startswith("embedded://"):
            report.warnings.append(
                f"图片未落盘（{image_path}），无法 OCR；"
                f"解析时传 assets_dir 才能识别"
            )
            continue

        target = Path(image_path)
        if not target.is_absolute():
            target = config.BASE_DIR / target
        if not target.is_file():
            report.warnings.append(f"图片文件不存在：{image_path}")
            continue

        attempted += 1
        report.ocr_attempted += 1
        try:
            result = engine.recognize(str(target), location=block.source_location)
        except Exception as exc:  # OCR 失败不该让整份解析失败
            report.warnings.append(f"OCR 失败 {image_path}：{type(exc).__name__}: {exc}")
            continue
        if result is None or not result.text.strip():
            continue

        # 关键：OCR 结果和原文分开放。`ocr_text` + `ocr_source` 让人一眼能看出
        # "这段字是机器认出来的"，而不是文件里本来就有的。
        block.extra["ocr_text"] = result.text.strip()
        block.extra["ocr_source"] = "ocr"
        block.extra["ocr_engine"] = result.engine
        if result.confidence is not None:
            block.extra["ocr_confidence"] = round(float(result.confidence), 4)
        report.ocr_succeeded += 1


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def normalize(
    doc: ParsedDocument, options: NormalizeOptions | None = None
) -> tuple[ParsedDocument, NormalizeReport]:
    """清洗一份已解析的文档，返回 (新文档, 清洗台账)。

    返回新对象而不是原地改：调用方可能还要用原始解析结果做对照
    （阶段四验收就要求"清洗前后可追溯"）。
    """
    opts = options or NormalizeOptions()
    report = NormalizeReport(blocks_total=len(doc.blocks))

    clone = doc.model_copy(deep=True)

    if opts.clean_text:
        renames: dict[str, str] = {}
        for block in clone.blocks:
            changed = False

            # 表格单元格要在判断 text 之前清洗：表格块的 text 本来就是空的，
            # 如果先看 text 有没有变，`"" == ""` 会直接 continue，
            # 单元格里的脏字符就永远清不掉。
            if block.table_data is not None:
                for cell in block.table_data.cells:
                    cleaned_cell = clean_cell(cell.text)
                    if cleaned_cell != cell.text:
                        cell.text = cleaned_cell
                        changed = True

            original = block.text
            cleaned = clean_text(original, report)
            if cleaned != original:
                if not cleaned:
                    report.warnings.append(
                        f"块 {block.block_id} 清洗后为空，已保留原文"
                    )
                else:
                    block.text = cleaned
                    changed = True
                    if block.block_type is BlockType.HEADING:
                        renames[original.strip()] = cleaned

            if changed:
                report.blocks_changed += 1

        # 标题文字变了，所有块的 heading_path 也要跟着变，
        # 否则会出现"块上挂的章节名"和"标题块的文字"对不上。
        if renames:
            for block in clone.blocks:
                if block.heading_path:
                    block.heading_path = [
                        renames.get(part, part) for part in block.heading_path
                    ]

    # 表格 caption 长度异常多半是解析时把正文段落误认成了表标题
    for block in clone.blocks:
        table = block.table_data
        if table is None or not table.caption:
            continue
        if len(table.caption) > opts.boilerplate_max_chars:
            report.warnings.append(
                f"表标题过长（{len(table.caption)} 字），已清空：{table.caption[:30]}…"
            )
            table.caption = ""

    if opts.detect_boilerplate:
        _mark_boilerplate(clone, opts, report)

    _apply_ocr(clone, opts, report)

    # 结构没变，只改文本/extra，但仍然重新校验一遍：
    # 万一某条规则把 heading_path 改坏了，这里能立刻炸出来，而不是等到入库。
    result = ParsedDocument.model_validate(clone.model_dump())
    return result, report


def normalize_file(
    path: str | Path, options: NormalizeOptions | None = None
) -> tuple[ParsedDocument, NormalizeReport]:
    """解析 + 清洗一步到位。阶段五接流水线时会用到。"""
    from app.parsing.factory import parse_file

    doc = parse_file(path)
    return normalize(doc, options)


def iter_boilerplate(doc: ParsedDocument) -> Iterable[DocumentBlock]:
    """列出被标记为页眉页脚的块，方便前端/管理端展示。"""
    for block in doc.blocks:
        if block.extra.get("is_boilerplate"):
            yield block
