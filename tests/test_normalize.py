"""内容标准化与清洗的验收测试。

重点不是"函数跑通了"，而是四件容易出事的地方：
    1. 该删的脏字符删了
    2. **不该删的没删**（全角空格、正文里的重复句）
    3. 页眉页脚只是标记，原文一个字不少
    4. OCR 结果和原文能区分开

运行：
    D:/AgentLearnVenv/Scripts/python.exe -m pytest tests/test_normalize.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing import (
    BlockType,
    DocumentBlock,
    FileType,
    ParsedDocument,
    ParseStatus,
    SourceLocation,
    document_id_for,
    new_block_id,
)
from app.parsing.normalize import (
    NormalizeOptions,
    NormalizeReport,
    OcrEngine,
    OcrResult,
    clean_cell,
    clean_text,
    drop_boilerplate,
    normalize,
)

PAGE_HEIGHT = 800.0


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def block(
    index: int,
    text: str,
    page: int,
    *,
    bbox: tuple[float, float, float, float] = (50.0, 300.0, 500.0, 340.0),
    block_type: BlockType = BlockType.PARAGRAPH,
    heading_path: list[str] | None = None,
    heading_level: int | None = None,
) -> DocumentBlock:
    doc_id = document_id_for("demo.pdf")
    return DocumentBlock(
        block_id=new_block_id(doc_id, index),
        block_type=block_type,
        text=text,
        heading_level=heading_level,
        heading_path=heading_path or [],
        source_location=SourceLocation(page_start=page, page_end=page, bbox=bbox),
        extra={"page_height": PAGE_HEIGHT},
    )


def make_doc(blocks: list[DocumentBlock]) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id_for("demo.pdf"),
        file_name="demo.pdf",
        file_type=FileType.PDF,
        title="演示",
        blocks=blocks,
        parser_name="fake",
        parse_status=ParseStatus.COMPLETED,
    )


def header_block(index: int, page: int, text: str = "XX大学教务处文件") -> DocumentBlock:
    return block(index, text, page, bbox=(50.0, 20.0, 500.0, 45.0))


def footer_block(index: int, page: int, text: str) -> DocumentBlock:
    return block(index, text, page, bbox=(50.0, 770.0, 500.0, 790.0))


# ---------------------------------------------------------------------------
# 文本清洗
# ---------------------------------------------------------------------------


def test_clean_text_removes_control_and_zero_width() -> None:
    raw = "关于做好\u200b奖学金\x07评审工作的通知\r\n第二行\ufeff"
    cleaned = clean_text(raw)

    assert "\u200b" not in cleaned
    assert "\ufeff" not in cleaned
    assert "\x07" not in cleaned
    assert "\r" not in cleaned
    assert cleaned == "关于做好奖学金评审工作的通知\n第二行"


def test_clean_text_keeps_full_width_space() -> None:
    """全角空格是中文排版的一部分（「第一章　总　则」），不能压。"""
    raw = "第一章\u3000总\u3000则"
    assert clean_text(raw) == raw


def test_clean_text_collapses_long_spaces() -> None:
    assert clean_text("国家奖学金      名额") == "国家奖学金 名额"
    # 两个空格是正常排版，不动
    assert clean_text("A  B") == "A  B"


def test_clean_text_reports_counts() -> None:
    report = NormalizeReport()
    clean_text("a\u200b\x07b     c", report)
    assert report.control_chars_removed == 2
    assert report.whitespace_collapsed == 1


def test_clean_cell_does_not_collapse_spaces() -> None:
    """单元格里的连续空格常用来对齐数字，压掉会破坏可读性。"""
    assert clean_cell("8   000") == "8   000"
    assert clean_cell("第一行\r\n第二行") == "第一行 第二行"


# ---------------------------------------------------------------------------
# 页眉页脚：标记而不是删除
# ---------------------------------------------------------------------------


def test_repeated_header_across_pages_is_marked() -> None:
    blocks = [header_block(0, 1)]
    for page in range(1, 6):
        blocks.append(block(page * 2, f"第{page}页正文内容", page))
    blocks[1:1] = [header_block(10 + p, p) for p in range(2, 6)]

    doc = make_doc(blocks)
    normalized, report = normalize(doc)

    marked = [b for b in normalized.blocks if b.extra.get("is_boilerplate")]
    assert len(marked) == 5
    assert all(b.text == "XX大学教务处文件" for b in marked)
    assert report.boilerplate_blocks == 5

    # 关键：原文一个字都不能少
    assert len(normalized.blocks) == len(doc.blocks)
    assert sum(1 for b in normalized.blocks if b.text == "XX大学教务处文件") == 5


def test_page_number_in_footer_is_marked() -> None:
    blocks = []
    for page in range(1, 5):
        blocks.append(block(page * 2, f"正文{page}", page))
        blocks.append(footer_block(page * 2 + 1, page, f"{page}/4"))

    normalized, _ = normalize(make_doc(blocks))
    marked = [b for b in normalized.blocks if b.extra.get("is_boilerplate")]

    assert len(marked) == 4
    assert all("页码" in b.extra["boilerplate_reason"] for b in marked)


def test_single_page_document_is_not_touched() -> None:
    """单页没有"重复"这回事，任何判定都是猜。"""
    doc = make_doc([header_block(0, 1), block(1, "正文", 1)])
    normalized, report = normalize(doc)

    assert report.boilerplate_blocks == 0
    assert not any(b.extra.get("is_boilerplate") for b in normalized.blocks)


def test_repeated_text_in_body_is_not_marked() -> None:
    """"特此通知"在每页正文出现，但它不在页面边缘，不能当页眉。"""
    blocks = []
    for page in range(1, 5):
        blocks.append(block(page * 2, "特此通知。", page))
    normalized, report = normalize(make_doc(blocks))

    assert report.boilerplate_blocks == 0


def test_long_text_at_top_is_not_marked() -> None:
    """页面顶部但很长，那是正文第一段，不是页眉。"""
    long_text = "根据上级有关文件精神，现将我校本年度奖学金评审工作的有关事项通知如下。" * 3
    blocks = [block(p, long_text, p, bbox=(50.0, 20.0, 500.0, 60.0)) for p in range(1, 5)]
    normalized, report = normalize(make_doc(blocks))

    assert report.boilerplate_blocks == 0


def test_drop_boilerplate_returns_copy_and_keeps_original() -> None:
    blocks = []
    for page in range(1, 5):
        blocks.append(header_block(page * 2, page))
        blocks.append(block(page * 2 + 1, f"正文{page}", page))

    doc = make_doc(blocks)
    normalized, _ = normalize(doc)
    trimmed = drop_boilerplate(normalized)

    assert len(trimmed.blocks) == 4
    assert len(normalized.blocks) == 8, "原文档必须保持完整"
    assert all(b.text.startswith("正文") for b in trimmed.blocks)


# ---------------------------------------------------------------------------
# 标题改写后 heading_path 要同步
# ---------------------------------------------------------------------------


def test_heading_path_follows_cleaned_heading_text() -> None:
    dirty = "一、申请条件\u200b"
    doc = make_doc(
        [
            block(0, dirty, 1, block_type=BlockType.HEADING, heading_path=[dirty],
                  heading_level=2),
            block(1, "具有中华人民共和国国籍。", 1, heading_path=[dirty]),
        ]
    )
    # 构造时 heading 的 text 与 heading_path 末项一致，才能通过模型校验
    normalized, report = normalize(doc)

    assert normalized.blocks[0].text == "一、申请条件"
    assert normalized.blocks[0].heading_path == ["一、申请条件"]
    assert normalized.blocks[1].heading_path == ["一、申请条件"]
    assert report.blocks_changed == 1


def test_overlong_table_caption_is_cleared() -> None:
    from app.parsing import TableCell, TableData

    table = TableData(
        n_rows=1,
        n_cols=1,
        cells=[TableCell(text="名额", row=0, col=0, is_header=True)],
        header_row_count=1,
        caption="这是一段明显不是表标题的正文" * 8,
    )
    doc_id = document_id_for("demo.pdf")
    doc = make_doc(
        [
            DocumentBlock(
                block_id=new_block_id(doc_id, 0),
                block_type=BlockType.TABLE,
                table_data=table,
                source_location=SourceLocation(page_start=1),
                extra={"page_height": PAGE_HEIGHT},
            )
        ]
    )
    normalized, report = normalize(doc)

    assert normalized.blocks[0].table_data.caption == ""
    assert any("表标题过长" in w for w in report.warnings)


def test_table_cells_are_cleaned() -> None:
    from app.parsing import TableCell, TableData

    table = TableData(
        n_rows=1,
        n_cols=2,
        cells=[
            TableCell(text="学\u200b院", row=0, col=0, is_header=True),
            TableCell(text="名额\x07", row=0, col=1, is_header=True),
        ],
        header_row_count=1,
    )
    doc_id = document_id_for("demo.pdf")
    doc = make_doc(
        [
            DocumentBlock(
                block_id=new_block_id(doc_id, 0),
                block_type=BlockType.TABLE,
                table_data=table,
                source_location=SourceLocation(page_start=1),
                extra={"page_height": PAGE_HEIGHT},
            )
        ]
    )
    normalized, _ = normalize(doc)
    cells = normalized.blocks[0].table_data.cells
    assert cells[0].text == "学院"
    assert cells[1].text == "名额"


# ---------------------------------------------------------------------------
# OCR 通道
# ---------------------------------------------------------------------------


class FakeOcr:
    name = "fake-ocr"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def recognize(self, image_path: str, *, location=None) -> OcrResult:
        self.calls.append(image_path)
        return OcrResult(text="图 3 报名流程", engine=self.name, confidence=0.93)


def test_ocr_marks_generated_text_separately(tmp_path: Path) -> None:
    image = tmp_path / "p1.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    doc_id = document_id_for("demo.pdf")
    doc = make_doc(
        [
            DocumentBlock(
                block_id=new_block_id(doc_id, 0),
                block_type=BlockType.IMAGE,
                image_path=str(image),
                source_location=SourceLocation(page_start=1),
                extra={"page_height": PAGE_HEIGHT},
            )
        ]
    )
    engine = FakeOcr()
    normalized, report = normalize(doc, NormalizeOptions(ocr_engine=engine))

    block0 = normalized.blocks[0]
    assert block0.extra["ocr_text"] == "图 3 报名流程"
    assert block0.extra["ocr_source"] == "ocr"
    assert block0.extra["ocr_engine"] == "fake-ocr"
    assert report.ocr_succeeded == 1
    # OCR 结果必须能出现在渲染文本里，否则下游读不到
    assert "图 3 报名流程" in block0.render_text()


def test_ocr_skips_embedded_images_with_warning() -> None:
    doc_id = document_id_for("demo.pdf")
    doc = make_doc(
        [
            DocumentBlock(
                block_id=new_block_id(doc_id, 0),
                block_type=BlockType.IMAGE,
                image_path="embedded://pdf/page1/image1.png",
                source_location=SourceLocation(page_start=1),
                extra={"page_height": PAGE_HEIGHT},
            )
        ]
    )
    engine = FakeOcr()
    _, report = normalize(doc, NormalizeOptions(ocr_engine=engine))

    assert engine.calls == []
    assert any("未落盘" in w for w in report.warnings)


def test_ocr_engine_failure_does_not_break_document() -> None:
    class BoomOcr:
        name = "boom"

        def recognize(self, image_path: str, *, location=None):
            raise RuntimeError("模型崩了")

    doc_id = document_id_for("demo.pdf")
    doc = make_doc(
        [
            DocumentBlock(
                block_id=new_block_id(doc_id, 0),
                block_type=BlockType.IMAGE,
                image_path="test_corpus/docx/word_sample.docx",
                source_location=SourceLocation(page_start=1),
                extra={"page_height": PAGE_HEIGHT},
            )
        ]
    )
    normalized, report = normalize(doc, NormalizeOptions(ocr_engine=BoomOcr()))

    assert normalized.parse_status is ParseStatus.COMPLETED
    assert any("OCR 失败" in w for w in report.warnings)


def test_ocr_is_off_by_default() -> None:
    doc_id = document_id_for("demo.pdf")
    doc = make_doc(
        [
            DocumentBlock(
                block_id=new_block_id(doc_id, 0),
                block_type=BlockType.IMAGE,
                image_path="embedded://x.png",
                source_location=SourceLocation(page_start=1),
            )
        ]
    )
    _, report = normalize(doc)
    assert report.ocr_attempted == 0


# ---------------------------------------------------------------------------
# 真实文件
# ---------------------------------------------------------------------------


def test_real_pdf_page_numbers_are_marked() -> None:
    """真实公文：每页底部是 "n/30"，应全部标出来且零误标。"""
    path = Path("test_corpus/pdf/eea_我省2026年普通高考本科批次正式投档.pdf")
    if not path.exists():
        pytest.skip("样例文件不存在")

    from app.parsing.factory import parse_file

    doc = parse_file(path)
    normalized, report = normalize(doc)

    assert report.boilerplate_blocks >= 20
    marked = [b for b in normalized.blocks if b.extra.get("is_boilerplate")]
    assert all(len(b.text) <= 12 for b in marked), "被标记的应该是页码这种短文本"
    assert len(normalized.blocks) == len(doc.blocks)


def test_normalize_is_idempotent() -> None:
    """洗两遍结果必须一样，否则说明规则之间在互相打架。"""
    blocks = []
    for page in range(1, 5):
        blocks.append(header_block(page * 2, page))
        blocks.append(block(page * 2 + 1, f"正文\u200b{page}", page))

    once, _ = normalize(make_doc(blocks))
    twice, _ = normalize(once)

    assert [b.text for b in once.blocks] == [b.text for b in twice.blocks]
    assert [b.extra.get("is_boilerplate") for b in once.blocks] == [
        b.extra.get("is_boilerplate") for b in twice.blocks
    ]
