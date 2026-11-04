"""统一中间结构的验收测试。

对应阶段二的八条验收要求：
    1. PDF 内容可以表示
    2. Word 内容可以表示
    3. PPT 内容可以表示
    4. Excel 内容可以表示
    5. 表格保留完整结构（含合并单元格）
    6. 图片保留引用
    7. 来源位置正确表示
    8. Pydantic 能捕获不符合规范的数据

运行：
    D:/AgentLearnVenv/Scripts/python.exe -m pytest tests/test_parsing_models.py -v
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from pydantic import ValidationError

from app.parsing import (
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    FileType,
    ParseError,
    ParsedDocument,
    ParseStatus,
    SourceLocation,
    TableCell,
    TableData,
    document_id_for,
    new_block_id,
)


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def make_doc(blocks: list[DocumentBlock], **kwargs) -> ParsedDocument:
    """默认造一份「解析成功」的文档，便于各用例只关心自己那部分。"""
    params = {
        "document_id": document_id_for("data/raw/demo.pdf"),
        "file_name": "demo.pdf",
        "file_type": FileType.PDF,
        "title": "演示文件",
        "blocks": blocks,
        "parser_name": "fake",
        "parse_status": ParseStatus.COMPLETED,
    }
    params.update(kwargs)
    return ParsedDocument(**params)


def merged_table() -> TableData:
    """2 行表头 + 合并单元格的典型校园表格。

    真实形态：
        | 学院   | 国家奖学金 | 国家奖学金 | 国家励志奖学金 |
        |        | 名额       | 金额       | 名额           |
        | 计算机 | 10         | 8000       | 20             |
    """
    return TableData(
        n_rows=3,
        n_cols=4,
        header_row_count=2,
        caption="表 1 各学院奖助名额分配",
        cells=[
            TableCell(text="学院", row=0, col=0, rowspan=2, is_header=True),
            TableCell(text="国家奖学金", row=0, col=1, colspan=2, is_header=True),
            TableCell(text="国家励志奖学金", row=0, col=3, rowspan=2, is_header=True),
            TableCell(text="名额", row=1, col=1, is_header=True),
            TableCell(text="金额", row=1, col=2, is_header=True),
            TableCell(text="计算机学院", row=2, col=0),
            TableCell(text="10", row=2, col=1),
            TableCell(text="8000", row=2, col=2),
            TableCell(text="20", row=2, col=3),
        ],
    )


# ---------------------------------------------------------------------------
# 1-4：四种格式都能表示
# ---------------------------------------------------------------------------


def test_pdf_document_is_representable() -> None:
    """PDF：跨页标题 + 正文 + 带 bbox 的块 + 页数统计。"""
    doc_id = document_id_for("data/raw/37_奖学金_通知.pdf")
    doc = ParsedDocument(
        document_id=doc_id,
        file_name="37_奖学金_通知.pdf",
        file_type=FileType.PDF,
        title="关于做好2026年国家励志奖学金评审工作的通知",
        metadata=DocumentMetadata(
            file_path="data/raw/37_奖学金_通知.pdf",
            page_count=3,
            publish_date=dt.date(2026, 9, 18),
        ),
        parser_name="pdf",
        parser_version="0.1.0",
        blocks=[
            DocumentBlock(
                block_id=new_block_id(doc_id, 0),
                block_type=BlockType.HEADING,
                text="第一章 总则",
                heading_level=1,
                heading_path=["第一章 总则"],
                source_location=SourceLocation(
                    page_start=1, page_end=1, bbox=(72.0, 700.0, 523.0, 730.0)
                ),
                extra={"font_size": 18.0},
            ),
            DocumentBlock(
                block_id=new_block_id(doc_id, 1),
                block_type=BlockType.PARAGRAPH,
                text="为做好本年度评审工作，现将有关事项通知如下。",
                heading_path=["第一章 总则"],
                source_location=SourceLocation(page_start=1, page_end=1, char_offset=0),
            ),
        ],
    )

    assert doc.file_type is FileType.PDF
    assert doc.metadata.page_count == 3
    assert doc.metadata.year == 2026, "year 应由 publish_date 推导出来"
    assert doc.block_counts() == {"heading": 1, "paragraph": 1}
    assert doc.locality_report()["覆盖率"] == 1.0
    assert doc.summary()["解析状态"] == "completed"


def test_docx_document_keeps_paragraph_table_order() -> None:
    """Word：段落与表格必须能按原顺序交错表达（这是当前代码最大的缺陷）。"""
    doc_id = document_id_for("data/raw/102_学生申诉处理办法.docx")
    blocks = [
        DocumentBlock(
            block_id=new_block_id(doc_id, 0),
            block_type=BlockType.HEADING,
            text="一、受理范围",
            heading_level=2,
            heading_path=["学生申诉处理办法", "一、受理范围"],
            source_location=SourceLocation(paragraph_index=0),
        ),
        DocumentBlock(
            block_id=new_block_id(doc_id, 1),
            block_type=BlockType.PARAGRAPH,
            text="学生对下列处理决定不服的，可以提出申诉。",
            heading_path=["学生申诉处理办法", "一、受理范围"],
            source_location=SourceLocation(paragraph_index=1),
        ),
        DocumentBlock(
            block_id=new_block_id(doc_id, 2),
            block_type=BlockType.TABLE,
            table_data=merged_table(),
            heading_path=["学生申诉处理办法", "一、受理范围"],
            source_location=SourceLocation(table_index=0, paragraph_index=2),
        ),
        DocumentBlock(
            block_id=new_block_id(doc_id, 3),
            block_type=BlockType.PARAGRAPH,
            text="以上表格由学生工作部负责解释。",
            heading_path=["学生申诉处理办法", "一、受理范围"],
            source_location=SourceLocation(paragraph_index=3),
        ),
        DocumentBlock(
            block_id=new_block_id(doc_id, 4),
            block_type=BlockType.IMAGE,
            text="申诉流程图",
            image_path="var/parsed_assets/102/flow.png",
            heading_path=["学生申诉处理办法", "一、受理范围"],
            source_location=SourceLocation(paragraph_index=4),
            extra={"ocr_text": "学生提出申诉 → 学院受理 → 学校复核"},
        ),
    ]
    doc = make_doc(
        blocks,
        document_id=doc_id,
        file_name="102_学生申诉处理办法.docx",
        file_type=FileType.DOCX,
        title="学生申诉处理办法",
    )

    types = [block.block_type for block in doc.blocks]
    assert types == [
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.TABLE,
        BlockType.PARAGRAPH,
        BlockType.IMAGE,
    ], "段落与表格必须保持原始交错顺序"

    table_block = doc.blocks[2]
    assert table_block.table_data is not None
    assert table_block.table_data.n_rows == 3
    assert table_block.source_location.table_index == 0
    # 图片引用与 OCR 文本都要保留，且能区分「原始正文」与「模型生成」
    assert doc.blocks[4].image_path.endswith("flow.png")
    assert doc.blocks[4].extra["ocr_text"].startswith("学生提出申诉")


def test_pptx_document_is_representable() -> None:
    """PPT：一张幻灯片是一个组织单位，备注放 extra，不混进正文。"""
    doc_id = document_id_for("test_corpus/pptx/powerpoint_sample.pptx")
    doc = make_doc(
        [
            DocumentBlock(
                block_id=new_block_id(doc_id, 0),
                block_type=BlockType.HEADING,
                text="校园网络安全宣讲",
                heading_level=1,
                heading_path=["校园网络安全宣讲"],
                source_location=SourceLocation(slide_number=1),
            ),
            DocumentBlock(
                block_id=new_block_id(doc_id, 1),
                block_type=BlockType.LIST,
                text="1. 弱口令风险\n2. 钓鱼邮件识别",
                heading_path=["校园网络安全宣讲"],
                source_location=SourceLocation(slide_number=2),
                extra={"ordered": True},
            ),
            DocumentBlock(
                block_id=new_block_id(doc_id, 2),
                block_type=BlockType.TABLE,
                table_data=TableData(
                    n_rows=2,
                    n_cols=2,
                    header_row_count=1,
                    cells=[
                        TableCell(text="风险类型", row=0, col=0, is_header=True),
                        TableCell(text="占比", row=0, col=1, is_header=True),
                        TableCell(text="弱口令", row=1, col=0),
                        TableCell(text="43%", row=1, col=1),
                    ],
                ),
                source_location=SourceLocation(slide_number=2),
            ),
        ],
        document_id=doc_id,
        file_name="powerpoint_sample.pptx",
        file_type=FileType.PPTX,
        title="校园网络安全宣讲",
        metadata=DocumentMetadata(slide_count=2),
    )

    assert doc.metadata.slide_count == 2
    assert [b.source_location.slide_number for b in doc.blocks] == [1, 2, 2]
    assert doc.blocks[1].extra["ordered"] is True


def test_xlsx_document_is_representable() -> None:
    """Excel：多 Sheet、多级表头、cell_range、隐藏行列标记。"""
    doc_id = document_id_for("data/raw/奖学金名额.xlsx")
    sheet1 = TableData(
        n_rows=3,
        n_cols=3,
        header_row_count=2,
        caption="2026 年名额分配",
        cells=[
            TableCell(text="学院", row=0, col=0, rowspan=2, is_header=True),
            TableCell(text="国家奖学金", row=0, col=1, colspan=2, is_header=True),
            TableCell(text="名额", row=1, col=1, is_header=True),
            TableCell(text="金额", row=1, col=2, is_header=True),
            TableCell(text="计算机学院", row=2, col=0),
            TableCell(text="10", row=2, col=1),
            TableCell(text="8000", row=2, col=2),
        ],
    )
    doc = make_doc(
        [
            DocumentBlock(
                block_id=new_block_id(doc_id, 0),
                block_type=BlockType.TABLE,
                table_data=sheet1,
                source_location=SourceLocation(sheet_name="名额分配", cell_range="A1:C3"),
                extra={"hidden_rows": [4], "hidden_cols": ["D"]},
            ),
            DocumentBlock(
                block_id=new_block_id(doc_id, 1),
                block_type=BlockType.TABLE,
                table_data=TableData(
                    n_rows=1,
                    n_cols=2,
                    header_row_count=1,
                    cells=[
                        TableCell(text="项目", row=0, col=0, is_header=True),
                        TableCell(text="金额", row=0, col=1, is_header=True),
                    ],
                ),
                source_location=SourceLocation(sheet_name="其他", cell_range="B2:C2"),
            ),
        ],
        document_id=doc_id,
        file_name="奖学金名额.xlsx",
        file_type=FileType.XLSX,
        title="奖学金名额",
        metadata=DocumentMetadata(sheet_count=2),
    )

    assert doc.metadata.sheet_count == 2
    assert doc.blocks[0].source_location.sheet_name == "名额分配"
    assert doc.blocks[0].source_location.cell_range == "A1:C3"
    assert doc.blocks[0].extra["hidden_rows"] == [4]
    # Excel 不能退化成「无上下文的字符串」：表头信息必须还在
    assert doc.blocks[0].table_data is not None
    assert doc.blocks[0].table_data.header_row_count == 2


# ---------------------------------------------------------------------------
# 5：表格结构完整
# ---------------------------------------------------------------------------


def test_merged_cells_keep_structure_and_expand_correctly() -> None:
    table = merged_table()

    # 原始结构：9 个单元格，合并信息在
    assert len(table.cells) == 9
    span_cell = next(c for c in table.cells if c.text == "国家奖学金")
    assert (span_cell.row, span_cell.col, span_cell.rowspan, span_cell.colspan) == (0, 1, 1, 2)

    grid = table.grid()
    # 展开后，合并区域右下角也应该是同一个值
    assert grid[0][1] == grid[0][2] == "国家奖学金"
    assert grid[2][2] == "8000"

    markdown = table.to_markdown()
    assert "表 1 各学院奖助名额分配" in markdown, "表格标题必须保留"
    assert "国家奖学金 / 名额" in markdown, "多级表头应压成可读的层级文字"
    assert "| 计算机学院 | 10 | 8000 | 20 |" in markdown
    # 生成的是合法 Markdown 表格：表头分隔行
    assert "| --- | --- | --- | --- |" in markdown


def test_table_without_header_still_renders() -> None:
    table = TableData(
        n_rows=2,
        n_cols=2,
        cells=[
            TableCell(text="a", row=0, col=0),
            TableCell(text="b", row=0, col=1),
            TableCell(text="c", row=1, col=0),
            TableCell(text="d", row=1, col=1),
        ],
    )
    md = table.to_markdown()
    assert md.startswith("| a | b |")
    assert "| c | d |" in md


# ---------------------------------------------------------------------------
# 6-7：图片引用、来源位置
# ---------------------------------------------------------------------------


def test_image_keeps_reference_and_source() -> None:
    block = DocumentBlock(
        block_id="b-0001",
        block_type=BlockType.IMAGE,
        image_path="var/parsed_assets/doc123/p3_img1.png",
        source_location=SourceLocation(page_start=3, page_end=3),
        extra={"ocr_text": "图 3 报名流程"},
    )
    rendered = block.render_text()
    assert "var/parsed_assets/doc123/p3_img1.png" in rendered
    assert "图 3 报名流程" in rendered
    assert "图片" in rendered


def test_source_location_flat_dict_has_no_none() -> None:
    loc = SourceLocation(
        page_start=2,
        page_end=4,
        slide_number=None,
        sheet_name=None,
        cell_range="b2:d9",
        char_offset=120,
    )
    flat = loc.to_flat_dict()
    assert flat == {
        "page_start": 2,
        "page_end": 4,
        "cell_range": "B2:D9",
        "char_offset": 120,
    }
    # Chroma 不接受 None，这里必须一个都没有
    assert all(value is not None for value in flat.values())


def test_locality_report_detects_missing_positions() -> None:
    doc = make_doc(
        [
            DocumentBlock(
                block_id="b-0001",
                block_type=BlockType.PARAGRAPH,
                text="有页码",
                source_location=SourceLocation(page_start=1),
            ),
            DocumentBlock(
                block_id="b-0002",
                block_type=BlockType.PARAGRAPH,
                text="没有页码",
            ),
        ]
    )
    report = doc.locality_report()
    assert report == {"总块数": 2, "有位置信息": 1, "缺位置信息": 1, "覆盖率": 0.5}


# ---------------------------------------------------------------------------
# 8：Pydantic 捕获非法数据
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "factory", "match"),
    [
        (
            "页码从 0 开始",
            lambda: SourceLocation(page_start=0),
            "greater than or equal to 1",
        ),
        (
            "page_end 小于 page_start",
            lambda: SourceLocation(page_start=5, page_end=3),
            "不能小于",
        ),
        (
            "只有 page_end",
            lambda: SourceLocation(page_end=3),
            "必须给 page_start",
        ),
        (
            "bbox 坐标倒置",
            lambda: SourceLocation(bbox=(500.0, 700.0, 100.0, 730.0)),
            "顺序反了",
        ),
        (
            "cell_range 不合法",
            lambda: SourceLocation(cell_range="第1行到第9行"),
            "cell_range 格式不合法",
        ),
        (
            "heading 缺 heading_level",
            lambda: DocumentBlock(
                block_id="b-1",
                block_type=BlockType.HEADING,
                text="标题",
                heading_path=["标题"],
            ),
            "heading_level",
        ),
        (
            "heading 的 heading_path 末项与 text 不一致",
            lambda: DocumentBlock(
                block_id="b-1",
                block_type=BlockType.HEADING,
                text="真标题",
                heading_level=1,
                heading_path=["旧标题"],
            ),
            "最后一项",
        ),
        (
            "table 缺 table_data",
            lambda: DocumentBlock(block_id="b-1", block_type=BlockType.TABLE, text="看表"),
            "必须给 table_data",
        ),
        (
            "image 缺 image_path",
            lambda: DocumentBlock(block_id="b-1", block_type=BlockType.IMAGE, text="图"),
            "必须给 image_path",
        ),
        (
            "空块",
            lambda: DocumentBlock(block_id="b-1", block_type=BlockType.PARAGRAPH),
            "没有任何内容载体",
        ),
        (
            "block_id 为空",
            lambda: DocumentBlock(block_id="", block_type=BlockType.PARAGRAPH, text="x"),
            "at least 1 character",
        ),
        (
            "表格单元格越界",
            lambda: TableData(
                n_rows=1,
                n_cols=1,
                cells=[TableCell(text="x", row=0, col=0, rowspan=2)],
            ),
            "纵向越界",
        ),
        (
            "表格单元格重叠",
            lambda: TableData(
                n_rows=2,
                n_cols=2,
                cells=[
                    TableCell(text="a", row=0, col=0, rowspan=2),
                    TableCell(text="b", row=1, col=0, colspan=2),
                ],
            ),
            "单元格重叠",
        ),
        (
            "表头行数超过总行数",
            lambda: TableData(n_rows=1, n_cols=1, header_row_count=2),
            "表头行数",
        ),
        (
            "sha256 长度不对",
            lambda: DocumentMetadata(sha256="abc123"),
            "64 位十六进制",
        ),
    ],
)
def test_invalid_values_are_rejected(name: str, factory, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        factory()


def test_duplicate_block_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="block_id 重复"):
        make_doc(
            [
                DocumentBlock(block_id="b-1", block_type=BlockType.PARAGRAPH, text="甲"),
                DocumentBlock(block_id="b-1", block_type=BlockType.PARAGRAPH, text="乙"),
            ]
        )


@pytest.mark.parametrize(
    ("name", "kwargs", "match"),
    [
        (
            "空文档却标 completed",
            {"blocks": [], "parse_status": ParseStatus.COMPLETED},
            "completed 但没有任何内容块",
        ),
        (
            "有块却标 empty",
            {
                "blocks": [
                    DocumentBlock(block_id="b-1", block_type=BlockType.PARAGRAPH, text="x")
                ],
                "parse_status": ParseStatus.EMPTY,
            },
            "状态与内容矛盾",
        ),
        (
            "partial 但没有错误记录",
            {
                "blocks": [
                    DocumentBlock(block_id="b-1", block_type=BlockType.PARAGRAPH, text="x")
                ],
                "parse_status": ParseStatus.PARTIAL,
            },
            "没有任何错误记录",
        ),
        (
            "failed 但没有 fatal 错误",
            {
                "blocks": [],
                "parse_status": ParseStatus.FAILED,
                "errors": [ParseError(stage="parse", message="表坏了")],
            },
            "没有任何 fatal 错误",
        ),
    ],
)
def test_status_consistency_is_enforced(name: str, kwargs: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        make_doc(**kwargs)


def test_partial_and_failed_happy_paths() -> None:
    """状态机本身要可表达：PARTIAL 能描述「正文拿到了、表格没拿到」。"""
    partial = make_doc(
        [DocumentBlock(block_id="b-1", block_type=BlockType.PARAGRAPH, text="正文")],
        parse_status=ParseStatus.PARTIAL,
        errors=[ParseError(stage="table", message="第 2 页表格解析失败", fatal=False)],
    )
    assert partial.parse_status is ParseStatus.PARTIAL

    failed = ParsedDocument(
        document_id=document_id_for("bad.pdf"),
        file_name="bad.pdf",
        file_type=FileType.PDF,
        parser_name="pdf",
        blocks=[],
        parse_status=ParseStatus.FAILED,
        errors=[ParseError(stage="open", message="文件加密，无法打开", fatal=True)],
    )
    assert failed.summary()["致命错误数"] == 1


# ---------------------------------------------------------------------------
# ID 稳定性与序列化
# ---------------------------------------------------------------------------


def test_document_id_is_stable_and_path_normalized() -> None:
    a = document_id_for("data/raw/foo.pdf")
    b = document_id_for("data\\raw\\foo.pdf")
    c = document_id_for("  data/raw/foo.pdf  ")
    d = document_id_for("data/raw/bar.pdf")

    assert a == b == c, "路径分隔符和首尾空格不应改变文档身份"
    assert a != d
    assert len(a) == 16

    with pytest.raises(ValueError, match="不能为空"):
        document_id_for("")


def test_block_id_is_traceable() -> None:
    doc_id = document_id_for("data/raw/foo.pdf")
    assert new_block_id(doc_id, 7) == f"{doc_id}-b0007"


def test_document_serializes_to_json() -> None:
    """解析产物要能落盘成 JSON（阶段三的验收产物、阶段六的失败重试都要用）。"""
    doc = make_doc(
        [
            DocumentBlock(
                block_id="b-1",
                block_type=BlockType.TABLE,
                table_data=merged_table(),
                source_location=SourceLocation(page_start=1, bbox=(1.0, 2.0, 3.0, 4.0)),
            )
        ]
    )
    payload = json.loads(doc.model_dump_json())
    assert payload["file_type"] == "pdf"
    assert payload["blocks"][0]["table_data"]["cells"][0]["rowspan"] == 2
    # 反序列化能原样回来，才说明模型是自洽的
    restored = ParsedDocument.model_validate(payload)
    assert restored.to_text() == doc.to_text()


def test_to_text_preserves_order_and_renders_table() -> None:
    doc = make_doc(
        [
            DocumentBlock(
                block_id="b-1",
                block_type=BlockType.HEADING,
                text="一、申请条件",
                heading_level=2,
                heading_path=["通知", "一、申请条件"],
            ),
            DocumentBlock(
                block_id="b-2",
                block_type=BlockType.PARAGRAPH,
                text="具有中华人民共和国国籍。",
                heading_path=["通知", "一、申请条件"],
            ),
            DocumentBlock(
                block_id="b-3",
                block_type=BlockType.TABLE,
                table_data=merged_table(),
                heading_path=["通知", "一、申请条件"],
            ),
        ]
    )
    text = doc.to_text()
    assert text.startswith("## 一、申请条件")
    assert text.index("具有中华人民共和国国籍") < text.index("计算机学院")
    assert "| 学院 |" in text


def test_unknown_suffix_maps_to_unknown() -> None:
    assert FileType.from_suffix(".PDF") is FileType.PDF
    assert FileType.from_suffix("xlsx") is FileType.XLSX
    assert FileType.from_suffix(".doc") is FileType.UNKNOWN


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
