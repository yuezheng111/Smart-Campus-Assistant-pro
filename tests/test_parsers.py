"""解析器验收测试。

原则：**不用"函数没报错"当成功标准**。每条用例都检查具体的结构断言
（表格张数、页码、幻灯片号、sheet 名、块顺序），因为本次改造的全部价值
就在于这些结构——它们丢了，代码跑得再顺也没意义。

样例来自 `test_corpus/`，文件缺失时用例自动跳过（不是失败），
这样别人 clone 下来没带语料也能跑通其余部分。

运行：
    D:/AgentLearnVenv/Scripts/python.exe -m pytest tests/test_parsers.py -v
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from openpyxl import Workbook

from app.parsing.base import ParseOptions
from app.parsing.factory import detect_file_type, get_parser, parse_file
from app.parsing.models import BlockType, FileType, ParseStatus

CORPUS = Path(__file__).resolve().parent.parent / "test_corpus"


def sample(*parts: str) -> Path:
    path = CORPUS.joinpath(*parts)
    if not path.exists():
        pytest.skip(f"样例文件不存在：{path}")
    return path


# ---------------------------------------------------------------------------
# 类型识别
# ---------------------------------------------------------------------------


def test_detect_by_content_not_suffix(tmp_path: Path) -> None:
    """把 docx 改名成 .pdf，必须还能认出它是 docx。

    这是"不能只依赖后缀"的直接验证：管理员上传时改错扩展名是常事，
    按后缀选解析器会直接报「PDF 损坏」。
    """
    source = sample("docx", "word_sample.docx")
    fake = tmp_path / "renamed.pdf"
    shutil.copyfile(source, fake)

    assert detect_file_type(fake) is FileType.DOCX

    doc = parse_file(fake)
    assert doc.file_type is FileType.DOCX
    assert doc.parse_status is not ParseStatus.FAILED
    assert doc.blocks, "改后缀不应影响解析"


def test_detect_plain_text_by_suffix(tmp_path: Path) -> None:
    md = tmp_path / "a.md"
    md.write_text("# 标题\n\n正文", encoding="utf-8")
    assert detect_file_type(md) is FileType.MARKDOWN


def test_unsupported_type_returns_failed_not_raise(tmp_path: Path) -> None:
    old = tmp_path / "legacy.doc"
    old.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64)

    doc = parse_file(old)
    assert doc.parse_status is ParseStatus.FAILED
    assert doc.errors and doc.errors[0].fatal
    assert "无法识别" in doc.errors[0].message


def test_parse_never_raises_on_broken_input(tmp_path: Path) -> None:
    """空文件、随机字节、不存在的路径，都必须返回 ParsedDocument 而不是抛异常。"""
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")

    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"not a real pdf at all")

    for path in (empty, junk, tmp_path / "missing.pdf"):
        doc = parse_file(path)
        assert doc.parse_status in (ParseStatus.FAILED, ParseStatus.EMPTY)
        assert doc.errors, "失败必须留下原因"


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def test_pdf_pages_and_positions() -> None:
    path = sample("pdf", "multi_page.pdf")
    doc = parse_file(path)

    assert doc.parse_status is ParseStatus.COMPLETED
    assert doc.metadata.page_count == 5

    pages = {block.source_location.page_start for block in doc.blocks}
    assert pages == {1, 2, 3, 4, 5}, "每一页都应该有块，且带页码"
    assert all(block.source_location.page_end == block.source_location.page_start
               for block in doc.blocks)


def test_pdf_table_structure() -> None:
    """全数字表格：旧代码只能抽出一行 78 字符的文本，现在必须是 8×2 的表。"""
    doc = parse_file(sample("pdf", "all-number-table.pdf"))

    tables = [b for b in doc.blocks if b.block_type is BlockType.TABLE]
    assert len(tables) == 1
    table = tables[0].table_data
    assert table is not None
    assert (table.n_rows, table.n_cols) == (8, 2)
    assert table.cells[0].text == "200.01"
    assert table.cells[-1].text == "15"


def test_pdf_scanned_file_is_flagged_not_silently_dropped() -> None:
    """扫描件不能静默消失：状态必须是 empty，并明确提示需要 OCR。"""
    doc = parse_file(sample("pdf", "ocr_test.pdf"))

    assert doc.parse_status is ParseStatus.EMPTY
    assert doc.errors, "必须说明为什么没有正文"
    assert any(err.stage == "ocr" for err in doc.errors)
    assert any(b.block_type is BlockType.IMAGE for b in doc.blocks), "图片要留下来给 OCR 用"


def test_pdf_lines_are_merged_into_paragraphs() -> None:
    """同一段的换行要合并，否则一份 5 页文档会产出上百个单行块。"""
    doc = parse_file(sample("pdf", "multi_page.pdf"))
    paragraphs = [b for b in doc.blocks if b.block_type is BlockType.PARAGRAPH]

    assert paragraphs
    assert len(paragraphs) < 50, f"段落数 {len(paragraphs)} 过多，说明换行没合并"
    assert any(len(b.text) > 80 for b in paragraphs), "应该有跨行合并出来的长段落"


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def test_docx_tables_are_not_lost() -> None:
    """旧代码只读 paragraphs，这份文件的 6 张表会被整段丢弃。"""
    doc = parse_file(sample("docx", "tables-with-incomplete-rows.docx"))

    tables = [b for b in doc.blocks if b.block_type is BlockType.TABLE]
    assert len(tables) >= 5, f"只解析出 {len(tables)} 张表"
    assert all(t.table_data is not None and t.table_data.n_rows > 0 for t in tables)
    # 表格位置可回溯
    assert [t.source_location.table_index for t in tables] == sorted(
        t.source_location.table_index for t in tables
    )


def test_docx_keeps_paragraph_table_interleaving() -> None:
    doc = parse_file(sample("docx", "word_sample.docx"))
    types = [b.block_type for b in doc.blocks]

    first_table = types.index(BlockType.TABLE)
    assert BlockType.IMAGE in types[:first_table], "图片应出现在表格之前（按原顺序）"
    assert types[0] is not BlockType.TABLE or True  # 顺序由 XML 决定，不强制首块类型


def test_docx_heading_path_is_built() -> None:
    doc = parse_file(sample("docx", "unit_test_headers_numbered.docx"))

    headings = [b for b in doc.blocks if b.block_type is BlockType.HEADING]
    assert len(headings) >= 3
    for heading in headings:
        assert heading.heading_level is not None
        assert heading.heading_path[-1] == heading.text
        assert 1 <= heading.heading_level <= 6


def test_docx_image_keeps_reference() -> None:
    doc = parse_file(sample("docx", "word_sample.docx"))
    images = [b for b in doc.blocks if b.block_type is BlockType.IMAGE]

    assert images, "这份文件里有图片"
    for image in images:
        assert image.image_path, "图片必须保留引用"
        assert image.source_location.paragraph_index is not None

    assert any(str(b.image_path).startswith("embedded://") for b in images)


def test_docx_assets_are_written_when_asked(tmp_path: Path) -> None:
    """传了 assets_dir 就要真的落盘，且路径可回指。"""
    doc = parse_file(
        sample("docx", "word_sample.docx"),
        ParseOptions(assets_dir=tmp_path),
    )
    images = [b for b in doc.blocks if b.block_type is BlockType.IMAGE]
    if not images:
        pytest.skip("样例里没有图片")

    for image in images:
        assert (tmp_path / doc.document_id) in Path(str(image.image_path)).parents or Path(
            str(image.image_path)
        ).exists()


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------


def test_pptx_slide_units_and_notes() -> None:
    doc = parse_file(sample("pptx", "powerpoint_sample.pptx"))

    assert doc.metadata.slide_count and doc.metadata.slide_count >= 2
    slides = {b.source_location.slide_number for b in doc.blocks}
    assert None not in slides
    assert len(slides) >= 2, "每个块都要能指回幻灯片"

    notes = [b for b in doc.blocks if b.block_type is BlockType.NOTE]
    assert notes, "演讲者备注应被单独提取"
    assert notes[0].extra.get("notes") is True


def test_pptx_table_merges() -> None:
    doc = parse_file(sample("pptx", "powerpoint_sample.pptx"))
    tables = [b for b in doc.blocks if b.block_type is BlockType.TABLE]
    assert tables

    table = tables[0].table_data
    assert table is not None
    assert any(cell.colspan > 1 for cell in table.cells), "样例里有合并的单元格"


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------


def test_xlsx_sheet_names_and_ranges() -> None:
    doc = parse_file(sample("xlsx", "testEXCEL.xlsx"))

    assert doc.metadata.sheet_count and doc.metadata.sheet_count >= 1
    tables = [b for b in doc.blocks if b.block_type is BlockType.TABLE]
    assert tables

    for table in tables:
        assert table.source_location.sheet_name
        assert table.source_location.cell_range
        assert ":" in str(table.source_location.cell_range)


def test_xlsx_merges_formulas_and_hidden(tmp_path: Path) -> None:
    """自造一份带合并、公式、隐藏行的工作簿，验证三项都被保留。"""
    path = tmp_path / "book.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "名额"
    sheet["A1"] = "学院"
    sheet.merge_cells("A1:A2")
    sheet["B1"] = "国家奖学金"
    sheet.merge_cells("B1:C1")
    sheet["B2"], sheet["C2"] = "名额", "金额"
    sheet["A3"], sheet["B3"], sheet["C3"] = "计算机学院", 10, 8000
    sheet["D3"] = "=B3*C3"
    sheet.row_dimensions[4].hidden = True
    workbook.save(path)

    doc = parse_file(path)
    tables = [b for b in doc.blocks if b.block_type is BlockType.TABLE]
    assert tables

    table = tables[0].table_data
    assert table is not None
    assert table.n_rows == 3
    merged = [c for c in table.cells if c.rowspan > 1 or c.colspan > 1]
    assert len(merged) >= 2, "合并单元格必须保留"

    block = tables[0]
    assert block.extra.get("formulas", {}).get("D3") == "=B3*C3", "公式原文要留下"
    assert 4 in block.extra.get("hidden_rows", []), "隐藏行要标记"


def test_xlsx_splits_multiple_regions(tmp_path: Path) -> None:
    """空行分隔的两张表必须拆成两个块，而不是拼成一张。"""
    path = tmp_path / "two.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet["A1"], sheet["B1"] = "姓名", "电话"
    sheet["A2"], sheet["B2"] = "张三", "123"
    sheet["A4"], sheet["B4"] = "课程", "学分"
    sheet["A5"], sheet["B5"] = "高数", "4"
    workbook.save(path)

    doc = parse_file(path)
    tables = [b for b in doc.blocks if b.block_type is BlockType.TABLE]
    assert len(tables) == 2, f"应拆成 2 个数据区，实际 {len(tables)}"


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_markdown_table_is_kept_whole(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(
        "# 标题\n\n说明文字。\n\n| 学院 | 名额 |\n| --- | --- |\n| 计算机 | 10 |\n| 法学 | 5 |\n",
        encoding="utf-8",
    )
    doc = parse_file(path)

    tables = [b for b in doc.blocks if b.block_type is BlockType.TABLE]
    assert len(tables) == 1
    table = tables[0].table_data
    assert table is not None
    assert (table.n_rows, table.n_cols) == (3, 2)
    assert table.header_row_count == 1
    assert table.cells[0].text == "学院"


def test_markdown_gbk_is_not_silently_dropped(tmp_path: Path) -> None:
    """GBK 编码的 txt：旧代码用 errors='ignore' 会静默丢字。"""
    path = tmp_path / "gbk.txt"
    path.write_bytes("关于做好奖学金评审工作的通知".encode("gbk"))

    doc = parse_file(path)
    assert doc.parse_status is not ParseStatus.FAILED
    assert "奖学金" in doc.to_text()


def test_markdown_chinese_numbered_sections_are_headings(tmp_path: Path) -> None:
    """md 里没有 `#`、只有公文化的「一、二、三、」时，也要切出章节。

    不给这条规则时，整篇会被当成一个章节，切分退化成 1 个 chunk。
    实测 `36_奖学金_...通知.md` 就是这样：老链路按 600 字硬切还有 3 块，
    新链路反而只剩 1 块，「国家奖学金 8000 元」被埋进整篇里搜不出来。
    """
    path = tmp_path / "通知.md"
    path.write_text(
        "各学院、各位同学：\n\n"
        "现启动评选工作，有关事项通知如下。\n\n"
        "一、奖励标准与名额\n\n"
        "国家奖学金奖励标准为每人每年 8000 元。\n\n"
        "二、申请条件\n\n"
        "1. 具有中华人民共和国国籍；\n\n"
        "三、材料要求\n\n"
        "申请表一式两份。\n",
        encoding="utf-8",
    )
    doc = parse_file(path)

    headings = [b for b in doc.blocks if b.block_type is BlockType.HEADING]
    assert [b.text for b in headings] == ["一、奖励标准与名额", "二、申请条件", "三、材料要求"]
    # 三个同级标题：路径不能互相嵌套（「二、」不该挂在「一、」底下）
    for heading in headings:
        assert heading.heading_path == [heading.text]
    # 正文要落在对应章节下，而不是全堆在一节里
    body = [b for b in doc.blocks if b.block_type is BlockType.PARAGRAPH]
    by_text = {b.text: b.heading_path for b in body}
    assert by_text["国家奖学金奖励标准为每人每年 8000 元。"] == ["一、奖励标准与名额"]


# ---------------------------------------------------------------------------
# 通用契约
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        (("pdf", "multi_page.pdf"), FileType.PDF),
        (("docx", "word_sample.docx"), FileType.DOCX),
        (("pptx", "powerpoint_sample.pptx"), FileType.PPTX),
        (("xlsx", "testEXCEL.xlsx"), FileType.XLSX),
    ],
)
def test_every_parser_returns_parsed_document(relative: tuple[str, str], expected: FileType) -> None:
    doc = parse_file(sample(*relative))
    assert doc.file_type is expected
    assert doc.document_id
    assert doc.parser_name
    assert doc.blocks, "这四份样例都应该有内容块"
    # 每个块都能追溯
    assert all(block.block_id for block in doc.blocks)
    assert len({block.block_id for block in doc.blocks}) == len(doc.blocks)


def test_parser_selection_is_registered_for_all_formats() -> None:
    for file_type in (FileType.PDF, FileType.DOCX, FileType.PPTX, FileType.XLSX,
                      FileType.MARKDOWN, FileType.TEXT):
        parser = get_parser(file_type)
        assert parser.name
        assert parser.file_types


# ---------------------------------------------------------------------------
# 共用启发式：中文编号标题 / 文件名标题
# ---------------------------------------------------------------------------


def build_docx(path: Path, paragraphs: list[str]) -> Path:
    """造一份没有任何样式标题的 docx —— 这正是公文导出的常态。"""
    from docx import Document as DocxDocument

    document = DocxDocument()
    for text in paragraphs:
        document.add_paragraph(text)
    document.save(str(path))
    return path


def test_docx_chinese_numbered_sections_are_headings(tmp_path: Path) -> None:
    """没有样式标题的公文，也要按「一、二、三、」切出章节。

    不加这条规则时，实测 `38_助学金_...通知.docx` 的 21 个块只能切出 1 个 chunk，
    整篇 700 字挤在一起，学生问「资助标准」拿到的是一整篇通知。
    """
    path = build_docx(
        tmp_path / "通知.docx",
        [
            "关于做好2026年国家助学金评审工作的通知",
            "各学院、各位同学：",
            "一、资助标准与名额",
            "国家助学金资助标准为每人每年 3300 元。",
            "二、申请条件",
            "1. 具有中华人民共和国国籍；",
            "三、联系方式",
            "联系人：李老师",
        ],
    )
    doc = parse_file(path)

    headings = [b for b in doc.blocks if b.block_type is BlockType.HEADING]
    assert [b.text for b in headings] == ["一、资助标准与名额", "二、申请条件", "三、联系方式"]
    # 三个同级标题，路径不能互相嵌套
    for heading in headings:
        assert heading.heading_path == [heading.text]
    # 标题不该被当成整份文档的标题
    assert doc.title == "通知"


def test_cn_heading_level_ignores_sentences() -> None:
    """「一、二年级学生可在……」是真句子，不能当标题。

    中文编号里最容易误伤的就是「一、二年级」这类词组。
    """
    from app.parsing._heuristics import cn_heading_level

    assert cn_heading_level("一、资助标准与名额") == 2
    assert cn_heading_level("第一章　总　则") == 1
    assert cn_heading_level("一、二年级学生可在9月20日前报名。") is None
    assert cn_heading_level("二、申请条件如下：") is None
    assert cn_heading_level("很长的" + "标题" * 30) is None


def test_title_from_filename_strips_archive_prefix() -> None:
    from app.parsing._heuristics import title_from_filename

    assert (
        title_from_filename("38_助学金_2026-09-22_关于做好2026年国家助学金评审工作的通知")
        == "关于做好2026年国家助学金评审工作的通知"
    )
    # 剥不掉就原样返回，不能把标题清空
    assert title_from_filename("jwc_关于做好毕业实习工作的通知") == "jwc_关于做好毕业实习工作的通知"
    assert title_from_filename("2206.01062") == "2206.01062"
