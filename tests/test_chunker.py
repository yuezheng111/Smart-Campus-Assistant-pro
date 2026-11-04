"""结构感知切分的验收测试。

重点不是"能切出 chunk"，而是四件会影响学生实际体验的事：

    1. **表格带着表头被切开**，任何一块单独拿出来都能读懂；
    2. **章节归属不错位** —— 前缀里写的是这一块真正的章节；
    3. **不产出垃圾块** —— 一字不差的模板页只留一份，40 字的碎片不单独入库；
    4. **编号连续**，前端按序号追溯「第几块」时对得上。

运行：
    D:/AgentLearnVenv/Scripts/python.exe -m pytest tests/test_chunker.py -v
"""

from __future__ import annotations

import pytest

from app.parsing import (
    BlockType,
    DocumentBlock,
    FileType,
    ParsedDocument,
    ParseStatus,
    SourceLocation,
    TableCell,
    TableData,
    document_id_for,
    new_block_id,
)
from app.parsing._heuristics import normalize_heading_paths
from app.parsing.chunker import ChunkingOptions, chunk_document, chunk_stats

DOC_ID = document_id_for("demo.pdf")
TITLE = "关于做好2026年奖学金评审工作的通知"


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def block(
    index: int,
    text: str,
    *,
    page: int = 1,
    block_type: BlockType = BlockType.PARAGRAPH,
    heading_path: list[str] | None = None,
    heading_level: int | None = None,
    extra: dict | None = None,
    image_path: str | None = None,
) -> DocumentBlock:
    return DocumentBlock(
        block_id=new_block_id(DOC_ID, index),
        block_type=block_type,
        text=text,
        heading_level=heading_level,
        heading_path=heading_path or [],
        image_path=image_path,
        source_location=SourceLocation(page_start=page, page_end=page),
        extra=extra or {},
    )


def heading(index: int, text: str, level: int, path: list[str], page: int = 1) -> DocumentBlock:
    return block(
        index, text, page=page,
        block_type=BlockType.HEADING, heading_path=path, heading_level=level,
    )


def table_block(
    index: int,
    rows: list[list[str]],
    *,
    header_rows: int = 1,
    caption: str = "",
    heading_path: list[str] | None = None,
    page: int = 1,
) -> DocumentBlock:
    n_rows = len(rows)
    n_cols = len(rows[0])
    cells = [
        TableCell(text=value, row=r, col=c, is_header=r < header_rows)
        for r, row in enumerate(rows)
        for c, value in enumerate(row)
    ]
    return DocumentBlock(
        block_id=new_block_id(DOC_ID, index),
        block_type=BlockType.TABLE,
        table_data=TableData(
            n_rows=n_rows,
            n_cols=n_cols,
            cells=cells,
            caption=caption,
            header_row_count=header_rows,
        ),
        heading_path=heading_path or [],
        source_location=SourceLocation(page_start=page, page_end=page),
    )


def make_doc(blocks: list[DocumentBlock], title: str = TITLE) -> ParsedDocument:
    return ParsedDocument(
        document_id=DOC_ID,
        file_name="demo.pdf",
        file_type=FileType.PDF,
        title=title,
        blocks=blocks,
        parser_name="fake",
        parse_status=ParseStatus.COMPLETED if blocks else ParseStatus.EMPTY,
    )


def long_text(length: int, seed: str = "报名") -> str:
    """造一段长度可控的中文正文。"""
    unit = f"{seed}截止时间为2026年9月30日。"
    return (unit * (length // len(unit) + 1))[:length]


# ---------------------------------------------------------------------------
# 表格：结构不能丢
# ---------------------------------------------------------------------------


def test_table_chunk_keeps_header_and_prefix() -> None:
    """整张表进一个 chunk：表头在、分隔行在、章节前缀在。"""
    doc = make_doc(
        [
            heading(0, "一、奖励标准", 2, [TITLE, "一、奖励标准"]),
            table_block(
                1,
                [["奖项", "名额", "金额"], ["国家奖学金", "10", "8000"]],
                heading_path=[TITLE, "一、奖励标准"],
            ),
        ]
    )
    chunks = chunk_document(doc)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.content_type == "table"
    assert chunk.text.startswith(f"【{TITLE} > 一、奖励标准】")
    assert "| 奖项 | 名额 | 金额 |" in chunk.text
    assert "| --- | --- | --- |" in chunk.text
    assert "| 国家奖学金 | 10 | 8000 |" in chunk.text


def test_long_table_is_split_by_rows_and_every_piece_has_header() -> None:
    """超长表按行切开后，**每一块都要重新挂表头**。

    这是最初那个 bug 的回归测试：旧流程把 9 行表格切成「9 行 + 3 行」两块，
    第二块的首行是数据行，学生问「答辩时间」时模型看到的是没有列名的一堆日期。
    """
    rows = [["环节", "时间", "负责单位"]]
    rows += [[f"环节{i}", f"2026-05-{i:02d}", "教务处"] for i in range(1, 40)]
    doc = make_doc(
        [
            heading(0, "二、工作安排", 2, [TITLE, "二、工作安排"]),
            table_block(1, rows, heading_path=[TITLE, "二、工作安排"]),
        ]
    )
    chunks = chunk_document(doc, ChunkingOptions(table_max_chars=200))

    assert len(chunks) > 1
    for chunk in chunks:
        assert "| 环节 | 时间 | 负责单位 |" in chunk.text
        assert "| --- | --- | --- |" in chunk.text
        assert chunk.content_type == "table"


def test_long_table_row_range_metadata_covers_body() -> None:
    """切开的每一块都要记清自己是第几行到第几行，且首尾相接、不重不漏。"""
    rows = [["序号", "事项"]]
    rows += [[str(i), f"事项{i}的说明文字"] for i in range(1, 60)]
    doc = make_doc([table_block(0, rows)])
    chunks = chunk_document(doc, ChunkingOptions(table_max_chars=160))

    ranges = [(c.metadata["table_row_start"], c.metadata["table_row_end"]) for c in chunks]
    assert ranges[0][0] == 1  # 表头占第 0 行，正文从第 1 行开始
    assert ranges[-1][1] == len(rows) - 1
    for previous, current in zip(ranges, ranges[1:]):
        assert current[0] == previous[1] + 1
    for chunk in chunks:
        assert chunk.metadata["table_rows"] == len(rows)
        assert chunk.metadata["table_header_rows"] == 1


def test_table_caption_goes_before_table() -> None:
    doc = make_doc([
        table_block(0, [["项目", "金额"], ["学费", "5000"]], caption="表1 收费标准"),
    ])
    chunk = chunk_document(doc)[0]
    assert "表1 收费标准" in chunk.text
    assert chunk.text.index("表1 收费标准") < chunk.text.index("| 项目 |")


# ---------------------------------------------------------------------------
# 章节归属
# ---------------------------------------------------------------------------


def test_paragraphs_are_not_merged_across_sections() -> None:
    doc = make_doc([
        heading(0, "一、奖励标准", 2, [TITLE, "一、奖励标准"]),
        block(1, long_text(300, "奖励"), heading_path=[TITLE, "一、奖励标准"]),
        heading(2, "二、申请条件", 2, [TITLE, "二、申请条件"]),
        block(3, long_text(300, "条件"), heading_path=[TITLE, "二、申请条件"]),
    ])
    chunks = chunk_document(doc)

    assert len(chunks) == 2
    assert "奖励标准" in chunks[0].text.splitlines()[0]
    assert "申请条件" in chunks[1].text.splitlines()[0]
    assert chunks[0].text.splitlines()[0].count("二、申请条件") == 0


def test_prefix_falls_back_to_title_without_heading() -> None:
    doc = make_doc([block(0, long_text(200))])
    chunk = chunk_document(doc)[0]
    assert chunk.text.startswith(f"【{TITLE}】")


def test_cross_page_chunk_reports_page_range() -> None:
    doc = make_doc([
        heading(0, "三、材料要求", 2, [TITLE, "三、材料要求"]),
        block(1, long_text(200, "材料"), page=1, heading_path=[TITLE, "三、材料要求"]),
        block(2, long_text(200, "补充"), page=3, heading_path=[TITLE, "三、材料要求"]),
    ])
    chunk = chunk_document(doc)[0]
    assert chunk.source_location.page_start == 1
    assert chunk.source_location.page_end == 3


# ---------------------------------------------------------------------------
# 标题层级归一化（切分前缀的上游）
# ---------------------------------------------------------------------------


def test_normalize_treats_same_level_headings_as_siblings() -> None:
    """没有第 1 级标题时，「一、」「二、」是同级的，不能叠成父子。

    实测来源：`37_奖学金_...通知.pdf` 全文没有一级标题，两个同级小节
    被拼成 `["一、", "二、"]`，切出来的前缀写成「一、奖励标准 > 二、申请条件」。
    """
    first = heading(0, "一、奖励标准与名额", 2, ["一、奖励标准与名额"])
    second = heading(1, "二、申请条件", 2, ["一、奖励标准与名额", "二、申请条件"])
    blocks = [first, second, block(2, "正文", heading_path=second.heading_path)]

    normalize_heading_paths(blocks)

    assert first.heading_path == ["一、奖励标准与名额"]
    assert second.heading_path == ["二、申请条件"]
    assert blocks[2].heading_path == ["二、申请条件"]
    # 最浅的标题被抬成第 1 级
    assert first.heading_level == 1
    assert second.heading_level == 1


def test_normalize_keeps_nested_levels() -> None:
    top = heading(0, "第一章 总则", 1, ["第一章 总则"])
    sub = heading(1, "一、适用对象", 2, ["第一章 总则", "一、适用对象"])
    leaf = heading(2, "（一）本科生", 3, ["第一章 总则", "一、适用对象", "（一）本科生"])
    after = heading(3, "二、解释权", 2, ["第一章 总则", "二、解释权"])
    blocks = [top, sub, leaf, after, block(4, "正文")]

    normalize_heading_paths(blocks)

    assert leaf.heading_path == ["第一章 总则", "一、适用对象", "（一）本科生"]
    assert after.heading_path == ["第一章 总则", "二、解释权"]
    assert top.heading_level == 1 and sub.heading_level == 2


def test_normalize_is_noop_without_headings() -> None:
    """xlsx 用 sheet 名当路径、pptx 用幻灯片标题，没有标题块时不能清空路径。"""
    only = block(0, "内容", heading_path=["Sheet1"])
    normalize_heading_paths([only])
    assert only.heading_path == ["Sheet1"]


# ---------------------------------------------------------------------------
# 去重与合并
# ---------------------------------------------------------------------------


def test_duplicate_chunks_are_dropped() -> None:
    """模板文档里一字不差的表格只留一份。

    实测来源：`jwc_...毕业实习工作的通知.docx` 是一本实习记录本，
    40 页模板切出 82 个 chunk，其中 38 个是同一张「实习日记」表 ——
    不去重就会让一次检索的 top_k 全被同一段文字占满。
    """
    rows = [["实习部门或岗位："], ["实习内容："], ["实习记事和体会"]]
    doc = make_doc([table_block(i, rows) for i in range(6)])
    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_duplicate_chunks_kept_when_dedupe_disabled() -> None:
    rows = [["实习部门或岗位："], ["实习内容："]]
    doc = make_doc([table_block(i, rows) for i in range(3)])
    chunks = chunk_document(doc, ChunkingOptions(dedupe=False, min_chars=1))

    assert len(chunks) == 3
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_short_table_is_merged_into_next_table_of_same_section() -> None:
    """同一节里两张小表挨着，前一张短到没信息量时并进后一张。"""
    doc = make_doc(
        [
            heading(0, "四、联系方式", 2, [TITLE, "四、联系方式"]),
            table_block(1, [["联系人"], ["教务处"]], heading_path=[TITLE, "四、联系方式"]),
            table_block(
                2,
                [["电话"], ["020-84096000"], ["邮箱"], ["jwc@gdufe.edu.cn"]],
                heading_path=[TITLE, "四、联系方式"],
            ),
        ]
    )
    chunks = chunk_document(doc, ChunkingOptions(min_chars=200))

    assert len(chunks) == 1
    assert "教务处" in chunks[0].text
    assert "020-84096000" in chunks[0].text


def test_short_chunk_is_not_merged_across_sections() -> None:
    """跨章节合并会让「参考章节」和内容对不上，比短一点更糟。"""
    doc = make_doc([
        heading(0, "五、附则", 2, [TITLE, "五、附则"]),
        block(1, "本办法由教务处负责解释。", heading_path=[TITLE, "五、附则"]),
        heading(2, "六、附件", 2, [TITLE, "六、附件"]),
        block(3, long_text(400, "附件"), heading_path=[TITLE, "六、附件"]),
    ])
    chunks = chunk_document(doc)

    assert len(chunks) == 2
    assert "附则" in chunks[0].text.splitlines()[0]
    assert "负责解释" in chunks[0].text
    assert "附件" not in chunks[0].text.splitlines()[0]


def test_short_paragraph_is_merged_back_into_same_section_past_a_table() -> None:
    """同一节被一张表隔成「表前正文 / 表后短尾巴」，尾巴要并回本节的正文块。

    回看窗口的意义就在这里：中间隔着一个表格 chunk，但它仍属同一节。
    """
    doc = make_doc(
        [
            heading(0, "三、评审程序", 2, [TITLE, "三、评审程序"]),
            block(1, long_text(400, "程序"), heading_path=[TITLE, "三、评审程序"]),
            table_block(
                2,
                [["阶段", "时间"], ["学院初评", "9月20日"]],
                heading_path=[TITLE, "三、评审程序"],
            ),
            block(3, "逾期不再受理。", heading_path=[TITLE, "三、评审程序"]),
        ]
    )
    chunks = chunk_document(doc, ChunkingOptions(min_chars=200))

    paragraph_chunks = [c for c in chunks if c.content_type == "paragraph"]
    assert len(paragraph_chunks) == 1
    assert "逾期不再受理" in paragraph_chunks[0].text


def test_chunk_index_and_id_are_continuous_after_cleanup() -> None:
    """去重和合并会留下空洞，编号必须重排成 0..n-1。"""
    rows = [["序号", "事项"]]
    rows += [[str(i), f"事项{i}的说明文字"] for i in range(1, 40)]
    doc = make_doc([
        table_block(0, rows),
        table_block(1, rows),  # 与上一张完全相同，会被去重
    ])
    chunks = chunk_document(doc, ChunkingOptions(table_max_chars=180))

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert [c.chunk_id for c in chunks] == [
        f"{DOC_ID}-c{i:04d}" for i in range(len(chunks))
    ]
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_ids_are_stable_across_runs() -> None:
    """同样的输入切两次要得到同样的 ID —— 这是「重复入库幂等」的前提。"""
    doc = make_doc([block(0, long_text(300))])
    first = chunk_document(doc)
    second = chunk_document(doc)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


# ---------------------------------------------------------------------------
# 图片、页眉页脚、超长段落
# ---------------------------------------------------------------------------


def test_image_block_without_ocr_is_skipped() -> None:
    """没有 OCR 文本的图片进库只是 `embedded://...` 噪音。"""
    doc = make_doc([
        block(0, long_text(200)),
        DocumentBlock(
            block_id=new_block_id(DOC_ID, 1),
            block_type=BlockType.IMAGE,
            image_path="embedded://pdf/page1/img.png",
            heading_path=[],
            source_location=SourceLocation(page_start=1, page_end=1),
        ),
    ])
    assert [c.content_type for c in chunk_document(doc)] == ["paragraph"]


def test_image_ocr_text_becomes_its_own_chunk() -> None:
    doc = make_doc([
        block(
            0, "", page=2, block_type=BlockType.IMAGE,
            heading_path=[TITLE, "一、奖励标准"],
            image_path="embedded://test/award.png",
            extra={"ocr_text": "国家奖学金 8000 元/人", "ocr_source": "tesseract"},
        ),
    ])
    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].content_type == "image_ocr"
    assert "国家奖学金 8000 元/人" in chunks[0].text
    assert chunks[0].source_location.page_start == 2


def test_image_ocr_can_be_disabled() -> None:
    doc = make_doc([
        block(
            0, "", block_type=BlockType.IMAGE,
            image_path="embedded://test/a.png", extra={"ocr_text": "图里的字"},
        ),
    ])
    assert chunk_document(doc, ChunkingOptions(include_image_ocr=False)) == []


def test_boilerplate_blocks_are_dropped_by_default() -> None:
    doc = make_doc([
        block(0, "XX大学教务处文件", extra={"is_boilerplate": True}),
        block(1, long_text(200)),
    ])
    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert "教务处文件" not in chunks[0].text


def test_boilerplate_can_be_kept() -> None:
    doc = make_doc([
        block(0, "XX大学教务处文件", extra={"is_boilerplate": True}),
        block(1, long_text(200)),
    ])
    chunks = chunk_document(doc, ChunkingOptions(drop_boilerplate=False))
    assert "教务处文件" in chunks[0].text


def test_long_paragraph_is_split_at_sentence_end() -> None:
    sentence = "申请材料须于2026年9月30日前提交至学院办公室。"
    doc = make_doc([block(0, sentence * 30)])
    chunks = chunk_document(doc, ChunkingOptions(max_chars=200, overlap=20))

    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.text.split("\n", 1)[1]
        # 每块都在句号处收尾，不会把一句话劈成两半
        assert body.endswith("。") or len(body) <= 200


def test_empty_document_produces_no_chunks() -> None:
    assert chunk_document(make_doc([])) == []


# ---------------------------------------------------------------------------
# Chroma 适配
# ---------------------------------------------------------------------------


def test_chroma_metadata_contains_only_scalars() -> None:
    doc = make_doc([
        heading(0, "一、奖励标准", 2, [TITLE, "一、奖励标准"]),
        table_block(
            1,
            [["奖项", "名额"], ["国家奖学金", "10"]],
            heading_path=[TITLE, "一、奖励标准"],
        ),
    ])
    metadata = chunk_document(doc)[0].to_chroma_metadata()

    assert all(isinstance(v, (str, int, float, bool)) for v in metadata.values())
    assert metadata["document_id"] == DOC_ID
    assert metadata["heading_path"] == f"{TITLE} > 一、奖励标准"
    assert metadata["content_type"] == "table"
    assert metadata["page_start"] == 1
    assert metadata["table_rows"] == 2


def test_chroma_metadata_flattens_lists_and_drops_none() -> None:
    doc = make_doc([block(0, long_text(200))])
    chunk = chunk_document(doc)[0]
    chunk.metadata["tags"] = ["奖学金", "评审"]
    chunk.metadata["empty"] = None

    metadata = chunk.to_chroma_metadata()

    assert metadata["tags"] == "奖学金、评审"
    assert "empty" not in metadata


def test_chunk_stats_reports_distribution() -> None:
    doc = make_doc([
        block(0, long_text(300, "甲")),
        table_block(1, [["a", "b"], ["1", "2"]]),
    ])
    stats = chunk_stats(chunk_document(doc))

    assert stats["chunk数"] == 2
    assert stats["按类型"] == {"paragraph": 1, "table": 1}
    assert stats["最短"] <= stats["最长"]


def test_chunk_document_returns_chunks_with_unique_ids_for_mixed_doc() -> None:
    """混合文档（标题 / 正文 / 表格 / 图）整体跑一遍，ID 不撞。"""
    doc = make_doc([
        heading(0, "一、奖励标准", 2, [TITLE, "一、奖励标准"]),
        block(1, long_text(300, "奖励"), heading_path=[TITLE, "一、奖励标准"]),
        table_block(2, [["奖项", "名额"], ["国家奖学金", "10"]], heading_path=[TITLE, "一、奖励标准"]),
        heading(3, "二、申请条件", 2, [TITLE, "二、申请条件"]),
        block(4, long_text(300, "条件"), heading_path=[TITLE, "二、申请条件"]),
        block(
            5, "", block_type=BlockType.IMAGE,
            heading_path=[TITLE, "二、申请条件"],
            image_path="embedded://test/flow.png", extra={"ocr_text": "申请流程图"},
        ),
    ])
    chunks = chunk_document(doc)

    assert [c.content_type for c in chunks] == [
        "paragraph", "table", "paragraph", "image_ocr",
    ]
    assert len({c.chunk_id for c in chunks}) == 4
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]


@pytest.mark.parametrize("max_chars", [120, 400, 700])
def test_chunks_never_empty_for_any_max_chars(max_chars: int) -> None:
    doc = make_doc([block(0, long_text(1500))])
    chunks = chunk_document(doc, ChunkingOptions(max_chars=max_chars))

    assert chunks
    for chunk in chunks:
        body = chunk.text.split("\n", 1)[1]
        assert body.strip()
