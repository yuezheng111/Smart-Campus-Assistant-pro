"""入库流水线（阶段六）的验收测试。

不连 MySQL、不连 Chroma、不花钱：索引和元数据写入都是假的，
所以这里检查的是**流程承诺**本身：

    1. 状态是按顺序走完的，失败时能说清死在哪一步
    2. 一份文件炸了不影响其他文件
    3. 只有"可能是网络抖动"的步骤才重试
    4. 重试仍失败会把半成品删干净，不留"搜得到但不全"的状态
    5. 内容没变就不重算向量（也就是不花钱）

运行：
    D:/AgentLearnVenv/Scripts/python.exe -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.index import IndexResult
from app.parsing.pipeline import (
    BatchReport,
    FileOutcome,
    IllegalTransition,
    IngestOptions,
    IngestPipeline,
    IngestStage,
    group_by_hash,
)

CORPUS = Path(__file__).resolve().parent.parent / "test_corpus"


class FakeIndex:
    """假索引：只记调用，不碰真库。"""

    def __init__(
        self,
        *,
        fail_embed_times: int = 0,
        index_errors: list[str] | None = None,
        index_written: int | None = None,
    ) -> None:
        self.warmed = 0
        self.embed_calls = 0
        self.fail_embed_times = fail_embed_times
        self.index_errors = index_errors or []
        self.index_written = index_written
        self.stored: dict[str, list] = {}
        self.deleted: list[str] = []
        self.statuses_seen: list[str] = []
        self.replaces: list[bool] = []

    def warmup(self) -> None:
        self.warmed += 1

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        if self.embed_calls <= self.fail_embed_times:
            raise RuntimeError("429 Too Many Requests")
        return [[0.1, 0.2, 0.3] for _ in texts]

    def upsert_document(self, chunks, embeddings=None, replace: bool = True) -> IndexResult:
        document_id = chunks[0].document_id
        self.statuses_seen.append(str(chunks[0].metadata.get("status")))
        self.replaces.append(replace)
        if self.index_errors and (self.index_written or 0) == 0:
            return IndexResult(document_id=document_id, written=0, errors=list(self.index_errors))
        written = len(chunks) if self.index_written is None else self.index_written
        self.stored[document_id] = list(chunks)
        return IndexResult(
            document_id=document_id, written=written, errors=list(self.index_errors)
        )

    def delete_document(self, document_id: str) -> int:
        self.deleted.append(document_id)
        return len(self.stored.pop(document_id, []))


class BulkFakeIndex(FakeIndex):
    """会记账的假索引：用来验证「整批一次删」，而不是每份删一次。"""

    def __init__(self) -> None:
        super().__init__()
        self.bulk_calls = 0
        self.bulk_ids: list[str] = []
        self.bulk_paths: list[str] = []

    def delete_documents_bulk(self, document_ids, file_paths=()) -> int:
        self.bulk_calls += 1
        self.bulk_ids = list(document_ids)
        self.bulk_paths = list(file_paths)
        return 0


class FakeMetadata:
    """假元数据表：只记写入，不碰 MySQL。"""

    def __init__(self, *, explode: bool = False) -> None:
        self.rows: list = []
        self.explode = explode
        self.states: list[dict] = []

    def __call__(self, record, states) -> None:
        if self.explode:
            raise RuntimeError("MySQL 连接被重置")
        self.rows.append(record)
        self.states.append(dict(states or {}))


def write_md(path: Path, body: str, front_matter: str = "") -> Path:
    text = f"---\n{front_matter}\n---\n\n{body}" if front_matter else body
    path.write_text(text, encoding="utf-8")
    return path


def make_pipeline(
    index: FakeIndex,
    metadata: FakeMetadata,
    *,
    unchanged_check=None,
    read_states=None,
    retries: int = 1,
) -> IngestPipeline:
    return IngestPipeline(
        index,
        metadata,
        read_states=read_states,
        unchanged_check=unchanged_check,
        options=IngestOptions(workers=2, retries=retries, retry_backoff=0.0),
    )


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


def test_happy_path_walks_every_stage(tmp_path: Path) -> None:
    path = write_md(tmp_path / "通知.md", "# 关于奖学金的通知\n\n国家奖学金每人每年 8000 元。\n")
    index, metadata = FakeIndex(), FakeMetadata()

    report = make_pipeline(index, metadata).run([path])

    assert report.total == 1 and report.completed == 1 and report.failed == 0
    outcome = report.outcomes[0]
    assert outcome.transitions == [
        IngestStage.VALIDATING,
        IngestStage.PARSING,
        IngestStage.NORMALIZING,
        IngestStage.CHUNKING,
        IngestStage.EMBEDDING,
        IngestStage.INDEXING,
        IngestStage.PERSISTING,
        IngestStage.COMPLETED,
    ]
    assert outcome.chunks >= 1
    assert len(metadata.rows) == 1
    assert index.warmed == 1  # 预热只做一次，在主线程


def test_record_carries_front_matter_metadata(tmp_path: Path) -> None:
    path = write_md(
        tmp_path / "通知.md",
        "# 关于国家奖学金的通知\n\n正文内容。\n",
        front_matter="title: 关于国家奖学金的通知\ndoc_type: 奖学金\ncollege: 会计学院\npublish_date: 2026-09-18",
    )
    index, metadata = FakeIndex(), FakeMetadata()

    make_pipeline(index, metadata).run([path])

    record = metadata.rows[0]
    assert record.title == "关于国家奖学金的通知"
    assert record.doc_type == "奖学金"
    assert record.college == "会计学院"
    assert record.publish_date is not None and record.publish_date.year == 2026
    assert record.content_hash  # 内容哈希用于"内容变没变"的判断
    assert record.summary


def test_publish_date_falls_back_to_filename(tmp_path: Path) -> None:
    """没有 front-matter 也没有侧车时，从文件名里的日期兜底。"""
    path = write_md(tmp_path / "07_考试_2025-06-20_关于等级考试报名的通知.md", "正文内容够长一些。\n")
    index, metadata = FakeIndex(), FakeMetadata()

    make_pipeline(index, metadata).run([path])

    assert metadata.rows[0].publish_date is not None
    assert metadata.rows[0].publish_date.isoformat() == "2025-06-20"


def test_sidecar_fills_fields_that_have_defaults(tmp_path: Path) -> None:
    """档案字段有非空默认值（college=全校、doc_type=其他）时侧车也得能填上。

    这一条是回归测试：原来的判据是"字段值为空才让侧车填"，
    而 college 的默认值就是"全校"，于是上传表单里选的学院永远不生效。
    """
    path = write_md(tmp_path / "通知.txt", "关于评选工作的通知正文。\n")
    (tmp_path / "通知.txt.meta.yaml").write_text(
        "title: 关于评选工作的通知\ndoc_type: 奖学金\ncollege: 金融学院\n"
        "doc_no: 粤财大〔2026〕12 号\n",
        encoding="utf-8",
    )
    index, metadata = FakeIndex(), FakeMetadata()

    make_pipeline(index, metadata).run([path])

    record = metadata.rows[0]
    assert record.college == "金融学院"
    assert record.doc_type == "奖学金"
    assert record.doc_no == "粤财大〔2026〕12 号"


def test_sidecar_title_beats_filename_but_not_front_matter(tmp_path: Path) -> None:
    """标题优先级：front-matter > 侧车 > 文件名。

    文件名兜底出来的标题不算"文件事实"，侧车应该盖过它；
    但 front-matter 里的 title 是作者写在文件里的，侧车不能盖。
    """
    archived = write_md(
        tmp_path / "38_助学金_2026-09-22_关于评选的通知.md", "正文内容够长。\n"
    )
    (tmp_path / "38_助学金_2026-09-22_关于评选的通知.md.meta.yaml").write_text(
        "title: 关于评选国家奖学金的通知\n", encoding="utf-8"
    )
    authored = write_md(
        tmp_path / "另一份.md", "正文内容够长。\n", front_matter="title: 作者写的标题"
    )
    (tmp_path / "另一份.md.meta.yaml").write_text(
        "title: 侧车写的标题\n", encoding="utf-8"
    )

    index, metadata = FakeIndex(), FakeMetadata()
    make_pipeline(index, metadata).run([archived, authored])

    assert metadata.rows[0].title == "关于评选国家奖学金的通知"
    assert metadata.rows[1].title == "作者写的标题"


# ---------------------------------------------------------------------------
# 跳过 / 失败 / 隔离
# ---------------------------------------------------------------------------


def test_scanned_pdf_is_skipped_with_reason_not_dropped(tmp_path: Path) -> None:
    """扫描件没有文本层：跳过，但要留下**原因**。

    老链路在这里只打一行 WARNING 就 continue，文件从此无声消失 ——
    管理员明明放进去了，学生搜不到，谁也说不清为什么。
    """
    source = CORPUS / "pdf" / "ocr_test.pdf"
    if not source.exists():
        pytest.skip("样例文件不存在")

    index, metadata = FakeIndex(), FakeMetadata()
    report = make_pipeline(index, metadata).run([source])

    assert report.skipped == 1 and report.completed == 0
    outcome = report.outcomes[0]
    assert outcome.stage is IngestStage.SKIPPED
    assert "OCR" in outcome.reason
    assert not metadata.rows  # 没有内容就不要在库里留一行空记录


def test_truncated_docx_fails_at_parsing(tmp_path: Path) -> None:
    source = CORPUS / "edge_cases" / "testWORD_truncated.docx"
    if not source.exists():
        pytest.skip("样例文件不存在")

    index, metadata = FakeIndex(), FakeMetadata()
    report = make_pipeline(index, metadata).run([source])

    assert report.failed == 1
    outcome = report.outcomes[0]
    assert outcome.stage is IngestStage.FAILED
    assert outcome.reason.startswith("parsing")
    assert not metadata.rows


def test_unsupported_format_fails_at_validating(tmp_path: Path) -> None:
    path = tmp_path / "表格.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    index, metadata = FakeIndex(), FakeMetadata()
    report = make_pipeline(index, metadata).run([path])

    assert report.failed == 1
    assert report.outcomes[0].reason.startswith("validating")
    assert "不支持" in report.outcomes[0].reason


def test_empty_file_fails_fast(tmp_path: Path) -> None:
    path = tmp_path / "空文件.md"
    path.write_bytes(b"")

    index, metadata = FakeIndex(), FakeMetadata()
    report = make_pipeline(index, metadata).run([path])

    assert report.failed == 1
    assert "0 字节" in report.outcomes[0].reason


def test_one_broken_file_does_not_stop_the_batch(tmp_path: Path) -> None:
    """三份文件中间放一个坏的：另外两份必须照常入库。"""
    good1 = write_md(tmp_path / "a.md", "# 甲\n\n甲的内容。\n")
    bad = tmp_path / "b.csv"
    bad.write_text("坏文件", encoding="utf-8")
    good2 = write_md(tmp_path / "c.md", "# 丙\n\n丙的内容。\n")

    index, metadata = FakeIndex(), FakeMetadata()
    report = make_pipeline(index, metadata).run([good1, bad, good2])

    assert report.completed == 2
    assert report.failed == 1
    assert len(metadata.rows) == 2
    assert report.outcomes[1].stage is IngestStage.FAILED


# ---------------------------------------------------------------------------
# 重试与回滚
# ---------------------------------------------------------------------------


def test_embedding_failure_is_retried_then_succeeds(tmp_path: Path) -> None:
    path = write_md(tmp_path / "通知.md", "# 通知\n\n正文内容够长。\n")
    index = FakeIndex(fail_embed_times=1)
    metadata = FakeMetadata()

    report = make_pipeline(index, metadata, retries=2).run([path])

    outcome = report.outcomes[0]
    assert outcome.stage is IngestStage.COMPLETED
    assert outcome.attempts == 2
    assert len(metadata.rows) == 1


def test_embedding_failure_after_retries_rolls_back(tmp_path: Path) -> None:
    """重试用尽仍失败：状态 FAILED、记在 embedding 上、元数据一个字都不写。"""
    path = write_md(tmp_path / "通知.md", "# 通知\n\n正文内容够长。\n")
    index = FakeIndex(fail_embed_times=10)
    metadata = FakeMetadata()

    report = make_pipeline(index, metadata, retries=1).run([path])

    outcome = report.outcomes[0]
    assert outcome.stage is IngestStage.FAILED
    assert outcome.reason.startswith("embedding")
    assert outcome.attempts == 2  # 1 次正常 + 1 次重试
    assert not metadata.rows
    assert outcome.document_id in index.deleted  # 回滚执行了


def test_index_failure_rolls_back_partial_chunks(tmp_path: Path) -> None:
    path = write_md(tmp_path / "通知.md", "# 通知\n\n正文内容够长。\n")
    index = FakeIndex(index_errors=["Chroma 写入超时"])
    metadata = FakeMetadata()

    report = make_pipeline(index, metadata, retries=0).run([path])

    outcome = report.outcomes[0]
    assert outcome.stage is IngestStage.FAILED
    assert outcome.reason.startswith("indexing")
    assert not metadata.rows
    assert index.deleted


def test_metadata_failure_is_reported_as_persisting(tmp_path: Path) -> None:
    """元数据写不进去时，失败阶段要指到 persisting —— 否则排查得翻三处。"""
    path = write_md(tmp_path / "通知.md", "# 通知\n\n正文内容够长。\n")
    index, metadata = FakeIndex(), FakeMetadata(explode=True)

    report = make_pipeline(index, metadata, retries=0).run([path])

    outcome = report.outcomes[0]
    assert outcome.stage is IngestStage.FAILED
    assert outcome.reason.startswith("persisting")
    # 向量已经写进去了，失败后要回滚掉，否则会留下"有片段、没登记"的孤儿
    assert index.deleted


def test_partial_index_write_marks_partial(tmp_path: Path) -> None:
    """部分片段写失败 → PARTIAL（能搜到一部分），但仍要记元数据。"""
    path = write_md(
        tmp_path / "通知.md",
        "# 通知\n\n" + "正文内容。" * 200,
    )
    index = FakeIndex(index_errors=["第 2 批写入失败"], index_written=1)
    metadata = FakeMetadata()

    report = make_pipeline(index, metadata, retries=0).run([path])

    outcome = report.outcomes[0]
    assert outcome.stage is IngestStage.PARTIAL
    assert len(metadata.rows) == 1
    assert "写入失败" in outcome.reason


# ---------------------------------------------------------------------------
# 增量与去重
# ---------------------------------------------------------------------------


def test_unchanged_file_is_skipped_without_embedding(tmp_path: Path) -> None:
    """内容没变就不调向量接口 —— 这条直接对应"不白花钱"。"""
    path = write_md(tmp_path / "通知.md", "# 通知\n\n正文内容够长。\n")
    index, metadata = FakeIndex(), FakeMetadata()

    report = make_pipeline(index, metadata, unchanged_check=lambda record: True).run([path])

    assert report.skipped == 1
    assert report.outcomes[0].stage is IngestStage.SKIPPED
    assert index.embed_calls == 0
    assert not metadata.rows


def test_duplicate_content_is_reported_not_merged(tmp_path: Path) -> None:
    """两份文件字节完全一样：只报告，不自动合并。

    同一份通知发给两个学院是**故意**的，自动合并反而会把学生需要的那个版本藏起来。
    注意判据是**文件字节**相同（`_sha256` 读原始字节），不是"解析出来的正文"相同：
    学院/文号写在 front-matter 里，字节就不同，那种情况属于"两份不同的文件"，
    不该进重复报表。
    """
    body = "# 关于奖学金的通知\n\n国家奖学金每人每年 8000 元。\n"
    first = tmp_path / "甲_通知.md"
    second = tmp_path / "乙_通知.md"
    first.write_text(body, encoding="utf-8")
    second.write_text(body, encoding="utf-8")

    index, metadata = FakeIndex(), FakeMetadata()
    report = make_pipeline(index, metadata).run([first, second])

    assert report.completed == 2
    assert len(report.duplicates) == 1
    assert len(next(iter(report.duplicates.values()))) == 2


def test_same_text_different_metadata_is_not_a_duplicate(tmp_path: Path) -> None:
    """正文一样但学院不同：不是重复内容，两份都要能分别命中。"""
    body = "# 关于奖学金的通知\n\n国家奖学金每人每年 8000 元。\n"
    first = write_md(tmp_path / "甲_通知.md", body, front_matter="college: 会计学院")
    second = write_md(tmp_path / "乙_通知.md", body, front_matter="college: 金融学院")

    index, metadata = FakeIndex(), FakeMetadata()
    report = make_pipeline(index, metadata).run([first, second])

    assert report.completed == 2
    assert report.duplicates == {}


def test_old_chunks_are_cleared_in_one_bulk_call(tmp_path: Path) -> None:
    """一批文件的旧片段必须**整批一次**清掉，不能每份删一次。

    这条是踩出来的，不是洁癖：实测（`var/_exp_chroma_churn2.py`，805 条 / 1024 维）
    逐份「删一次 → 写一批」，chroma 的 HNSW 索引会被写坏，换进程打开直接
    `Error loading hnsw index`，整个知识库不可检索。分相（先删 115 次再统一写）
    同样会坏，所以变量是删除次数。一次删完再写就正常。
    """
    paths = [
        write_md(tmp_path / f"{name}.md", f"# {name}通知\n\n{name}的正文内容。\n")
        for name in ("甲", "乙", "丙")
    ]
    index = BulkFakeIndex()
    metadata = FakeMetadata()

    report = make_pipeline(index, metadata).run(paths)

    assert report.completed == 3
    assert index.bulk_calls == 1            # 整批一次
    assert len(index.bulk_ids) == 3         # 三份文件都在这一批里
    assert index.deleted == []              # 没有逐份删
    assert index.replaces == [False] * 3    # 写入时不再自己删（旧片段已清）


def test_bulk_delete_failure_does_not_stop_the_batch(tmp_path: Path) -> None:
    """清理旧片段失败也不能让整批停摆 —— 大不了这批写成覆盖，比整批不写强。"""
    class ExplodingBulkIndex(FakeIndex):
        def delete_documents_bulk(self, document_ids, file_paths=()) -> int:
            raise RuntimeError("chroma 超时")

    path = write_md(tmp_path / "通知.md", "# 通知\n\n正文内容。\n")
    index = ExplodingBulkIndex()

    report = make_pipeline(index, FakeMetadata()).run([path])

    assert report.completed == 1


def test_group_by_hash_ignores_unique_hashes() -> None:
    from app.parsing.pipeline import DocumentRecord

    records = [
        DocumentRecord(file_path="a.md", document_id="x", title="a", content_hash="h1"),
        DocumentRecord(file_path="b.md", document_id="y", title="b", content_hash="h1"),
        DocumentRecord(file_path="c.md", document_id="z", title="c", content_hash="h2"),
    ]
    duplicates = group_by_hash(records)

    assert duplicates == {"h1": ["a.md", "b.md"]}


# ---------------------------------------------------------------------------
# 状态侧字段
# ---------------------------------------------------------------------------


def test_status_is_backfilled_into_chunks(tmp_path: Path) -> None:
    """已废止的文件重灌后，向量副本必须还是 superseded。

    不然检索时「只搜现行」的过滤会失效，学生拿到一份已经作废的政策。
    """
    path = write_md(tmp_path / "旧通知.md", "# 旧通知\n\n这是被替代的旧规定。\n")
    index, metadata = FakeIndex(), FakeMetadata()

    report = make_pipeline(
        index, metadata, read_states=lambda file_path: {"status": "superseded"}
    ).run([path])

    assert report.completed == 1
    assert index.statuses_seen == ["superseded"]
    assert metadata.states[0]["status"] == "superseded"


# ---------------------------------------------------------------------------
# 状态机本身
# ---------------------------------------------------------------------------


def test_illegal_transition_is_rejected() -> None:
    outcome = FileOutcome(file_path="x.md")

    with pytest.raises(IllegalTransition):
        outcome.advance(IngestStage.INDEXING)


def test_same_stage_reentry_is_allowed() -> None:
    """写库重试会重跑同一阶段，状态停在原地不该报错。"""
    outcome = FileOutcome(file_path="x.md")
    outcome.advance(IngestStage.VALIDATING)

    outcome.advance(IngestStage.VALIDATING)  # 不抛

    assert outcome.transitions == [IngestStage.VALIDATING]


def test_terminal_stage_has_no_way_out() -> None:
    outcome = FileOutcome(file_path="x.md", stage=IngestStage.COMPLETED)

    with pytest.raises(IllegalTransition):
        outcome.advance(IngestStage.FAILED)


def test_empty_batch_returns_empty_report() -> None:
    report = make_pipeline(FakeIndex(), FakeMetadata()).run([])

    assert report.total == 0 and report.chunks == 0
    assert report.outcomes == []


def test_report_summary_counts_add_up(tmp_path: Path) -> None:
    good = write_md(tmp_path / "a.md", "# 甲\n\n甲的内容。\n")
    bad = tmp_path / "b.csv"
    bad.write_text("x", encoding="utf-8")

    report: BatchReport = make_pipeline(FakeIndex(), FakeMetadata()).run([good, bad])

    assert report.completed + report.partial + report.failed + report.skipped == report.total
    assert "2 份" in report.summary_line()
