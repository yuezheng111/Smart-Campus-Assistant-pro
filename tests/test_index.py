"""Chroma 适配层的验收测试（阶段五）。

不联网、不花钱：用一个确定性的假向量函数替代 DashScope，其余全部走真实
chromadb。因此这些断言检查的是「集合行为」，不是"接口通不通"。

四条必须守住的底线：
    1. 写不进生产集合（保险丝真的会跳）
    2. 同一份文件写两次，条数不变（幂等）
    3. 文件变短时，多出来的旧片段要被清掉（不留幽灵）
    4. 一批写失败不拖垮整份文件

运行：
    D:/AgentLearnVenv/Scripts/python.exe -m pytest tests/test_index.py -v
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app import config
from app.parsing.chunker import Chunk
from app.parsing.index import ChromaChunkIndex
from app.parsing.models import SourceLocation


class FakeEmbeddings:
    """确定性的 8 维假向量：同样的文本永远得到同样的向量。"""

    model = "fake-embedding"

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:8]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FlakyEmbeddings(FakeEmbeddings):
    """第 `fail_at` 次调用开始抛错，用来验证单批失败被隔离。"""

    def __init__(self, fail_at: int = 2) -> None:
        self.calls = 0
        self.fail_at = fail_at

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls >= self.fail_at:
            raise RuntimeError("embedding 服务 429 Too Many Requests")
        return super().embed_documents(texts)


@pytest.fixture()
def index(tmp_path: Path) -> ChromaChunkIndex:
    return ChromaChunkIndex(persist_dir=tmp_path, embeddings=FakeEmbeddings())


def make_chunks(document_id: str = "abc123", count: int = 3) -> list[Chunk]:
    # file_path 必须跟着 document_id 走。真实链路里 document_id 就是
    # sha256(file_path) 的前 16 位，两个不同的 document_id 不可能共享同一个路径。
    # 夹具里写死一个路径会造出"真实世界里不存在"的状态，而写入又确实会
    # 按 file_path 清旧片段（兼容老链路没有 document_id 的 chunk），
    # 于是后写的文件把先写的清掉，看起来像隔离失效。
    file_path = f"data/raw/{document_id}.pdf"
    return [
        Chunk(
            chunk_id=f"{document_id}-c{i:04d}",
            document_id=document_id,
            text=f"【关于奖学金的通知 > 一、奖励标准】\n第 {i} 段：国家奖学金每人 8000 元。",
            content_type="paragraph",
            chunk_index=i,
            heading_path=["关于奖学金的通知", "一、奖励标准"],
            source_location=SourceLocation(page_start=i + 1, page_end=i + 1),
            metadata={"file_name": "notice.pdf", "file_type": "pdf", "file_path": file_path},
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# 保险丝
# ---------------------------------------------------------------------------


def test_refuses_to_write_production_collection() -> None:
    """构造时就该拒绝，而不是等写到一半才发现写错了库。"""
    with pytest.raises(ValueError, match="拒绝写入生产集合"):
        ChromaChunkIndex(collection_name=config.COLLECTION_NAME, embeddings=FakeEmbeddings())


# ---------------------------------------------------------------------------
# 写入 / 幂等 / 清理
# ---------------------------------------------------------------------------


def test_upsert_then_read_back(index: ChromaChunkIndex) -> None:
    result = index.upsert_document(make_chunks(count=3))

    assert result.ok and result.written == 3 and result.deleted == 0
    assert index.chunk_ids_for("abc123") == ["abc123-c0000", "abc123-c0001", "abc123-c0002"]
    assert index.stats()["chunk数"] == 3
    assert index.stats()["文件数"] == 1


def test_repeated_write_is_idempotent(index: ChromaChunkIndex) -> None:
    """同一份文件写三次，库里仍然只有一份 —— 这是「重复入库不会翻倍」的底线。"""
    chunks = make_chunks(count=3)
    for _ in range(3):
        index.upsert_document(chunks)

    assert index.stats()["chunk数"] == 3
    assert index.chunk_ids_for("abc123") == [c.chunk_id for c in chunks]


def test_shrunk_document_leaves_no_ghost_chunks(index: ChromaChunkIndex) -> None:
    """文件改短后，旧的高序号片段必须清掉。

    这是「chunk_id 稳定 + 纯 upsert」做不到的事：新文件只有 2 块，
    c0002 不会被覆盖，会永远留在库里，检索时冒出来但原文里找不到。
    """
    index.upsert_document(make_chunks(count=5))
    assert index.stats()["chunk数"] == 5

    result = index.upsert_document(make_chunks(count=2))

    assert result.deleted == 5
    assert index.chunk_ids_for("abc123") == ["abc123-c0000", "abc123-c0001"]


def test_documents_are_isolated(index: ChromaChunkIndex) -> None:
    index.upsert_document(make_chunks("doc-a", 2))
    index.upsert_document(make_chunks("doc-b", 3))

    assert index.chunk_ids_for("doc-a") == ["doc-a-c0000", "doc-a-c0001"]
    assert index.document_ids() == {"doc-a", "doc-b"}

    index.delete_document("doc-a")
    assert index.chunk_ids_for("doc-a") == []
    assert index.stats()["chunk数"] == 3


def test_delete_missing_document_is_a_noop(index: ChromaChunkIndex) -> None:
    index.upsert_document(make_chunks("doc-a", 1))

    assert index.delete_document("not-there") == 0
    assert index.stats()["chunk数"] == 1


def test_empty_chunk_list_is_rejected_not_written(index: ChromaChunkIndex) -> None:
    result = index.upsert_document([])

    assert not result.ok
    assert index.stats()["chunk数"] == 0


def test_mixed_document_ids_are_rejected(index: ChromaChunkIndex) -> None:
    """一次只写一份文件：混着写会让「按 document_id 先删后写」失去意义。"""
    chunks = make_chunks("doc-a", 1) + make_chunks("doc-b", 1)

    result = index.upsert_document(chunks)

    assert not result.ok
    assert index.stats()["chunk数"] == 0


def test_one_failed_batch_does_not_kill_the_file(tmp_path: Path) -> None:
    """第二批开始 embedding 报错：第一批照样落地，错误如实记下来。"""
    index = ChromaChunkIndex(
        persist_dir=tmp_path, embeddings=FlakyEmbeddings(fail_at=2), batch_size=2
    )

    result = index.upsert_document(make_chunks(count=6))

    assert result.written == 2
    assert result.skipped == 4
    assert not result.ok
    assert "embedding 服务 429" in result.errors[0]
    assert index.stats()["chunk数"] == 2


# ---------------------------------------------------------------------------
# 读取与检索
# ---------------------------------------------------------------------------


def test_get_chunk_returns_text_and_metadata(index: ChromaChunkIndex) -> None:
    index.upsert_document(make_chunks(count=2))

    stored = index.get_chunk("abc123-c0001")

    assert stored is not None
    assert "第 1 段" in stored["text"]
    assert stored["metadata"]["file_name"] == "notice.pdf"
    assert stored["metadata"]["heading_path"] == "关于奖学金的通知 > 一、奖励标准"
    assert stored["metadata"]["page_start"] == 2


def test_get_missing_chunk_returns_none(index: ChromaChunkIndex) -> None:
    assert index.get_chunk("abc123-c9999") is None


def test_query_returns_chunk_id_and_score(index: ChromaChunkIndex) -> None:
    index.upsert_document(make_chunks(count=3))

    hits = index.query("国家奖学金多少钱", top_k=2)

    assert len(hits) == 2
    assert all(hit["chunk_id"].startswith("abc123-") for hit in hits)
    assert all(isinstance(hit["score"], float) for hit in hits)


def test_query_can_filter_by_document(index: ChromaChunkIndex) -> None:
    index.upsert_document(make_chunks("doc-a", 2))
    index.upsert_document(make_chunks("doc-b", 2))

    hits = index.query("奖学金", top_k=5, where={"document_id": "doc-b"})

    assert hits
    assert {hit["metadata"]["document_id"] for hit in hits} == {"doc-b"}


def test_drop_collection_clears_everything(index: ChromaChunkIndex) -> None:
    index.upsert_document(make_chunks(count=2))

    index.drop_collection()

    assert index.stats()["chunk数"] == 0
