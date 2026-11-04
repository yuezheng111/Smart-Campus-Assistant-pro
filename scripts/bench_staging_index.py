"""阶段五验收：把真实文件走完「解析 → 清洗 → 切分 → 写入 staging 集合 → 检索」。

刻意只挑 6 份代表性的文件，不是全量灌库 —— 这一步要证明的是**链路通**，
不是吞吐。全量入库属于阶段六（批量入库与稳定性）。

两条必须自己验证的底线：
    1. 生产集合 school_documents 的条数，跑前跑后**完全一致**；
    2. 同一份文件重复写入，条数不变（幂等）。

运行：
    D:/AgentLearnVenv/Scripts/python.exe scripts/bench_staging_index.py
输出：
    var/bench_staging_index.txt
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb

from app import config
from app.parsing.chunker import chunk_document
from app.parsing.index import DEFAULT_COLLECTION, ChromaChunkIndex
from app.parsing.normalize import normalize_file

OUT = io.StringIO()


def w(*args):
    print(*args, file=OUT)


SAMPLE_FILES = [
    # 带 9 行表格的 Markdown：验证「表格不被切碎、表头不丢」
    "data/raw/31_教学_2025-09-01_本科毕业论文写作流程与规范要点.md",
    # 公文 PDF：验证标题层级、页眉页脚标记
    "data/raw/37_奖学金_2026-09-18_关于做好2026年国家励志奖学金评审工作的通知.pdf",
    # 表格密集 PDF
    "data/raw/89_教学_2026-04-10_关于做好2026届本科毕业论文（设计）答辩工作的通知.pdf",
    # 大量重复模板的 DOCX：验证去重
    "data/raw/102_规章制度_2025-09-01_XX大学学生申诉处理办法.docx",
    # 表格通知 DOCX
    "data/raw/38_助学金_2026-09-22_关于做好2026年国家助学金评审工作的通知.docx",
]

QUESTIONS = [
    "国家励志奖学金的奖励标准是多少钱？",
    "毕业论文答辩需要准备哪些材料？",
    "学生如果对处分决定不服，可以怎么申诉？",
]


def production_count() -> int:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        return client.get_collection(config.COLLECTION_NAME).count()
    except Exception as exc:  # noqa: BLE001
        w(f"  （读生产集合失败：{exc}）")
        return -1


def main() -> None:
    before = production_count()
    w(f"生产集合 {config.COLLECTION_NAME} 跑前条数：{before}")
    w("")

    index = ChromaChunkIndex()
    started = time.time()
    total_chunks = 0
    per_file: list[tuple[str, int, str]] = []

    for raw in SAMPLE_FILES:
        path = Path(raw)
        if not path.exists():
            w(f"跳过（文件不存在）：{raw}")
            continue
        t0 = time.time()
        doc, report = normalize_file(path)
        chunks = chunk_document(doc)
        result = index.upsert_document(chunks)
        elapsed = time.time() - t0
        total_chunks += result.written
        per_file.append((path.name, result.written, doc.parse_status.value))

        # 「能追踪单个文件对应的全部 Chunk」——拿库里的 id 和切分结果对一遍
        stored = index.chunk_ids_for(doc.document_id)
        expected = [chunk.chunk_id for chunk in chunks]
        match = "一致" if stored == expected else f"不一致（库里 {len(stored)} / 切出 {len(expected)}）"

        w(f"{path.name}")
        w(
            f"  状态 {doc.parse_status.value} | 块 {len(doc.blocks)} | chunk {result.written} "
            f"| 页眉页脚标记 {report.boilerplate_blocks} | 耗时 {elapsed:.1f}s"
        )
        w(f"  document_id={doc.document_id} | 写入错误 {len(result.errors)} | 追溯清单：{match}")
        for error in result.errors[:3]:
            w(f"    ! {error}")
        w("")

    elapsed = time.time() - started
    w(f"合计写入 {total_chunks} 条 chunk，用时 {elapsed:.1f}s")
    w(f"staging 集合当前：{index.stats()}")
    w("")

    # ---- 幂等：同一份文件再写一次 ----
    repeat_target = Path(SAMPLE_FILES[0])
    if repeat_target.exists():
        count_before = index.stats()["chunk数"]
        doc, _ = normalize_file(repeat_target)
        again = index.upsert_document(chunk_document(doc))
        count_after = index.stats()["chunk数"]
        w("幂等检查（同一份文件重写一次）")
        w(f"  重写前 {count_before} → 重写后 {count_after}（{repeat_target.name}）")
        w(f"  本次删除旧片段 {again.deleted} 条、新增 {again.written} 条")
        w("")

    # ---- 检索 ----
    w("检索验证（top 3）")
    for question in QUESTIONS:
        w(f"  Q: {question}")
        hits = index.query(question, top_k=3)
        for rank, hit in enumerate(hits, start=1):
            metadata = hit["metadata"]
            where = metadata.get("heading_path") or "（无章节）"
            page = metadata.get("page_start")
            w(
                f"    {rank}. [{hit['score']}] {metadata.get('file_name')} "
                f"| p{page} | {where}"
            )
            w(f"       {hit['text'].splitlines()[-1][:80]}")
        w("")

    after = production_count()
    w(f"生产集合 {config.COLLECTION_NAME} 跑后条数：{after}")
    w("结论：" + ("生产集合未被改动 ✔" if before == after else "生产集合条数变化 ✘"))
    w(f"staging 集合名：{DEFAULT_COLLECTION}")

    Path("var").mkdir(exist_ok=True)
    Path("var/bench_staging_index.txt").write_text(OUT.getvalue(), encoding="utf-8")
    print("done -> var/bench_staging_index.txt")


if __name__ == "__main__":
    main()
