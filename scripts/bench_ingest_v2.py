"""阶段六上线验收：真实全量重建 → 换进程检索 → 幂等复跑。

为什么分三步、三个进程
----------------------
Chroma 的 HNSW 索引是**进程内存里的快照**：写入进程自己能立刻搜到，
换一个进程就搜不到（同一个持久化目录也没用，除非重启进程）。
所以「灌完马上搜一下」这种自测是假的 —— 学生在网页上就是另一个进程。
必须灌完退出、重新起一个进程再搜，才算验证了学生端真的能看到。

三步各自回答一个问题：

    rebuild      接上真库、真 embedding，能不能完整跑完不炸
    verify       换进程之后，学生对真实问题搜不搜得到、来源对不对
    incremental  再跑一次会不会重复入库、会不会白花钱重算向量

运行：
    D:/AgentLearnVenv/Scripts/python.exe scripts/bench_ingest_v2.py --step rebuild
    D:/AgentLearnVenv/Scripts/python.exe scripts/bench_ingest_v2.py --step verify
    D:/AgentLearnVenv/Scripts/python.exe scripts/bench_ingest_v2.py --step incremental
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# 直接 `python scripts/xxx.py` 跑时，脚本所在目录才是 sys.path[0]，
# 项目根不在里面，`import app` 会失败。这里补上，省得每次都要设 PYTHONPATH。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app import config, db
from app.ingest import ingest_paths
from app.rag import loader

OUT_DIR = _PROJECT_ROOT / "var"
CHUNK_ID_SUFFIX = "-c"

#: 验收用的问题。都是学生真会问的，且分布在不同的文件格式上
#: （pdf / docx / md / xlsx），只测 md 的话等于没测解析链路。
QUERIES = [
    "国家奖学金一年多少钱，怎么申请",
    "计算机二级考试什么时候报名",
    "助学金申请需要哪些材料",
    "毕业论文答辩流程是什么",
    "转专业有什么条件",
]


class Report:
    """把结论同时打到屏幕和文件里。"""

    def __init__(self, step: str) -> None:
        self.lines: list[str] = []
        self.path = OUT_DIR / f"bench_ingest_v2_{step}.txt"

    def __call__(self, message: str = "") -> None:
        print(message)
        self.lines.append(message)

    def save(self) -> None:
        OUT_DIR.mkdir(exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"\n[已保存] {self.path}")


# ---------------------------------------------------------------------------
# 基础盘点（只读）
# ---------------------------------------------------------------------------


def metadata_counts() -> dict[str, int]:
    total = db.fetch_one("SELECT COUNT(*) AS n FROM document")["n"]
    superseded = db.fetch_one(
        "SELECT COUNT(*) AS n FROM document WHERE status = 'superseded'"
    )["n"]
    missing = db.fetch_one(
        "SELECT COUNT(*) AS n FROM document WHERE file_missing_at IS NOT NULL"
    )["n"]
    return {"文档": int(total), "已废止": int(superseded), "原件缺失": int(missing)}


def vector_counts() -> dict[str, int]:
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    counts: dict[str, int] = {}
    for collection in client.list_collections():
        counts[collection.name] = collection.count()
    return counts


# ---------------------------------------------------------------------------
# step: rebuild
# ---------------------------------------------------------------------------


def step_rebuild(report: Report) -> None:
    paths = loader.discover_paths(config.DATA_DIR)
    report("=" * 70)
    report("阶段六验收 · 第 1 步：全量重建（真实向量库 + 真实 embedding）")
    report("=" * 70)
    report(f"源目录：{config.DATA_DIR}")
    report(f"文件数：{len(paths)}")
    report(f"链路　：PARSING_PIPELINE={config.PARSING_PIPELINE} "
           f"workers={config.INGEST_WORKERS} retries={config.INGEST_RETRIES}")
    report("")
    report(f"[跑前] 元数据：{metadata_counts()}")
    report(f"[跑前] 向量库：{vector_counts()}")
    report("")

    started = time.time()
    result = ingest_paths(paths, full=True, verbose=True)
    elapsed = time.time() - started

    report("")
    report("-" * 70)
    report(f"耗时：{elapsed:.1f}s")
    report(f"返回：文档数={result['文档数']} chunk数={result['chunk数']} "
           f"跳过数={result['跳过数']} 部分完成={result.get('部分完成数', 0)} "
           f"失败={result.get('失败数', 0)}")
    report(f"向量库总数：{result['向量库总数']}")
    report(f"[跑后] 元数据：{metadata_counts()}")
    report(f"[跑后] 向量库：{vector_counts()}")

    if result.get("跳过明细"):
        report("\n跳过明细（这些文件没进库，必须有原因）：")
        for path, reason in result["跳过明细"]:
            report(f"  - {path} | {reason}")

    if result.get("失败明细"):
        report("\n失败明细：")
        for path, reason in result["失败明细"]:
            report(f"  ! {path} | {reason}")

    if result.get("重复内容"):
        report(f"\n重复内容：{len(result['重复内容'])} 组（未自动合并）")
        for hash_, paths_ in list(result["重复内容"].items())[:5]:
            report(f"  = {hash_[:8]}… → {len(paths_)} 份")
            for item in paths_:
                report(f"      {item}")
    report.save()


# ---------------------------------------------------------------------------
# step: verify（只读，必须在另一个进程里跑）
# ---------------------------------------------------------------------------


def _collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.COLLECTION_NAME)


def step_verify(report: Report) -> None:
    report("=" * 70)
    report("阶段六验收 · 第 2 步：换进程复核（学生端视角）")
    report("=" * 70)
    report(f"元数据：{metadata_counts()}")
    report(f"向量库：{vector_counts()}")

    collection = _collection()
    report(f"\n生产集合 {config.COLLECTION_NAME}：{collection.count()} 条")

    found = collection.get(include=["metadatas"], limit=100_000)
    metadatas = [m or {} for m in (found.get("metadatas") or [])]
    ids = list(found.get("ids") or [])

    # --- 1. ID 形状：新链路的 id 必须是 <document_id>-cNNNN，老链路是 UUID ---
    stable = [i for i in ids if CHUNK_ID_SUFFIX in i[-7:] and len(i.split("-")[0]) == 16]
    report(f"\n[1] 稳定 chunk_id：{len(stable)}/{len(ids)}")
    if len(stable) != len(ids):
        report(f"    ! 有 {len(ids) - len(stable)} 条还是老的 UUID 形态")

    # --- 2. 检索地基字段：缺了这两条，默认检索会一条都搜不到 ---
    with_status = sum(1 for m in metadatas if m.get("status") == "active")
    with_missing = sum(1 for m in metadatas if "file_missing" in m)
    with_path = sum(1 for m in metadatas if (m.get("file_path") or "").endswith(
        (".pdf", ".docx", ".md", ".txt", ".xlsx", ".pptx")
    ))
    report(f"[2] status=active：{with_status}/{len(metadatas)} | "
           f"带 file_missing：{with_missing}/{len(metadatas)} | "
           f"file_path 完整：{with_path}/{len(metadatas)}")

    # --- 3. 每个 chunk 的 file_path 都必须能在磁盘上找到默认检索引用的那份 ---
    disk = {loader.relative_path(p) for p in loader.discover_paths(config.DATA_DIR)}
    orphan_paths = {m.get("file_path") for m in metadatas} - disk
    report(f"[3] 向量库里的路径共 {len({m.get('file_path') for m in metadatas})} 个，"
           f"其中不在磁盘上的 {len(orphan_paths)} 个")
    for path in sorted(orphan_paths)[:5]:
        report(f"    ? {path}")

    # --- 4. 元数据表与向量库对不对得上 ---
    rows = db.fetch_all("SELECT file_path FROM document")
    db_paths = {str(r["file_path"]) for r in rows}
    report(f"[4] 元数据表 {len(db_paths)} 份 / 向量库 {len({m.get('file_path') for m in metadatas})} 份")
    report(f"    元数据有、向量库没有：{len(db_paths - {m.get('file_path') for m in metadatas})}")
    report(f"    向量库有、元数据没有：{len({m.get('file_path') for m in metadatas} - db_paths)}")

    # --- 5. 真实提问，走学生端同一条检索路径 ---
    from app.rag import retriever

    report("\n[5] 真实提问（走 app.rag.retriever.search_documents）：")
    for question in QUERIES:
        started = time.time()
        result = retriever.search_documents(question, k=3)
        cost = (time.time() - started) * 1000
        hits = result.get("hits") or []
        documents = result.get("documents") or []
        report(f"\n  「{question}」 {cost:.0f}ms | 命中 {len(hits)} 个片段 / "
               f"{len(documents)} 份文件 | fallback={result.get('fallback')}")
        if not hits:
            report("    ! 一条都没搜到")
            continue
        top = hits[0]
        exists = (config.BASE_DIR / str(top.get("file_path") or "")).is_file()
        report(f"    首条：{top.get('title')} | score={top.get('score')} "
               f"| 原文{'在' if exists else '不见了！'}")
        report(f"    来源：{top.get('file_path')}")
        snippet = " ".join(str(top.get("content") or "").split())[:90]
        report(f"    片段：{snippet}…")
    report.save()


# ---------------------------------------------------------------------------
# step: incremental
# ---------------------------------------------------------------------------


def step_incremental(report: Report) -> None:
    paths = loader.discover_paths(config.DATA_DIR)
    before = vector_counts().get(config.COLLECTION_NAME, 0)
    meta_before = metadata_counts()

    report("=" * 70)
    report("阶段六验收 · 第 3 步：幂等复跑（增量模式，应当整批跳过）")
    report("=" * 70)
    report(f"文件数：{len(paths)}")
    report(f"[跑前] 向量库 {before} 条 | 元数据 {meta_before}")
    report("")

    started = time.time()
    result = ingest_paths(paths, full=False, verbose=True)
    elapsed = time.time() - started

    after = vector_counts().get(config.COLLECTION_NAME, 0)
    meta_after = metadata_counts()

    report("")
    report("-" * 70)
    report(f"耗时：{elapsed:.1f}s")
    report(f"文档数={result['文档数']} 跳过数={result['跳过数']} "
           f"chunk数={result['chunk数']} 失败={result.get('失败数', 0)}")
    report(f"[跑后] 向量库 {after} 条（变化 {after - before:+d}）| 元数据 {meta_after}")

    verdict = []
    verdict.append(
        ("整批跳过且一条没重算", result["跳过数"] == len(paths) and result["文档数"] == 0)
    )
    verdict.append(("向量条数没变", after == before))
    verdict.append(("元数据行数没变", meta_after["文档"] == meta_before["文档"]))
    report("\n判据：")
    for name, ok in verdict:
        report(f"  {'PASS' if ok else 'FAIL'}  {name}")
    report.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段六上线验收")
    parser.add_argument("--step", choices=("rebuild", "verify", "incremental"), required=True)
    args = parser.parse_args()

    report = Report(args.step)
    if args.step == "rebuild":
        step_rebuild(report)
    elif args.step == "verify":
        step_verify(report)
    else:
        step_incremental(report)


if __name__ == "__main__":
    main()
