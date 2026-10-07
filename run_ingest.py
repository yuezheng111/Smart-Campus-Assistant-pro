"""学校文件灌库：data/raw → MySQL(document 表) + Chroma(向量库)。

这条管线对应架构图里的「第一条线」：
    学校文件 → 解析 → 切分 → Embedding → 向量库 → RAG

运行：python run_ingest.py
      python run_ingest.py --purge    （连原件已消失的留档一起清掉）

它是**全量重建**入口：重写 data/raw 里的所有文件。
适合初次灌库、以及发现数据对不上时的「从源头重放」。

一个重要的行为差异
------------------
默认情况下，从 data/raw 里删掉的文件**不会被清掉**，而是被标记成
「原件缺失」：记录留档、向量片段保留，只是退出默认检索。理由是学生
可能还要查往年文件，而片段里的正文本身就是一份可用的备份。

如果确实想把它们彻底清掉，加 `--purge`。它才是字面意义上的「从零重建」，
换向量库时用它。

日常新增文件不要用这个脚本 —— 走管理后台上传（http://127.0.0.1:8000/admin），
那一边是增量的，只处理新上传的那一份，几秒钟完成、也只花那一份 embedding 的钱。
两条路共用同一套处理逻辑（app/ingest.py），所以结果一定一致。
"""

from __future__ import annotations

import argparse
import sys

from app import config
from app.ingest import ingest_paths
from app.logging_config import get_logger
from app.rag import loader

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="学校文件全量重建")
    parser.add_argument(
        "--purge",
        action="store_true",
        help="连「原件已从 data/raw 消失」的留档一起清掉（默认保留并标记）",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("学校文件灌库（RAG 第一条线 · 全量重建）")
    print("=" * 64)
    print(f"源目录：{config.DATA_DIR}")
    print(f"模式　：{'purge（彻底清空后重来）' if args.purge else '默认（保留原件已消失的留档）'}")

    paths = loader.discover_paths(config.DATA_DIR)
    if not paths:
        print("      没有找到任何文档，请先运行 python scripts/gen_sample_docs.py")
        sys.exit(1)

    result = ingest_paths(paths, full=True, purge=args.purge, verbose=True)

    if not result["文档数"]:
        print("\n没有任何文档被处理，已中止（原有数据保持不变）。")
        sys.exit(1)

    print("\n完成。可以运行 python -m app.rag.retriever 做一次检索自检。")
    log.info("全量重建完成：%d 份文档 / %d 个 chunk", result["文档数"], result["chunk数"])


if __name__ == "__main__":
    main()
