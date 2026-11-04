# -*- coding: utf-8 -*-
"""检索链路基准测试（规范版）：embedding API / Chroma 查询 / 合计，分层计时。

与旧版 measure_latency.py 的区别：
1. 先 warmup（预热连接与客户端），把冷启动那一发单独记，不混进样本；
2. 样本量 n>=30，而不是 6；
3. 报 p50 / p95 / max / mean，不只看中位数 —— 长尾才是用户体验的真凶；
4. 分层计时：embedding（网络+推理）与 Chroma（本地检索）分开，便于归因；
5. 结果写 UTF-8 文件（PowerShell 下 stdout 会被吞）。

只读，不改动任何数据。
"""
import io
import json
import os
import statistics
import sys
import time

sys.stdout = io.StringIO()

sys.path.insert(0, r"D:\111网安学习\Agent学习6")

PROJECT = r"D:\111网安学习\Agent学习6"
OUT = os.path.join(PROJECT, "var", "bench_retrieval_stats.txt")
N_RESULTS = 5
REPEAT = 6          # 每条查询重复次数
WARMUP = 3          # 预热次数（不计入样本）

lines = []
p = lines.append


def pct(values, q):
    """分位数：q=0.5 中位，q=0.95 → p95。用 nearest-rank，样本少时不插值更保守。"""
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def summarize(label, values):
    return (
        f"{label:12s} n={len(values):3d}  "
        f"p50={statistics.median(values):8.1f}ms  "
        f"p95={pct(values, 0.95):8.1f}ms  "
        f"max={max(values):8.1f}ms  "
        f"mean={statistics.fmean(values):8.1f}ms"
    )


api_key = os.environ.get("DASHSCOPE_API_KEY", "")
p(f"has_key={bool(api_key)}")

import chromadb

from app.rag.vectorstore import get_embeddings

client = chromadb.PersistentClient(path=os.path.join(PROJECT, "var", "chroma"))
col = client.get_collection("school_documents")
p(f"collection_count={col.count()}")

embeddings = get_embeddings()

queries = [
    "转专业的申请条件和流程是什么",
    "有哪些程序设计类的竞赛可以参加",
    "期末考试时间安排在哪里查",
    "奖学金评定的标准是什么",
    "计算机等级考试怎么报名",
    "毕业设计的选题要求",
]

# ---- 1. 预热：把冷启动（建连、懒加载）单独测出来，不污染正式样本 ----
p("\n[预热] 冷启动首发的耗时（不计入样本）")
t_cold = time.perf_counter()
vec_cold = embeddings.embed_query(queries[0])
col.query(query_embeddings=[vec_cold], n_results=N_RESULTS)
cold_ms = (time.perf_counter() - t_cold) * 1000
p(f"  cold_first_total={cold_ms:.0f}ms  <- 含 TLS 建连/客户端初始化")

for _ in range(WARMUP):
    v = embeddings.embed_query(queries[1])
    col.query(query_embeddings=[v], n_results=N_RESULTS)

# ---- 2. 正式测量 ----
emb_ms, chroma_ms, total_ms = [], [], []

p(f"\n[正式测量] {len(queries)} 条查询 × {REPEAT} 轮 = {len(queries) * REPEAT} 个样本")
for q in queries:
    row = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        vec = embeddings.embed_query(q)
        t1 = time.perf_counter()
        res = col.query(query_embeddings=[vec], n_results=N_RESULTS)
        t2 = time.perf_counter()

        emb_ms.append((t1 - t0) * 1000)
        chroma_ms.append((t2 - t1) * 1000)
        total_ms.append((t2 - t0) * 1000)
        row.append((t2 - t0) * 1000)
    p(f"  q={q[:14]:16s} total p50={statistics.median(row):6.0f}ms  hits={len(res['ids'][0])}")

# ---- 3. 汇总 ----
p("\n[汇总]")
p("  " + summarize("embedding", emb_ms))
p("  " + summarize("chroma", chroma_ms))
p("  " + summarize("total", total_ms))
p("")
p(f"占位口径：n_results={N_RESULTS}, repeat={REPEAT}, 每样本含 1 次 embed + 1 次 query")
p("注意：embedding 含到 DashScope 的公网往返，换机器/换网络结果会变；chroma 为本地检索。")

# 原始样本（升序），供画分布图 / 复核分位数用
p("\n[原始样本·升序]")
p("embedding=" + json.dumps([round(v, 1) for v in sorted(emb_ms)]))
p("chroma=" + json.dumps([round(v, 1) for v in sorted(chroma_ms)]))
p("total=" + json.dumps([round(v, 1) for v in sorted(total_ms)]))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("done ->", OUT)
