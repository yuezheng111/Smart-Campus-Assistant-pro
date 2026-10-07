# -*- coding: utf-8 -*-
"""实测检索链路延迟：embedding API + Chroma 查询，跑 6 条取中位数。只读。"""
import io, sys, os, time, statistics, json
sys.stdout = io.StringIO()

import urllib.request

lines = []
p = lines.append

api_key = os.environ.get("DASHSCOPE_API_KEY", "")
p(f"has_key={bool(api_key)}")

import chromadb
client = chromadb.PersistentClient(path=r"D:\111网安学习\Agent学习6\var\chroma")
col = client.get_collection("school_documents")

queries = [
    "转专业的申请条件和流程是什么",
    "有哪些程序设计类的竞赛可以参加",
    "期末考试时间安排在哪里查",
    "奖学金评定的标准是什么",
    "计算机等级考试怎么报名",
    "毕业设计的选题要求",
]

emb_times, search_times, total_times = [], [], []
n_results = 5

for q in queries:
    t0 = time.perf_counter()
    # --- 复用项目自身的 embedding 代码路径 ---
    sys.path.insert(0, r"D:\111网安学习\Agent学习6")
    from app.rag.vectorstore import get_embeddings
    vec = get_embeddings().embed_query(q)
    t1 = time.perf_counter()
    # --- Chroma 查询 ---
    res = col.query(query_embeddings=[vec], n_results=n_results)
    t2 = time.perf_counter()
    emb_times.append((t1 - t0) * 1000)
    search_times.append((t2 - t1) * 1000)
    total_times.append((t2 - t0) * 1000)
    p(f"q={q[:12]}... emb={emb_times[-1]:.0f}ms chroma={search_times[-1]:.1f}ms total={total_times[-1]:.0f}ms hits={len(res['ids'][0])}")

p(f"median_total_ms={statistics.median(total_times):.0f}")
p(f"median_chroma_ms={statistics.median(search_times):.1f}")

with open(r"D:\111网安学习\Agent学习6\var\latency_stats.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
