"""HTTP 接口压测：分级并发 + 延迟分位数，输出 JSON 原始样本与文本报告。

用法：
    python scripts/bench_load.py --group read          # 纯本机接口
    python scripts/bench_load.py --group rag           # 向量检索（每次调 embedding）
    python scripts/bench_load.py --group agent         # /api/chat（每次调 LLM）
    python scripts/bench_load.py --group read --concurrency 1,5,10,20,50 --total 200

设计约定：
- 每个并发级别固定总请求数，靠「总请求数 / 并发」而不是「固定时长」，
  否则慢接口会被截断成很少样本，快慢两边的样本量不可比。
- 先做 warmup 并丢弃，避免把首次连接建立、连接池预热算进来。
- 报告里同时给 p50 和 p99。样本 <100 时 p99 只由最后一个样本决定，仅供参考。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
VAR = Path(__file__).resolve().parent.parent / "var"

DOC_ID = 39
STUDENT_NO = "20230101"

GROUPS: dict[str, list[tuple[str, str, str, dict | None]]] = {
    "read": [
        ("health", "GET", "/health", None),
        ("docs_list", "GET", "/api/documents?limit=30", None),
        ("doc_detail", "GET", f"/api/documents/{DOC_ID}", None),
        ("doc_raw", "GET", f"/api/documents/{DOC_ID}/raw", None),
        ("stats", "GET", "/api/stats", None),
        ("competitions", "GET", "/api/competitions?limit=50", None),
        ("exams", "GET", "/api/exams?limit=50", None),
        ("student", "GET", f"/api/students/{STUDENT_NO}", None),
        ("recommend", "POST", "/api/recommend",
         {"grade": "大二", "major": "计算机科学与技术", "interests": "算法", "limit": 6}),
    ],
    "rag": [
        ("search", "GET", "/api/search?q=奖学金评定条件&top_k=5", None),
        ("search_filtered", "GET",
         "/api/search?q=转专业申请流程&doc_type=管理办法&top_k=5", None),
    ],
    "agent": [
        ("chat_simple", "POST", "/api/chat", {"question": "学校有哪些计算机类竞赛？"}),
        ("chat_doc", "POST", "/api/chat", {"question": "转专业需要什么条件？"}),
    ],
    # 对照组：静态文件不碰 MySQL / Chroma，用来区分「服务端排队」与
    # 「DB / 向量库排队」。同一并发下静态也塌，说明瓶颈在 ASGI 层或压测客户端。
    "static": [
        ("index_html", "GET", "/", None),
        ("base_css", "GET", "/styles/base.css", None),
        ("main_js", "GET", "/scripts/main.js", None),
    ],
}


def pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def summarize(samples: list[dict], wall: float) -> dict:
    ok = [s for s in samples if s["status"] == 200]
    lat = sorted(s["ms"] for s in ok)
    err: dict[str, int] = {}
    for s in samples:
        if s["status"] != 200:
            key = str(s["status"]) if s["status"] else s["err"]
            err[key] = err.get(key, 0) + 1
    return {
        "n": len(samples),
        "ok": len(ok),
        "fail": len(samples) - len(ok),
        "err": err,
        "wall_s": round(wall, 2),
        "qps": round(len(samples) / wall, 2) if wall else 0,
        "mean": round(sum(lat) / len(lat), 1) if lat else 0,
        "p50": round(pct(lat, 0.50), 1),
        "p90": round(pct(lat, 0.90), 1),
        "p95": round(pct(lat, 0.95), 1),
        "p99": round(pct(lat, 0.99), 1),
        "max": round(lat[-1], 1) if lat else 0,
    }


async def fire(client: httpx.AsyncClient, method: str, path: str,
               body: dict | None) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.request(method, BASE + path, json=body, timeout=180.0)
        ms = (time.perf_counter() - t0) * 1000
        return {"ms": ms, "status": r.status_code, "err": "", "size": len(r.content)}
    except Exception as exc:  # noqa: BLE001
        ms = (time.perf_counter() - t0) * 1000
        return {"ms": ms, "status": 0, "err": type(exc).__name__, "size": 0}


async def run_level(concurrency: int, total: int, method: str, path: str,
                    body: dict | None, duration: float = 0.0
                    ) -> tuple[list[dict], float]:
    limits = httpx.Limits(max_connections=concurrency + 20,
                          max_keepalive_connections=concurrency + 20)
    stop = asyncio.Event()
    async with httpx.AsyncClient(limits=limits, trust_env=False) as client:
        if duration:
            async def worker() -> list[dict]:
                out: list[dict] = []
                while not stop.is_set():
                    out.append(await fire(client, method, path, body))
                return out

            async def timer() -> None:
                await asyncio.sleep(duration)
                stop.set()

            t0 = time.perf_counter()
            tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
            await timer()
            wall = time.perf_counter() - t0
            chunks = [await t for t in tasks]
        else:
            per = max(1, total // concurrency)

            async def worker() -> list[dict]:
                return [await fire(client, method, path, body) for _ in range(per)]

            t0 = time.perf_counter()
            chunks = await asyncio.gather(*(worker() for _ in range(concurrency)))
            wall = time.perf_counter() - t0
    return [s for c in chunks for s in c], wall


def proc_sample(pid: int | None) -> tuple[float, int] | None:
    if not pid:
        return None
    cmd = (f"$p=Get-Process -Id {pid} -ErrorAction SilentlyContinue;"
           "if($p){ \"$($p.CPU)|$($p.WorkingSet64)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=15)
        line = out.stdout.strip()
        if "|" in line:
            cpu, mem = line.split("|", 1)
            return float(cpu), int(mem)
    except Exception:  # noqa: BLE001
        pass
    return None


async def bench_case(name: str, method: str, path: str, body: dict | None,
                     levels: list[int], total: int, warmup: int,
                     pid: int | None, raw: list[dict],
                     duration: float = 0.0) -> None:
    print(f"\n=== {name}  {method} {path}")
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=10),
                                 trust_env=False) as client:
        for _ in range(warmup):
            await fire(client, method, path, body)

    for c in levels:
        before = proc_sample(pid)
        t0 = time.perf_counter()
        samples, wall = await run_level(c, total, method, path, body, duration)
        after = proc_sample(pid)
        real_wall = time.perf_counter() - t0
        st = summarize(samples, wall)
        cpu_note = ""
        if before and after:
            d_cpu = after[0] - before[0]
            cpu_note = (f" cpu={d_cpu / real_wall * 100:.0f}%"
                        f" rss={after[1] / 1024 / 1024:.0f}MB")
        print(f"  c={c:<3} n={st['n']:<4} qps={st['qps']:<7} "
              f"p50={st['p50']:<7} p95={st['p95']:<7} p99={st['p99']:<7} "
              f"max={st['max']:<7} fail={st['fail']}{cpu_note}")
        raw.append({"case": name, "method": method, "path": path,
                    "concurrency": c, **st, "cpu_note": cpu_note})
        if st["fail"]:
            print(f"    错误分布: {st['err']}")


async def main_async(args: argparse.Namespace) -> None:
    cases = GROUPS[args.group]
    if args.only:
        cases = [c for c in cases if c[0] in args.only.split(",")]
    levels = [int(x) for x in args.concurrency.split(",")]
    raw: list[dict] = []
    for name, method, path, body in cases:
        await bench_case(name, method, path, body, levels, args.total,
                         args.warmup, args.pid, raw, args.duration)

    out_json = VAR / f"bench_load_{args.group}.json"
    out_json.write_text(json.dumps(raw, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n原始样本：{out_json}")

    lines = [f"# 压测结果 · group={args.group}",
             f"目标 {BASE} · 每级总请求数 {args.total} · warmup {args.warmup}",
             "",
             "| 用例 | 并发 | n | QPS | p50 | p90 | p95 | p99 | max | fail |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in raw:
        lines.append(f"| {r['case']} | {r['concurrency']} | {r['n']} | {r['qps']} | "
                     f"{r['p50']} | {r['p90']} | {r['p95']} | {r['p99']} | "
                     f"{r['max']} | {r['fail']} |")
    out_txt = VAR / f"bench_load_{args.group}.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告：{out_txt}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="read", choices=sorted(GROUPS))
    ap.add_argument("--concurrency", default="1,5,10,20,50")
    ap.add_argument("--total", type=int, default=200)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="给定后改为固定时长稳态压测（秒），忽略 --total")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--pid", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(main_async(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
