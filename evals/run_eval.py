# -*- coding: utf-8 -*-
"""校园智助 · 评测集运行器（三层，可单独跑）

    retrieval   只测召回：不调对话模型，只花 embedding 的钱，几十秒跑完
    router      只测意图路由：每题一次短 LLM 调用
    e2e         端到端：跑完整 LangGraph，最贵也最真实

用法：
    python evals/run_eval.py --layer retrieval
    python evals/run_eval.py --layer router
    python evals/run_eval.py --layer e2e --concurrency 3
    python evals/run_eval.py --layer all --limit 5      # 冒烟

产出：
    evals/results/eval_<tag>.json    逐题原始结果（含完整回答，便于回看）
    evals/results/report.md          人读的报告
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CASES_PATH = os.path.join(ROOT, "evals", "cases.jsonl")
RESULT_DIR = os.path.join(ROOT, "evals", "results")

_ASCII_ONLY = True  # 控制台只打英文，避免 PowerShell 编码吞中文


def say(msg: str) -> None:
    """控制台进度。Windows 控制台编码不稳，只输出 ASCII。"""
    if _ASCII_ONLY and any(ord(ch) > 127 for ch in msg):
        msg = msg.encode("ascii", "replace").decode("ascii")
    print(msg, flush=True)


# ----------------------------------------------------------------------
# 文本归一化与事实匹配
# ----------------------------------------------------------------------

_SEP = re.compile(r"[\s\-_/\\年月日（）()：:，,。.、；;]")
_DIGITS = re.compile(r"^\d+$")


def norm(text: str) -> str:
    """把「2024-11-27」「2024 年 11 月 27 日」压成同一个串，便于比对。"""
    return _SEP.sub("", str(text or "")).lower()


def contains(haystack: str, needle: str) -> bool:
    """事实匹配：日期忽略格式差异，纯数字要求数字边界（避免 45 命中 450）。"""
    h, n = norm(haystack), norm(needle)
    if not n:
        return False
    if _DIGITS.match(str(needle).strip()):
        return re.search(rf"(?<!\d){re.escape(n)}(?!\d)", h) is not None
    return n in h


def check_facts(answer: str, must: list[str]) -> tuple[int, list[str], list[str]]:
    hit, miss = [], []
    for item in must or []:
        (hit if contains(answer, item) else miss).append(item)
    return len(hit), hit, miss


# ----------------------------------------------------------------------
# 三层执行器
# ----------------------------------------------------------------------


RETRIEVAL_CANDIDATE_CHUNKS = 20
RETRIEVAL_TOP_FILES = 5


def run_retrieval(case: dict) -> dict:
    """文件级召回：先取 20 个 chunk，再按出现顺序取前 5 个**不同文件**。

    为什么不直接 k=5：一份文档通常有 2 个 chunk，k=5 时结果里往往只有
    2~3 个不同文件 —— 那样算出来的「文件级 hit@5」其实是在 hit@2，
    会系统性低估召回。所以放宽候选 chunk 数，把「文件」作为计量单位。
    """
    from app.rag.retriever import search_documents

    t0 = time.perf_counter()
    result = search_documents(case["question"], k=RETRIEVAL_CANDIDATE_CHUNKS)
    cost_ms = (time.perf_counter() - t0) * 1000

    distinct: list[str] = []
    for doc in result.get("documents", []):
        title = doc.get("title", "")
        if title and title not in distinct:
            distinct.append(title)
    titles = distinct[:RETRIEVAL_TOP_FILES]

    expected = case.get("expected_titles") or []
    rank = next((i for i, t in enumerate(titles, 1) if t in expected), None)
    return {
        "top_titles": titles,
        "hit": rank is not None,
        "rank": rank,
        "candidates": len(distinct),
        "latency_ms": round(cost_ms, 1),
        "fallback": result.get("fallback"),
    }


def route_ok(case: dict, route: str | None) -> bool:
    """路由判分。部分问题两套数据源都能答（甚至互相矛盾），给多个可接受答案。"""
    accept = case.get("route_accept") or [case["expected_route"]]
    return route in accept


def run_router(case: dict) -> dict:
    from app.agent.router import classify

    t0 = time.perf_counter()
    plan = classify(case["question"])
    cost_ms = (time.perf_counter() - t0) * 1000
    return {
        "route": plan.route,
        "correct": route_ok(case, plan.route),
        "plan": plan.model_dump(),
        "latency_ms": round(cost_ms, 1),
    }


def run_e2e(case: dict) -> dict:
    from app.agent.graph import run_agent

    t0 = time.perf_counter()
    result = run_agent(case["question"])
    cost_ms = (time.perf_counter() - t0) * 1000

    answer = result.get("answer") or ""
    sources = result.get("sources") or []
    titles = [s.get("title", "") for s in sources if s.get("title")]
    tools = [r.get("工具") for r in (result.get("data_records") or [])]

    expected = case.get("expected_titles") or []
    rank = next((i for i, t in enumerate(titles, 1) if t in expected), None)

    matched, hit, miss = check_facts(answer, case.get("must_contain") or [])
    if case.get("data_conflict"):
        # 两套数据源自相矛盾时，事实无法判分：判「对」是错的，判「错」也不公平
        matched, hit, miss = 0, [], []
    expect_refusal = bool(case.get("expect_refusal"))
    refusal_ok = None
    if expect_refusal:
        refusal_ok = any(
            k in answer
            for k in ("未检索到", "没有找到", "无法", "不清楚", "暂无",
                      "没有相关", "不掌握", "建议咨询", "建议你咨询", "不在我的")
        )

    return {
        "route": result.get("route"),
        "correct_route": route_ok(case, result.get("route")),
        "answer": answer,
        "answer_chars": len(answer),
        "sources": titles,
        "retrieval_hit": rank is not None,
        "retrieval_rank": rank,
        "tools": tools,
        "tool_ok": (case["expect_tool"] in tools) if case.get("expect_tool") else None,
        "facts_matched": matched,
        "facts_total": len(case.get("must_contain") or []),
        "facts_hit": hit,
        "facts_miss": miss,
        "citation": bool(re.search(r"\[资料\s*\d+\]", answer)),
        "refusal_ok": refusal_ok,
        "latency_ms": round(cost_ms, 1),
    }


# ----------------------------------------------------------------------
# 并发执行 + 重试
# ----------------------------------------------------------------------


def run_layer(cases: list[dict], runner, workers: int, label: str) -> dict[str, dict]:
    results: dict[str, dict] = {}
    lock = threading.Lock()
    done = 0
    total = len(cases)

    def one(case: dict) -> tuple[str, dict]:
        last_err = None
        for attempt in range(2):
            try:
                return case["id"], runner(case)
            except Exception as exc:  # 单题失败不该拖垮整轮
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(1.5 * (attempt + 1))
        return case["id"], {"error": last_err}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, c) for c in cases]
        for fut in futures:
            cid, payload = fut.result()
            with lock:
                results[cid] = payload
                done += 1
                if done % 5 == 0 or done == total:
                    say(f"  [{label}] {done}/{total}")

    return results


# ----------------------------------------------------------------------
# 指标汇总
# ----------------------------------------------------------------------


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _payload(results: dict, cid: str, layer: str) -> dict | None:
    item = results.get(cid, {}).get(layer)
    if not item or item.get("error"):
        return None
    return item


def summarize(cases: list[dict], results: dict[str, dict], layers: list[str]) -> dict:
    """只统计「这道题真的该被这一层判分」的样本。

    一条真实教训：第一版把 structured / general 题也算进召回率的分母
    （它们本来就没有预期文件、sources 恒为空），召回率被稀释到 46%，
    看上去像 RAG 崩了，其实是分母错了。指标的定义域比数值本身更重要。
    """
    summary: dict = {"n": len(cases), "layers": {}}

    if "retrieval" in layers:
        judged = [_payload(results, c["id"], "retrieval")
                  for c in cases if c.get("expected_titles")]
        judged = [r for r in judged if r]
        hits = [r for r in judged if r["hit"]]
        summary["layers"]["retrieval"] = {
            "n": len(judged),
            "hit@5": round(len(hits) / len(judged), 4) if judged else None,
            "mrr": round(sum(1 / r["rank"] for r in judged if r.get("rank")) / len(judged), 4) if judged else None,
            "p50_ms": round(pct([r["latency_ms"] for r in judged], 0.5), 1) if judged else None,
            "p95_ms": round(pct([r["latency_ms"] for r in judged], 0.95), 1) if judged else None,
        }

    if "router" in layers:
        rows = [r for r in (_payload(results, c["id"], "router") for c in cases) if r]
        summary["layers"]["router"] = {
            "n": len(rows),
            "accuracy": round(sum(1 for r in rows if r["correct"]) / len(rows), 4) if rows else None,
            "p50_ms": round(pct([r["latency_ms"] for r in rows], 0.5), 1) if rows else None,
            "p95_ms": round(pct([r["latency_ms"] for r in rows], 0.95), 1) if rows else None,
        }

    if "e2e" in layers:
        judged = [r for r in (_payload(results, c["id"], "e2e") for c in cases) if r and r.get("answer")]
        fact_rows = [r for r in judged if r.get("facts_total")]
        tool_rows = [r for r in judged if r.get("tool_ok") is not None]
        refusal_rows = [r for r in judged if r.get("refusal_ok") is not None]
        doc_ids = {c["id"] for c in cases if c.get("expected_titles")}
        doc_rows = [r for c, r in ((c, _payload(results, c["id"], "e2e")) for c in cases)
                    if c["id"] in doc_ids and r and r.get("answer")]
        conflicts = sum(1 for c in cases if c.get("data_conflict"))

        summary["layers"]["e2e"] = {
            "n": len(judged),
            "route_accuracy": round(sum(1 for r in judged if r["correct_route"]) / len(judged), 4) if judged else None,
            "retrieval_hit": round(sum(1 for r in doc_rows if r["retrieval_hit"]) / len(doc_rows), 4) if doc_rows else None,
            "doc_n": len(doc_rows),
            "fact_pass_rate": round(sum(1 for r in fact_rows if r["facts_matched"] == r["facts_total"]) / len(fact_rows), 4) if fact_rows else None,
            "fact_coverage": round(sum(r["facts_matched"] for r in fact_rows) / max(1, sum(r["facts_total"] for r in fact_rows)), 4) if fact_rows else None,
            "fact_n": len(fact_rows),
            "citation_rate": round(sum(1 for r in doc_rows if r["citation"]) / len(doc_rows), 4) if doc_rows else None,
            "tool_ok_rate": round(sum(1 for r in tool_rows if r["tool_ok"]) / len(tool_rows), 4) if tool_rows else None,
            "refusal_ok_rate": round(sum(1 for r in refusal_rows if r["refusal_ok"]) / len(refusal_rows), 4) if refusal_rows else None,
            "p50_ms": round(pct([r["latency_ms"] for r in judged], 0.5), 0) if judged else None,
            "p95_ms": round(pct([r["latency_ms"] for r in judged], 0.95), 0) if judged else None,
            "data_conflicts": conflicts,
        }

    return summary


def confusion(cases: list[dict], results: dict[str, dict]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for c in cases:
        r = results.get(c["id"], {})
        payload = r.get("router") or r.get("e2e") or {}
        got = payload.get("route")
        if not got:
            continue
        matrix.setdefault(c["expected_route"], {})
        matrix[c["expected_route"]][got] = matrix[c["expected_route"]].get(got, 0) + 1
    return matrix


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------


def ratio(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def build_report(cases: list[dict], results: dict[str, dict], summary: dict,
                 layers: list[str], elapsed: float) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    by_id = {c["id"]: c for c in cases}
    out: list[str] = []
    p = out.append

    p("# 校园智助 · 评测报告")
    p("")
    p(f"- 生成时间：{now}")
    p(f"- 用例数：{len(cases)}")
    p(f"- 执行层：{' / '.join(layers)}")
    p(f"- 总耗时：{elapsed / 60:.1f} 分钟")
    p("")
    p("> 所有标准答案来自 `data/raw/` 原文与 MySQL `competition` / `exam` 表真实行，非人工臆造。")
    p("")

    # ---- 总览 ----
    p("## 一、总览")
    p("")
    p("| 层级 | 样本 | 核心指标 |")
    p("|---|---|---|")
    if "retrieval" in summary["layers"]:
        s = summary["layers"]["retrieval"]
        p(f"| 召回 | {s['n']} | hit@5 = **{ratio(s['hit@5'])}**，MRR = **{s['mrr']}**，p50 {s['p50_ms']}ms / p95 {s['p95_ms']}ms |")
    if "router" in summary["layers"]:
        s = summary["layers"]["router"]
        p(f"| 路由 | {s['n']} | 准确率 = **{ratio(s['accuracy'])}**，p50 {s['p50_ms']}ms / p95 {s['p95_ms']}ms |")
    if "e2e" in summary["layers"]:
        s = summary["layers"]["e2e"]
        p(f"| 端到端 | {s['n']} | 路由 {ratio(s['route_accuracy'])}，事实通过 {ratio(s['fact_pass_rate'])}，"
          f"引用 {ratio(s['citation_rate'])}，p50 {s['p50_ms']}ms / p95 {s['p95_ms']}ms |")
    p("")

    # ---- 路由混淆矩阵 ----
    matrix = confusion(cases, results)
    if matrix:
        p("## 二、路由混淆矩阵（行=应有，列=实际）")
        p("")
        cols = sorted({got for row in matrix.values() for got in row})
        p("| 应有 \\ 实际 | " + " | ".join(cols) + " |")
        p("|---|" + "|".join(["---"] * len(cols)) + "|")
        for want in sorted(matrix):
            cells = " | ".join(str(matrix[want].get(c, "")) for c in cols)
            p(f"| **{want}** | {cells} |")
        p("")

    # ---- 数据源冲突（无法判分，单独列） ----
    conflicts = [c for c in cases if c.get("data_conflict")]
    if conflicts:
        p("## 三、数据源冲突（不参与判分）")
        p("")
        p("以下问题在两套数据源里答案不一致。**这不是模型的错，是数据本身矛盾**——")
        p("系统无论走哪条线都会和另一条线对不上，必须先做单一事实源。")
        p("")
        p("| 题号 | 问题 | 冲突点 |")
        p("|---|---|---|")
        for c in conflicts:
            p(f"| {c['id']} | {c['question']} | {c.get('conflict_note', '文档与数据库数值不一致')} |")
        p("")

    # ---- 失败清单 ----
    p("## 四、失败清单")
    p("")
    fails = []
    for c in cases:
        r = results.get(c["id"], {})
        problems = []
        route_reported = False
        for layer in ("retrieval", "router", "e2e"):
            payload = r.get(layer)
            if not payload:
                continue
            if payload.get("error"):
                problems.append(f"{layer} 层异常：{payload['error']}")
                continue
            if layer == "retrieval":
                # 没有预期文件的问题（structured / general）不存在召回对错
                if c.get("expected_titles") and payload.get("hit") is False:
                    problems.append(
                        f"召回未命中（返回前 3：{'；'.join(payload.get('top_titles', [])[:3])}）"
                    )
            if layer in ("router", "e2e"):
                key = "correct" if layer == "router" else "correct_route"
                if payload.get(key) is False and not route_reported:
                    accept = c.get("route_accept") or [c["expected_route"]]
                    problems.append(
                        f"路由错：应为 {'/'.join(accept)}，实际 {payload.get('route')}"
                    )
                    route_reported = True
            if layer == "e2e":
                if c.get("data_conflict"):
                    continue
                if payload.get("facts_total") and payload.get("facts_matched", 0) < payload["facts_total"]:
                    problems.append(f"事实缺失：{'、'.join(payload.get('facts_miss') or [])}")
                if payload.get("tool_ok") is False:
                    problems.append(f"未调用预期工具 {c.get('expect_tool')}（实际：{payload.get('tools')}）")
                if payload.get("refusal_ok") is False:
                    problems.append("本该拒答，但没有表达出「不知道」")
                if c.get("expected_titles") and payload.get("sources") and not payload.get("retrieval_hit"):
                    problems.append("来源里没有预期文件")
        if problems:
            fails.append((c, problems))

    if not fails:
        p("全部通过。")
    else:
        p(f"共 {len(fails)} 题有问题：")
        p("")
        for c, problems in fails:
            p(f"### {c['id']} · {c['question']}")
            p("")
            for item in dict.fromkeys(problems):
                p(f"- {item}")
            e2e = results.get(c["id"], {}).get("e2e") or {}
            if e2e.get("answer"):
                snippet = re.sub(r"\s+", " ", e2e["answer"])[:220]
                p(f"- 实际回答（截断）：{snippet}")
            p("")

    # ---- 结论 ----
    p("## 五、怎么看这份报告")
    p("")
    p("- **召回 hit@5** 是 RAG 的天花板：召回不到，后面写得再好也没用。")
    p("- **路由准确率** 决定后面走错分支的概率；混淆矩阵里「planning 被误判成 document」这类，就是典型问题。")
    p("- **事实通过率** 比「回答像不像人话」重要得多：金额、名额、日期错一个就是事故。")
    p("- **引用率** 是幻觉的廉价探针：该带 `[资料N]` 而没带，说明模型在凭记忆答。")
    p("- **p95 延迟** 才是用户体验的真实水位，别看均值。")
    p("")
    p("复现：`python evals/run_eval.py --layer all`")
    return "\n".join(out)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def load_cases(limit: int | None, path: str | None = None) -> list[dict]:
    cases = []
    with open(path or CASES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="all",
                    choices=["retrieval", "router", "e2e", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="", help="结果目录，默认 evals/results")
    ap.add_argument("--cases", default="",
                    help="用例文件路径，默认 evals/cases.jsonl。"
                         "换用例集要同时换 tag，否则结果文件会互相覆盖。")
    args = ap.parse_args()

    global RESULT_DIR
    if args.out:
        RESULT_DIR = args.out

    cases = load_cases(args.limit or None, args.cases or None)
    layers = ["retrieval", "router", "e2e"] if args.layer == "all" else [args.layer]
    os.makedirs(RESULT_DIR, exist_ok=True)

    say(f"cases={len(cases)} layers={layers} concurrency={args.concurrency}")
    t_start = time.perf_counter()

    results: dict[str, dict] = {c["id"]: {} for c in cases}
    runners = {"retrieval": run_retrieval, "router": run_router, "e2e": run_e2e}

    for layer in layers:
        say(f"[layer] {layer}")
        # 召回走本地 Chroma，并发太高会争 SQLite；顺序跑反而更稳
        workers = 1 if layer == "retrieval" else args.concurrency
        payload = run_layer(cases, runners[layer], workers, layer)
        for cid, value in payload.items():
            results.setdefault(cid, {})[layer] = value
        if layer != "e2e":  # 中间层先落盘，长跑中断也不至于全丢
            with open(os.path.join(RESULT_DIR, f"partial_{layer}.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)

    elapsed = time.perf_counter() - t_start
    summary = summarize(cases, results, layers)

    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "layers": layers,
        "elapsed_sec": round(elapsed, 1),
        "summary": summary,
        "matrix": confusion(cases, results),
        "results": results,
    }
    json_path = os.path.join(RESULT_DIR, f"eval_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)

    report = build_report(cases, results, summary, layers, elapsed)
    report_path = os.path.join(RESULT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    say(f"saved: {json_path}")
    say(f"saved: {report_path}")
    say(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
