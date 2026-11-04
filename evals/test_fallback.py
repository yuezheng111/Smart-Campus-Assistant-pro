"""路由空结果回退 + 证据校验的回归测试。

运行：
    python evals/test_fallback.py           # 离线断言，不调 LLM，几秒跑完
    python evals/test_fallback.py --live    # 额外跑两条真实链路（会调 LLM，约 30 秒）

设计原则：
    能用纯函数断言的地方不调模型。回退是否触发、状态怎么合成、证据算不算数，
    这三件事都是纯逻辑 —— 用 LLM 去测它们，慢、花钱、还不可复现。
    只有「路由会不会判错」这件事必须用真实链路，因为它是模型行为。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.evidence import from_doc_search, from_tool_records, payload_is_empty  # noqa: E402
from app.agent.graph import (  # noqa: E402
    _after_data,
    _combine_status,
    _record_status,
    retrieve_docs_node,
)
from app.agent.verifier import check_grounding  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def _empty_exam_payload() -> dict:
    return {"今天日期": "2026-10-30", "数量": 0, "考试列表": [],
            "说明": "按当前条件确实没有记录，可以换个关键词或去掉部分条件"}


def _hit_exam_payload() -> dict:
    return {"数量": 2, "考试列表": [{"name": "CET-4"}, {"name": "CET-6"}]}


# ======================================================================
# 场景 7：工具调用记录非空，但返回 0 条 —— verify 不得判定有事实依据
# ======================================================================
def t7_tool_trace_not_evidence() -> None:
    records = [{"工具": "query_exam", "参数": {"keyword": "体测"}, "结果": _empty_exam_payload()}]
    ev = from_tool_records(records, task_id="data")
    issues = check_grounding("未检索到相关内容。", [ev], "structured")
    check("7. 空查询 → kind=tool_trace", ev["kind"] == "tool_trace", f"kind={ev['kind']}")
    check("7. 空查询 → text 为空", ev["text"] == "", f"len={len(ev['text'])}")
    check("7. 空查询 + 拒答 → 仍判缺依据", bool(issues), f"issues={issues}")

    # 对照组：真查到记录时必须认出 structured_record
    ev2 = from_tool_records(
        [{"工具": "query_exam", "参数": {}, "结果": _hit_exam_payload()}], task_id="data")
    issues2 = check_grounding("四级 6 月 14 日。", [ev2], "structured")
    check("7. 对照：查到记录 → kind=structured_record", ev2["kind"] == "structured_record", ev2["kind"])
    check("7. 对照：查到记录 → 有依据", not issues2, f"issues={issues2}")


# ======================================================================
# 场景 4：工具报错 —— 标记 error，不误判 empty，不触发回退
# ======================================================================
def t4_tool_error() -> None:
    err = {"错误": "数据库连接失败", "类型": "OperationalError"}
    check("4. 报错 → record_status=error", _record_status(err) == "error", _record_status(err))
    check("4. 报错 → payload_is_empty=None（无法判断）", payload_is_empty(err) is None)
    check("4. 单条报错 → 整体=error", _combine_status(["error"]) == "error")
    check("4. 报错+查空 → 整体=error（不降级成 empty）",
          _combine_status(["error", "empty"]) == "error")
    decision = _after_data({"route": "structured", "data_query_status": "error",
                            "doc_fallback_attempted": False})
    check("4. error 不触发回退", decision == "compose", f"decision={decision}")


# ======================================================================
# 场景 3：structured 查空，文档也为空 —— 不得谎称知识库有内容
# ======================================================================
def t3_both_empty() -> None:
    state = {
        "question": "军训什么时候开始",
        "route": "structured",
        "plan": {"rewritten_query": "军训时间"},
        "data_query_status": "empty",
        "doc_fallback_attempted": False,
        "evidence": [],
    }
    empty_result = {"hits": [], "documents": [], "filter_used": None,
                    "fallback": False, "max_score": None, "notice": ""}
    with patch("app.agent.graph.search_documents", return_value=empty_result):
        out = retrieve_docs_node(state)
    check("3. 文档也空 → doc_query_status=empty", out["doc_query_status"] == "empty")
    check("3. 文档也空 → 回退标记已置位", out["doc_fallback_attempted"] is True)
    check("3. 文档也空 → 不产生 sources", out["sources"] == [])
    # 回退失败后，校验层必须说话
    ev = from_tool_records(
        [{"工具": "query_exam", "参数": {}, "结果": _empty_exam_payload()}], task_id="data")
    issues = check_grounding("未检索到相关内容。", [ev], "structured")
    check("3. 两条线都空 → verify 判缺依据", bool(issues), f"issues={issues}")


# ======================================================================
# 场景 5：hybrid 文档有数据、数据库为空 —— 保留文档证据，不因局部失败判整体无据
# ======================================================================
def t5_hybrid_partial() -> None:
    doc_result = {"documents": [{"title": "国家奖学金通知", "id": 42}],
                  "hits": [{"content": "名额 45 名，金额 8000 元"}],
                  "max_score": 0.66}
    ev_doc = from_doc_search(doc_result, task_id="docs")
    ev_data = from_tool_records(
        [{"工具": "query_exam", "参数": {}, "结果": _empty_exam_payload()}], task_id="data")
    issues = check_grounding("名额 45 名 [资料1]", [ev_doc, ev_data], "hybrid")
    check("5. 文档有据 + 数据查空 → 判有依据", not issues, f"issues={issues}")
    check("5. hybrid 不触发回退（避免 docs→data→docs 打转）",
          _after_data({"route": "hybrid", "data_query_status": "empty",
                       "doc_fallback_attempted": False}) == "compose")


# ======================================================================
# 场景 6：planning 数据为空 —— 保持原有流程，不被新逻辑打断
# ======================================================================
def t6_planning() -> None:
    check("6. planning 查空 → 仍走 retrieve_docs（原逻辑）",
          _after_data({"route": "planning", "data_query_status": "empty",
                       "doc_fallback_attempted": False}) == "retrieve_docs")
    check("6. planning 查空且已回退过 → 仍走 retrieve_docs",
          _after_data({"route": "planning", "data_query_status": "empty",
                       "doc_fallback_attempted": True}) == "retrieve_docs")


# ======================================================================
# 回退只允许一次（防循环）
# ======================================================================
def t_once_only() -> None:
    first = _after_data({"route": "structured", "data_query_status": "empty",
                         "doc_fallback_attempted": False})
    second = _after_data({"route": "structured", "data_query_status": "empty",
                          "doc_fallback_attempted": True})
    check("0. structured 查空首次 → 回退", first == "retrieve_docs", first)
    check("0. 已回退过 → 不再回退", second == "compose", second)
    check("0. 查到数据 → 不回退",
          _after_data({"route": "structured", "data_query_status": "success",
                       "doc_fallback_attempted": False}) == "compose")
    check("0. 未查过（None）→ 不回退",
          _after_data({"route": "structured", "data_query_status": None,
                       "doc_fallback_attempted": False}) == "compose")


# ======================================================================
# 回退时必须忽略路由抽出的过滤条件（体测那次 doc_type 被抽成「考试」）
# ======================================================================
def t_fallback_ignores_filter() -> None:
    state = {
        "question": "体测时间是什么时候",
        "route": "structured",
        "plan": {"rewritten_query": "体测 时间安排", "doc_type": "考试", "year": 2025},
        "data_query_status": "empty",
        "doc_fallback_attempted": False,
        "evidence": [],
    }
    captured: dict = {}

    def fake_search(query, **kwargs):
        captured["query"] = query
        captured["kwargs"] = kwargs
        return {"hits": [], "documents": [], "filter_used": None,
                "fallback": False, "max_score": None, "notice": ""}

    with patch("app.agent.graph.search_documents", side_effect=fake_search):
        retrieve_docs_node(state)

    check("F. 回退用原始问题检索", captured["query"] == "体测时间是什么时候", captured["query"])
    check("F. 回退不带任何业务过滤", captured["kwargs"] == {}, str(captured["kwargs"]))


# ======================================================================
# 真实链路（--live）：这是唯一必须调模型才能测的部分
# ======================================================================
def t_live_case_1_structured_hit() -> None:
    from app.agent.graph import run_agent

    res = run_agent("2025年蓝桥杯什么时候报名？")
    nodes = [t.get("节点") for t in res.get("trace") or []]
    check("1(live). structured 查到数据 → 不出现回退节点",
          "retrieve_docs" not in nodes, f"nodes={nodes}")
    check("1(live). 回答非空", bool((res.get("answer") or "").strip()))


def t_live_case_2_fallback() -> None:
    from app.agent.graph import run_agent

    res = run_agent("体测时间是什么时候")
    nodes = [t.get("节点") for t in res.get("trace") or []]
    titles = [s.get("title") or "" for s in res.get("sources") or []]
    answer = res.get("answer") or ""
    check("2(live). 出现 retrieve_docs（已回退）", "retrieve_docs" in nodes, f"nodes={nodes}")
    check("2(live). 来源含体测通知",
          any("体质健康测试" in t for t in titles), f"titles={titles}")
    check("2(live). 回答含真实日期 10 月 13 日", "10 月 13 日" in answer or "10月13日" in answer)
    check("2(live). 不再只答「未检索到」", len(answer) > 120, f"len={len(answer)}")


def main() -> int:
    live = "--live" in sys.argv

    print("=" * 68)
    print("路由空结果回退 · 回归测试（离线）")
    print("=" * 68)
    t_once_only()
    t7_tool_trace_not_evidence()
    t4_tool_error()
    t3_both_empty()
    t5_hybrid_partial()
    t6_planning()
    t_fallback_ignores_filter()

    if live:
        print("-" * 68)
        print("真实链路（会调用 LLM）")
        print("-" * 68)
        try:
            t_live_case_1_structured_hit()
        except Exception as exc:  # noqa: BLE001
            check("1(live). 执行异常", False, repr(exc))
        try:
            t_live_case_2_fallback()
        except Exception as exc:  # noqa: BLE001
            check("2(live). 执行异常", False, repr(exc))

    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))

    failed = [r for r in RESULTS if not r[1]]
    print("-" * 68)
    print(f"共 {len(RESULTS)} 项，通过 {len(RESULTS) - len(failed)}，失败 {len(failed)}")
    print(f"模式：{'离线 + 真实链路' if live else '仅离线'}")

    out = ROOT / "var" / f"test_fallback_{'live' if live else 'offline'}.json"
    out.write_text(json.dumps(
        [{"name": n, "pass": ok, "detail": d} for n, ok, d in RESULTS],
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"明细：{out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
