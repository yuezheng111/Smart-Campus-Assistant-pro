"""扩充评测用例：结构化、混合、规划、闲聊四类。

为什么单独一个脚本
------------------
document 类用例是从文档规格里自动派生的（`gen_corpus_multiformat.py`），
断言直接来自正文。但另外四类的依据不一样：

- structured: 断言来自 MySQL `exam` / `competition` 表的真实行，必须现场查库；
- hybrid: 「找文件 + 追问细节」，把 document 用例改写成混合问法；
- planning / general: 没有唯一答案，只能断言路由与是否拒答。

混在一个脚本里会让「断言从哪来」这件事变模糊，而评测最怕的就是
说不清断言出处的指标。

用法
----
    python scripts/gen_more_cases.py            # 生成 evals/cases_extra.jsonl
    python scripts/gen_more_cases.py --merge    # 同时合并出 evals/cases_full.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import fetch_all  # noqa: E402

DOC_CASES = ROOT / "evals" / "cases_bulk.jsonl"
BASE_CASES = ROOT / "evals" / "cases.jsonl"
OUT_CASES = ROOT / "evals" / "cases_extra.jsonl"
FULL_CASES = ROOT / "evals" / "cases_full.jsonl"


# ----------------------------------------------------------------------
# structured：断言来自数据库真实行
# ----------------------------------------------------------------------


# 赛事与考试的简称映射。
# 用 split 自动截断会切出「课外学术科技作品竞赛」这种既丢了品牌名、
# 又不像人会说的问法，模型和工具都难对上号。宁可手写一张表。
COMP_SHORT = {
    "全国大学生电子设计竞赛": "电子设计竞赛",
    "全国大学生英语竞赛（NECCS）": "大学生英语竞赛",
    "外研社·国才杯全国大学生英语演讲大赛": "英语演讲大赛",
    "蓝桥杯全国软件和信息技术专业人才大赛": "蓝桥杯",
    "「挑战杯」全国大学生课外学术科技作品竞赛": "挑战杯",
    "中国国际大学生创新大赛（原「互联网+」）": "互联网+大学生创新创业大赛",
    "中国大学生程序设计竞赛（CCPC）": "CCPC程序设计竞赛",
    "中国大学生计算机设计大赛": "计算机设计大赛",
    "中国高校计算机大赛·团体程序设计天梯赛": "团体程序设计天梯赛",
    "全国大学生信息安全竞赛": "信息安全竞赛",
    "全国大学生数学建模竞赛": "数学建模竞赛",
    "全国大学生数学竞赛": "数学竞赛",
}

EXAM_SHORT = {
    "全国大学英语六级考试（CET-6）": "大学英语六级",
    "全国大学英语四级考试（CET-4）": "大学英语四级",
    "全国计算机等级考试（NCRE）二级": "计算机二级",
    "普通话水平测试": "普通话水平测试",
    "计算机技术与软件专业技术资格考试（软考·中级）": "软考中级",
    "全国大学生英语竞赛（初赛）": "大学生英语竞赛初赛",
    "全国硕士研究生招生考试（初试）": "硕士研究生招生考试初试",
}


def short_name(name: str, table: dict[str, str]) -> str:
    return table.get(name) or name[:14]


def gen_structured() -> list[dict]:
    """每条用例绑定一行真实记录，断言就是那一行的日期。

    只挑 2024—2026 的记录：更早的年份在库里有 6 个年份的同类记录，
    问「2021 年蓝桥杯什么时候报名」时模型答错年份也能碰巧命中别的日期，
    断言就失去判别力。
    """
    cases: list[dict] = []

    rows = fetch_all(
        "SELECT name, year, signup_start, contest_start FROM `competition` "
        "WHERE year BETWEEN 2024 AND 2026 ORDER BY name, year"
    )
    for r in rows:
        cases.append({
            "id": f"S{100 + len(cases) + 1}",
            "category": "structured",
            "question": f"{r['year']}年{short_name(r['name'], COMP_SHORT)}什么时候开始报名？",
            "expected_route": "structured",
            "expect_tool": "query_competition",
            "must_contain": [str(r["signup_start"])[:10]],
        })

    rows = fetch_all(
        "SELECT name, year, signup_start, exam_date FROM `exam` "
        "WHERE year BETWEEN 2024 AND 2026 ORDER BY name, year"
    )
    for r in rows:
        cases.append({
            "id": f"S{100 + len(cases) + 1}",
            "category": "structured",
            "question": f"{r['year']}年{short_name(r['name'], EXAM_SHORT)}什么时候考试？",
            "expected_route": "structured",
            "expect_tool": "query_exam",
            "must_contain": [str(r["exam_date"])[:10]],
        })

    return cases


# ----------------------------------------------------------------------
# hybrid：找文件 + 追问一个正文里的数字
# ----------------------------------------------------------------------


def gen_hybrid(doc_cases: list[dict], struct_cases: list[dict]) -> list[dict]:
    """混合 = 一条腿要文件、另一条腿要数据库字段。

    第一版写成「找《某文件》，顺便说下文件里的细节」—— 那按路由定义就是
    纯 document：文件名都给了，细节也在正文里，没有第二条腿。
    258 条跑下来 16 条 hybrid 被判成 document，错的不是模型，是这个设计。
    现在后半句直接取一条 structured 用例，两条线缺一不可。
    """
    docs_comp = [c for c in doc_cases
                 if c.get("_doc_type") == "竞赛" and c["expected_titles"]][:10]
    docs_exam = [c for c in doc_cases
                 if c.get("_doc_type") == "考试" and c["expected_titles"]][:10]
    st_comp = [c for c in struct_cases if c.get("expect_tool") == "query_competition"][:10]
    st_exam = [c for c in struct_cases if c.get("expect_tool") == "query_exam"][:10]

    # 同主题配对：竞赛通知配竞赛数据、考试通知配考试数据。
    # 随机拼「找奖学金通知 + 问挑战杯报名」也能测出 hybrid，但那是生造场景，
    # 报告里不好解释，也容易被质疑用例不真实。
    pairs = list(zip(docs_comp, st_comp)) + list(zip(docs_exam, st_exam))
    out: list[dict] = []
    for i, (dc, sc) in enumerate(pairs):
        title = dc["expected_titles"][0]
        must = [sc["must_contain"][0]]
        if dc.get("must_contain"):
            must.append(dc["must_contain"][0])
        out.append({
            "id": f"H{100 + i + 1}",
            "category": "hybrid",
            "question": f"帮我找一下《{title}》，另外{sc['question']}",
            "expected_route": "hybrid",
            "expected_titles": [title],
            "must_contain": must,
        })
    return out


# ----------------------------------------------------------------------
# planning / general：只断言路由（和是否拒答）
# ----------------------------------------------------------------------

PLANNING_QUESTIONS = [
    "我是大二软件工程的学生，想参加蓝桥杯，现在应该怎么准备？",
    "我大一对人工智能感兴趣，大学四年应该怎么规划？",
    "我想保研，现在大二，绩点和竞赛该怎么平衡？",
    "我大三想找实习，应该从哪些方面准备？",
    "我成绩中等，想申请奖学金，有什么建议？",
    "我数学基础一般，想参加数学建模竞赛，要补哪些内容？",
    "我想跨专业考研到计算机，现在应该做什么准备？",
    "我大一想加入实验室，应该怎么联系老师？",
    "我想出国交换，需要提前准备什么？",
    "我大四了还没确定方向，就业和考研怎么选？",
    "我想参加创新创业比赛，但没有项目想法，怎么办？",
    "我英语基础差，怎么安排四六级的复习计划？",
    "我想做前端开发，大学期间应该学哪些东西？",
    "我打算考公务员，在校期间需要准备什么？",
    "我专业是信息管理，想转行做数据分析，怎么规划？",
    "我想参加电子设计竞赛，但我是计算机专业的，可以吗？",
    "我大二想辅修金融，值得吗？怎么安排时间？",
    "我想在大学期间发一篇论文，应该怎么开始？",
    "我性格内向，想锻炼表达能力，有什么建议？",
    "我打算毕业后创业，在校期间应该积累什么？",
]

GENERAL_CHAT = [
    ("你好", None),
    ("你能做什么？", None),
    ("帮我写一首关于春天的诗", None),
    ("讲个笑话吧", None),
    ("谢谢", None),
    ("你是谁开发的？", None),
    ("今天天气怎么样？", True),
    ("学校有多少名在校学生？", True),
    ("我们学校校长是谁？", True),
    ("今年学校的预算是多少？", True),
    ("学校有几个校区？", True),
    ("食堂一天的营业额是多少？", True),
    ("宿舍晚上几点熄灯？", True),
    ("学校有多少位教授？", True),
    ("图书馆一共有多少册藏书？", True),
    ("学校去年的就业率是多少？", True),
]


def gen_planning() -> list[dict]:
    return [{
        "id": f"P{100 + i + 1}",
        "category": "planning",
        "question": q,
        "expected_route": "planning",
    } for i, q in enumerate(PLANNING_QUESTIONS)]


def gen_general() -> list[dict]:
    out = []
    for i, (q, refuse) in enumerate(GENERAL_CHAT):
        c = {
            "id": f"G{100 + i + 1}",
            "category": "general",
            "question": q,
            "expected_route": "general",
        }
        if refuse:
            c["expect_refusal"] = True
        out.append(c)
    return out


# ----------------------------------------------------------------------
# 合并
# ----------------------------------------------------------------------


def merge(extra: list[dict]) -> int:
    """旧 50 条 + 新 document 80 条 + 其余四类。

    合并时有一件事必须做：**校验旧的拒答用例是否仍然成立**。
    这次往库里加了「食堂营业时间」通知，原来的 G06「食堂几点开门」
    本来靠「查不到」来判拒答，现在能答上来了 —— 它会稳定地判失败，
    而失败原因跟模型能力无关，是语料变了。这类用例必须替换问法，
    不能留着制造假信号。
    """
    base = [json.loads(l) for l in BASE_CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = [json.loads(l) for l in DOC_CASES.read_text(encoding="utf-8").splitlines() if l.strip()]

    dropped: list[str] = []
    for c in base:
        if c["id"] == "G06":  # 食堂几点开门 —— 现在库里有答案了
            c["question"] = "学校去年一共有多少名学生毕业？"
            c["expect_refusal"] = True
            dropped.append("G06 改问法：食堂营业时间已入库，原问法不再属于拒答场景")

    seen = set()
    merged: list[dict] = []
    for c in base + docs + extra:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        merged.append(c)

    FULL_CASES.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in merged) + "\n",
        encoding="utf-8",
    )
    for line in dropped:
        print("  [修正]", line)
    return len(merged)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    doc_cases = [json.loads(l) for l in DOC_CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    structured = gen_structured()
    hybrid = gen_hybrid(doc_cases, structured)
    planning = gen_planning()
    general = gen_general()
    extra = structured + hybrid + planning + general

    OUT_CASES.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in extra) + "\n",
        encoding="utf-8",
    )

    print(f"structured {len(structured)} 条，断言全部取自 MySQL 真实行")
    print(f"hybrid     {len(hybrid)} 条")
    print(f"planning   {len(planning)} 条")
    print(f"general    {len(general)} 条（其中拒答 {sum(1 for c in general if c.get('expect_refusal'))} 条）")
    print(f"小计 {len(extra)} 条 → {OUT_CASES}")

    if args.merge:
        total = merge(extra)
        print(f"合并完成：{total} 条 → {FULL_CASES}")
        by_cat: dict[str, int] = {}
        for line in FULL_CASES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
        print("类别分布：", dict(sorted(by_cat.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
