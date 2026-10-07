"""竞赛工具：结构化查询 + 个性化推荐。

这两个工具的价值不在「查库」本身，而在于把**业务规则**沉淀下来：
    - 「大二」要能翻译成竞赛表里的「本科」
    - 「我对算法感兴趣」要能翻译成「计算机类竞赛」
    - 「现在还能不能报名」要拿今天的日期去比
这些规则如果全丢给 LLM 现算，既不稳定也浪费 token。
"""

from __future__ import annotations

import datetime as dt
import json

from langchain_core.tools import tool

from app import db
from app.tools.common import grade_like, window_status

# 竞赛的俗称 → 库内正式名称片段（同样是为了解决「学生会怎么说」的问题）
KEYWORD_ALIASES: dict[str, list[str]] = {
    "acm": ["程序设计竞赛"],
    "icpc": ["程序设计竞赛"],
    "ccpc": ["程序设计竞赛"],
    "程序设计竞赛": ["程序设计竞赛"],
    "互联网+": ["中国国际大学生创新大赛", "互联网+"],
    "数模": ["数学建模"],
    "建模": ["数学建模"],
    "信安": ["信息安全"],
    "ctf": ["信息安全"],
    "蓝桥": ["蓝桥杯"],
    "天梯": ["天梯赛"],
    "大创": ["创新大赛", "挑战杯"],
}


def _expand_keyword(keyword: str) -> list[str]:
    kw = (keyword or "").strip()
    if not kw:
        return []
    lowered = kw.lower()
    for alias, values in KEYWORD_ALIASES.items():
        if alias.lower() == lowered:
            return values
    return [kw]


# 兴趣关键词 → 竞赛类别
INTEREST_TO_CATEGORY = {
    "计算机": ["算法", "编程", "程序", "代码", "软件", "开发", "web", "前端", "后端",
               "人工智能", "ai", "机器学习", "深度学习", "信息安全", "网络安全",
               "ctf", "逆向", "渗透", "攻防", "大数据", "数据", "计算机"],
    "数学": ["数学", "建模", "统计", "数值"],
    "英语": ["英语", "四六级", "口语", "演讲", "翻译", "写作"],
    "电子": ["电子", "硬件", "嵌入式", "单片机", "电路", "自动化", "物联网"],
    "创新创业": ["创业", "创新", "商业", "挑战杯", "互联网+", "项目"],
}


def _categories_from_interests(interests: str) -> list[str]:
    text = (interests or "").lower()
    if not text:
        return []
    matched = []
    for category, keywords in INTEREST_TO_CATEGORY.items():
        if any(kw.lower() in text for kw in keywords):
            matched.append(category)
    return matched


@tool
def query_competition(
    category: str = "",
    level: str = "",
    year: int = 0,
    grade: str = "",
    keyword: str = "",
    limit: int = 20,
) -> str:
    """从竞赛数据库查询学科竞赛信息（名称、报名时间、比赛时间、面向年级、主办单位）。

    适合回答「有哪些竞赛」「某年某类竞赛什么时候报名」这类问题。
    它只返回结构化字段，不含章程原文；要看竞赛通知原文请用 search_school_document。

    Args:
        category: 竞赛类别，可选：计算机、数学、英语、电子、创新创业。不确定传空字符串。
        level: 竞赛级别，可选：国家级、省级、校级。不确定传空字符串。
        year: 届次年份（比赛发生的年份），例如 2025。不确定传 0。
        grade: 面向年级，可传「大二」这样的口语，工具会自动匹配。不确定传空字符串。
        keyword: 名称关键词，例如「蓝桥杯」「数学建模」。不确定传空字符串。
        limit: 最多返回条数，默认 20。
    """
    expanded = _expand_keyword(keyword)
    conditions: list[tuple[str, str, object]] = [
        ("category", "=", category),
        ("level", "=", level),
        ("year", "=", int(year) if year else None),
    ]
    if len(expanded) == 1:
        conditions.append(("name", "like", expanded[0]))
    elif len(expanded) > 1:
        conditions.append(("name", "or_like", expanded))
    target = grade_like(grade)
    if target:
        conditions.append(("target_grade", "like", target))

    where, params = db.build_where(conditions)
    params["limit"] = max(1, min(int(limit), 50))
    rows = db.fetch_all(
        "SELECT id, name, category, level, organizer, year, signup_start, signup_end, "
        "contest_start, contest_end, target_grade, target_major, official_url, description "
        f"FROM competition{where} ORDER BY year DESC, signup_start DESC LIMIT :limit",
        params,
    )
    today = dt.date.today()
    items = [
        {
            **row,
            # 把「阶段」算好一起返回。让模型自己拿日期去推「现在还能不能报名」，
            # 它十次里有三次会推错（尤其是跨年的场次），不如在 Tool 里算准。
            **window_status(row["signup_start"], row["signup_end"], row["contest_start"], today),
        }
        for row in rows
    ]
    return json.dumps(
        {"今天日期": today.isoformat(), "数量": len(items), "竞赛列表": items},
        ensure_ascii=False,
        default=str,
    )


@tool
def recommend_competition(
    grade: str,
    major: str = "",
    interests: str = "",
    limit: int = 6,
) -> str:
    """根据学生的年级、专业和兴趣，推荐适合参加的竞赛，并给出报名窗口状态。

    当学生问「我适合参加什么竞赛」「有什么推荐」「我现在该准备什么」时使用。
    返回结果里带「阶段」字段（未开始报名 / 正在报名 / 报名已截止，待比赛 / 已结束）。

    Args:
        grade: 学生年级，例如「大二」。必填。
        major: 专业，例如「计算机科学与技术」。不确定传空字符串。
        interests: 兴趣方向，逗号分隔，例如「算法,信息安全」。不确定传空字符串。
        limit: 最多推荐几项，默认 6。
    """
    today = dt.date.today()
    categories = _categories_from_interests(interests) or _categories_from_interests(major)
    target = grade_like(grade)

    # 只看最近三年的数据，避免推荐 2021 年的老赛事
    conditions: list[tuple[str, str, object]] = [("year", ">=", today.year - 2)]
    if target:
        conditions.append(("target_grade", "like", target))
    where, params = db.build_where(conditions)

    rows = db.fetch_all(
        "SELECT id, name, category, level, organizer, year, signup_start, signup_end, "
        "contest_start, contest_end, target_grade, target_major, official_url, description "
        f"FROM competition{where}",
        params,
    )

    scored: list[dict] = []
    for row in rows:
        # 已经识别出方向了，方向不符的直接淘汰。
        # 否则「国家级 +2、专业不限 +1」会让英语演讲赛混进计算机学生的推荐里 ——
        # 加分项只应该用来**排序**，不应该用来**放行**。
        if categories and row["category"] not in categories:
            continue

        score = 0
        reasons: list[str] = []
        if categories:
            score += 3
            reasons.append(f"与你的兴趣方向「{row['category']}」匹配")
        if row["level"] == "国家级":
            score += 2
            reasons.append("国家级赛事，含金量高")
        if major and row.get("target_major") and (
            major in row["target_major"] or row["target_major"] == "不限"
        ):
            score += 1
            reasons.append("专业对口")
        if row.get("target_grade") and "不限" in row["target_grade"]:
            score += 1
        if not categories:
            score += 1

        stage = window_status(
            row["signup_start"], row["signup_end"], row["contest_start"], today
        )
        advice = ""
        if stage["阶段"] == "已结束":
            advice = (
                f"最近一届（{row['year']} 年）报名时间是 {row['signup_start']}，"
                "下一届通常同期开放，可按这个时间点提前准备"
            )
        elif stage["阶段"] == "正在报名":
            advice = f"正在报名中，还有 {stage.get('距离报名截止天数', 0)} 天截止，建议尽快报名"

        item = {
            "id": row["id"],
            "名称": row["name"],
            "类别": row["category"],
            "级别": row["level"],
            "届次": row["year"],
            "报名开始": row["signup_start"],
            "报名截止": row["signup_end"],
            "比赛时间": row["contest_start"],
            "面向年级": row["target_grade"],
            "官方网址": row["official_url"],
            "简介": row["description"],
            "推荐理由": "；".join(dict.fromkeys(reasons)) or "面向全体学生开放",
            "匹配分": score,
            "建议": advice,
            **stage,
        }
        scored.append(item)

    # 排序优先级：正在报名 → 未开始报名 → 已截止待举行 → 已结束；
    # 同阶段内先看方向匹配度，再看届次（新的在前），最后看报名截止时间。
    priority = {"正在报名": 0, "未开始报名": 1, "报名已截止，待举行": 2, "已结束": 3}
    scored.sort(
        key=lambda x: (
            priority.get(x["阶段"], 9),
            -x["匹配分"],
            -int(x["届次"] or 0),
            str(x["报名截止"]),
        )
    )

    # 同一赛事只保留最相关的一届，否则「蓝桥杯」会在列表里出现 5 次
    deduped: list[dict] = []
    seen_names: set[str] = set()
    for item in scored:
        if item["名称"] in seen_names:
            continue
        seen_names.add(item["名称"])
        deduped.append(item)

    top = deduped[: max(1, min(int(limit), 20))]
    return json.dumps(
        {
            "学生情况": {"年级": grade, "专业": major, "兴趣": interests},
            "识别到的方向": categories,
            "今天日期": today.isoformat(),
            "推荐数量": len(top),
            "推荐列表": top,
        },
        ensure_ascii=False,
        default=str,
    )
