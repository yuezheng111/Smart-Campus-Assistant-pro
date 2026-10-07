"""考试工具：查询各类考试的时间安排。

和数据表设计呼应：一场考试一年可能有多次（四六级上下半年各一次），
所以 exam 表里同一 name 会有多条记录 —— 查询时按日期升序返回，
LLM 才能回答出「上半年 3 月报名、下半年 9 月报名」这种完整的答案。
"""

from __future__ import annotations

import datetime as dt
import json

from langchain_core.tools import tool

from app import db
from app.tools.common import grade_like, window_status

# 学生口语 → 库里真实存在的名称片段。
# 学生会说「四六级」，但数据库里存的是「全国大学英语四级考试（CET-4）」，
# 直接 LIKE '%四六级%' 会一条都查不到 —— 这类「口径不一致」是真实项目里
# 最高频的坑之一，必须显式做一层同义词映射。
KEYWORD_ALIASES: dict[str, list[str]] = {
    "四六级": ["四级", "六级"],
    "四、六级": ["四级", "六级"],
    "46级": ["四级", "六级"],
    "cet": ["CET"],
    "考研": ["研究生招生", "考研"],
    "研究生考试": ["研究生招生"],
    "软考": ["软考"],
}


def _expand_keyword(keyword: str) -> list[str]:
    """把关键词展开成一组等价的库内片段。"""
    kw = (keyword or "").strip()
    if not kw:
        return []
    lowered = kw.lower()
    for alias, values in KEYWORD_ALIASES.items():
        if alias.lower() == lowered or alias.lower() in lowered:
            return values
    return [kw]


@tool
def query_exam(
    category: str = "",
    year: int = 0,
    grade: str = "",
    keyword: str = "",
    upcoming_only: bool = False,
    limit: int = 20,
) -> str:
    """查询考试信息（考试名称、报名时间、考试日期、报名费、报考要求）。

    适合回答「四六级什么时候报名」「有哪些考试」「软考中级什么时候考」这类问题。

    Args:
        category: 考试类别，可选：英语、计算机、职业资格、学业、其他。不确定传空字符串。
        year: 年份，例如 2025。不确定传 0。
        grade: 面向年级，可传「大二」这样的口语，工具会自动匹配。不确定传空字符串。
        keyword: 名称关键词，例如「四六级」「软考」「普通话」。不确定传空字符串。
        upcoming_only: 传 true 时只返回今天之后还会举行的考试。默认 false。
        limit: 最多返回条数，默认 20。
    """
    today = dt.date.today()
    expanded = _expand_keyword(keyword)
    conditions: list[tuple[str, str, object]] = [
        ("category", "=", category),
        ("year", "=", int(year) if year else None),
    ]
    if len(expanded) == 1:
        conditions.append(("name", "like", expanded[0]))
    elif len(expanded) > 1:
        conditions.append(("name", "or_like", expanded))
    target = grade_like(grade)
    if target:
        conditions.append(("target_grade", "like", target))
    if upcoming_only:
        conditions.append(("exam_date", ">=", today))

    where, params = db.build_where(conditions)
    params["limit"] = max(1, min(int(limit), 50))
    rows = db.fetch_all(
        "SELECT id, name, category, year, signup_start, signup_end, exam_date, fee, "
        "target_grade, remark "
        f"FROM exam{where} ORDER BY exam_date DESC LIMIT :limit",
        params,
    )

    items = []
    for row in rows:
        items.append(
            {
                "id": row["id"],
                "名称": row["name"],
                "类别": row["category"],
                "年份": row["year"],
                "报名开始": row["signup_start"],
                "报名截止": row["signup_end"],
                "考试日期": row["exam_date"],
                "报名费": row["fee"],
                "面向年级": row["target_grade"],
                "说明": row["remark"],
                **window_status(row["signup_start"], row["signup_end"], row["exam_date"], today),
            }
        )

    return json.dumps(
        {"今天日期": today.isoformat(), "数量": len(items), "考试列表": items},
        ensure_ascii=False,
        default=str,
    )
