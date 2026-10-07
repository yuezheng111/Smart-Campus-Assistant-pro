"""工具层共用的小函数：年级翻译、日期窗口判断。

单独抽出来是因为竞赛和考试两个工具都需要同样的逻辑，
放在各自文件里会重复两遍（也容易改一处漏一处）。
"""

from __future__ import annotations

import datetime as dt

# 学生口语 → 数据表里的 target_grade 值
GRADE_TO_TARGET = {
    "大一": "本科", "大二": "本科", "大三": "本科", "大四": "本科", "大五": "本科",
    "本科": "本科", "本科生": "本科",
    "研一": "研究生", "研二": "研究生", "研三": "研究生", "研究生": "研究生",
    "专科": "高职", "高职": "高职",
    "应届": "本科",
}


def grade_like(grade: str) -> str:
    """把「大二」这类口语翻译成数据表里 target_grade 的关键片段。"""
    grade = (grade or "").strip()
    if not grade:
        return ""
    return GRADE_TO_TARGET.get(grade, grade)


def as_date(value) -> dt.date | None:
    """把 MySQL 返回的 DATE 或字符串统一成 date。"""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str) and value:
        try:
            return dt.datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def window_status(
    signup_start, signup_end, event_start, today: dt.date | None = None
) -> dict:
    """判断某个报名/活动当前处于什么阶段 —— 这是学生最关心的信息。"""
    today = today or dt.date.today()
    start = as_date(signup_start)
    end = as_date(signup_end)
    event = as_date(event_start)

    if start and today < start:
        return {"阶段": "未开始报名", "距离报名开始天数": (start - today).days}
    if end and today <= end:
        return {"阶段": "正在报名", "距离报名截止天数": (end - today).days}
    if event and today <= event:
        return {"阶段": "报名已截止，待举行", "距离举行天数": (event - today).days}
    return {"阶段": "已结束"}
