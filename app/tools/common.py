"""工具层共用的小函数：年级翻译、日期窗口判断。

单独抽出来是因为竞赛和考试两个工具都需要同样的逻辑，
放在各自文件里会重复两遍（也容易改一处漏一处）。
"""

from __future__ import annotations

import datetime as dt
import json

# 枚举参数的白名单：键是参数名，值是库内真实存在的取值。
# 模型在 ReAct 循环里是自由生成参数的，schema 只能约束成 string，
# 传「计算机类」而不是「计算机」就会精确匹配 0 行 ——
# 而「0 行」和「库里没有」在模型看来是同一件事，会稳定产出看似诚实的错答。
ENUM_WHITELIST: dict[str, tuple[str, ...]] = {
    "category": ("计算机", "数学", "英语", "电子", "创新创业"),
    "level": ("国家级", "省级", "校级"),
    "exam_category": ("英语", "计算机", "职业资格", "学业", "其他"),
    "doc_type": (
        "奖学金", "助学金", "考试", "竞赛", "学籍",
        "教学", "规章制度", "活动", "后勤",
    ),
}

# 这些异常属于「模型改一下参数就能过」，值得让它重试；
# 其余（连接断开、权限、文件不存在）重试多少次都一样。
_RETRYABLE = (TypeError, ValueError, KeyError, IndexError, AttributeError)


def normalize_enum(
    param: str, value: str, allowed: tuple[str, ...] | None = None
) -> tuple[str, str | None]:
    """把模型给的枚举值对齐到库内取值。

    返回 (可用的值, 被拒绝的原因)。被拒绝时第一个值为空串，表示这个条件不生效。

    三种情况：
        空值            → ("", None)            不加条件
        完全匹配        → (value, None)
        「计算机类」这类 → 包含匹配成功，纠正后使用，但记录原因
        完全对不上       → ("", 原因)            忽略条件，并告诉模型「我忽略了」

    allowed 用于同名参数在不同工具里取值不同的情况（竞赛的 category
    和考试的 category 就不是同一套枚举）。
    """
    raw = (value or "").strip()
    if not raw:
        return "", None

    allowed = allowed or ENUM_WHITELIST.get(param)
    if not allowed:
        return raw, None
    if raw in allowed:
        return raw, None

    # 「计算机类」「国家级赛事」这类，值里包含了合法枚举：纠正到合法值
    hits = [item for item in allowed if item in raw or raw in item]
    if len(hits) == 1:
        return hits[0], f"{param}={raw} 已按「{hits[0]}」处理"
    if len(hits) > 1:
        return "", f"{param}={raw} 指向多个类别（{('、'.join(hits))}），该条件已忽略"

    return "", f"库内没有 {param}={raw}，该条件已忽略；可选值：{('、'.join(allowed))}"


def tool_error_payload(exc: BaseException) -> str:
    """把工具异常整理成回喂给模型的 JSON。

    为什么区分 retryable：以前一律返回「错误」，模型分不清「我把参数写错了」
    和「数据库断了」，会拿仅剩的步数反复重试同一个坏工具，把预算烧光。
    """
    retryable = isinstance(exc, _RETRYABLE)
    return json.dumps(
        {
            "错误": f"{type(exc).__name__}: {exc}",
            "可重试": retryable,
            "提示": (
                "检查参数后重试一次；仍然失败就直接说明查不到"
                if retryable
                else "这是环境或数据层的问题，重试无效，请直接告诉用户当前取不到数据"
            ),
        },
        ensure_ascii=False,
    )


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
