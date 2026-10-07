"""意图识别（Router）：Agent 的「第一道决策」。

这一步是整个 Agent 最关键的地方 —— 它决定后面走 RAG、走 SQL，还是两者都走。
做法是让 LLM 输出一个**受约束的结构体**（而不是自由文本），
再把结构体里的字段直接喂给下游的检索/查询函数。

为什么要用 with_structured_output 而不是让模型自由输出 JSON 文本？
    自由文本要自己解析，模型偶尔会加解释、加 markdown 代码块，
    线上就变成"偶发解析失败"。结构化输出把约束下沉到 API 层，稳定性完全不同。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.agent.llm import get_llm

# 允许的文件类型（与数据库 document.doc_type 取值保持一致）
DocType = Literal[
    "", "奖学金", "助学金", "考试", "竞赛", "学籍", "教学", "规章制度", "活动", "后勤"
]
# 竞赛类别（与 competition.category 保持一致）
Category = Literal["", "计算机", "数学", "英语", "电子", "创新创业"]
# 考试类别（与 exam.category 保持一致）
ExamCategory = Literal["", "英语", "计算机", "职业资格", "学业", "其他"]

RouteName = Literal["document", "structured", "hybrid", "planning", "general"]


class RoutePlan(BaseModel):
    """路由结果 + 抽取出的检索参数。"""

    route: RouteName = Field(
        description=(
            "问题类型。document=只要学校文件原文/政策规定；"
            "structured=只要竞赛或考试的时间类别等结构化字段；"
            "hybrid=既需要文件原文又需要结构化字段；"
            "planning=需要结合学生个人情况给建议/规划；"
            "general=闲聊或与本系统无关"
        )
    )
    reason: str = Field(description="一句话说明为什么这样分类，便于调试")
    rewritten_query: str = Field(
        description="把口语化问题改写成适合向量检索的检索语句，保留关键限定词"
    )

    doc_type: DocType = Field(default="", description="文件类型过滤，无法确定填空字符串")
    college: str = Field(default="", description="学院名，如「计算机学院」；不确定填空字符串")
    department: str = Field(default="", description="发布部门，如「教务处」；不确定填空字符串")
    year: int = Field(default=0, description="年份，如 2025；问的是多年或不确定则填 0")

    category: Category = Field(default="", description="竞赛类别过滤")
    level: str = Field(default="", description="竞赛级别：国家级/省级/校级；不确定填空字符串")
    exam_category: ExamCategory = Field(default="", description="考试类别过滤")
    keyword: str = Field(
        default="",
        description="专有名词关键词，如「蓝桥杯」「四六级」「转专业」「国家奖学金」；没有则填空字符串",
    )

    grade: str = Field(default="", description="学生年级，如「大二」；问题里没提学生情况则填空字符串")
    major: str = Field(default="", description="学生专业；没提则填空字符串")
    interests: str = Field(default="", description="学生兴趣方向，逗号分隔；没提则填空字符串")
    use_student_profile: bool = Field(
        default=False, description="是否需要按学号查学生画像（问题里出现学号时为 true）"
    )


ROUTER_SYSTEM = """你是一个校园信息服务 Agent 的意图识别模块。
你的唯一任务：判断学生的问题应该用哪种方式回答，并抽取出检索需要用到的参数。

五种路由的定义：
- document：答案是「某份学校文件里写了什么」。例如找通知、问政策条件、问办理流程。
- structured：答案是「竞赛/考试的时间、类别、数量」这类结构化字段。不需要读文件原文。
- hybrid：既需要文件原文，又需要结构化字段。例如「找竞赛通知」同时问「报名时间」。
- planning：需要结合「这个学生自己的情况」给建议或规划。问题里出现年级/专业/兴趣/学号，
  或者问「我适合参加什么」「我该怎么准备」「帮我规划」。
- general：打招呼、闲聊、与本系统知识无关的问题。

抽取规则：
1. 「我大二/大三」这类个人情况，一律走 planning，并把 grade 填上。
2. 「什么时候报名/比赛时间/有哪些竞赛/几年分别是几月」→ structured。
3. 「找一下 xx 通知/文件/政策怎么规定」→ document。
4. 「帮我找 xx 通知，再看看什么时候报名」→ hybrid。
5. 年份按字面填。出现「去年」而今天是 {today} 时填 {last_year}；「近五年/历年」填 0。
6. keyword 只填专有名词，不要把「通知」「时间」这类通用词填进去。

参考示例：
Q: 帮我找一下2025年国家奖学金评选通知
A: route=document, doc_type=奖学金, year=2025, keyword=国家奖学金

Q: 学校什么时候发布过关于四六级报名的通知？
A: route=document, doc_type=考试, keyword=四六级

Q: 过去三年数学竞赛分别什么时候报名？
A: route=structured, category=数学, keyword=数学竞赛

Q: 近五年计算机类有哪些竞赛？
A: route=structured, category=计算机

Q: 帮我找2025年计算机学院的竞赛通知，顺便看看报名截止时间
A: route=hybrid, college=计算机学院, doc_type=竞赛, year=2025, keyword=竞赛

Q: 我大二，计算机专业，想参加计算机类竞赛，有什么推荐？
A: route=planning, grade=大二, major=计算机, category=计算机

Q: 你好呀，你能做什么？
A: route=general
"""


def build_router():
    """返回一个「问题 -> RoutePlan」的可调用对象。"""
    import datetime as dt

    today = dt.date.today()
    system = ROUTER_SYSTEM.format(today=today.isoformat(), last_year=today.year - 1)
    llm = get_llm(temperature=0)
    return llm.with_structured_output(RoutePlan), system


def classify(question: str) -> RoutePlan:
    structured, system = _cached_router()
    return structured.invoke([("system", system), ("human", question)])


_ROUTER_CACHE = None


def _cached_router():
    global _ROUTER_CACHE
    if _ROUTER_CACHE is None:
        _ROUTER_CACHE = build_router()
    return _ROUTER_CACHE


if __name__ == "__main__":
    for q in [
        "帮我找一下2025年国家奖学金评选通知",
        "2022到2025年蓝桥杯分别什么时候报名？",
        "学校什么时候发布过关于四六级报名的通知？",
        "帮我找2025年计算机学院的竞赛通知，顺便看看报名截止时间",
        "我是大二计算机专业学生，明年适合参加哪些竞赛？",
        "近五年计算机类有哪些竞赛？",
        "你好呀",
    ]:
        plan = classify(q)
        print(f"Q: {q}")
        print(f"   route={plan.route:11s} keyword={plan.keyword!r} doc_type={plan.doc_type!r} "
              f"year={plan.year} category={plan.category!r} grade={plan.grade!r}")
        print(f"   reason={plan.reason}")
        print(f"   rewrite={plan.rewritten_query}")
