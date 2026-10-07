"""结构化种子数据：竞赛 / 考试 / 学生画像。

这里用「模板 + 年份展开」的方式造数据，而不是手写几百行 INSERT：
真实项目里历年数据本身就是重复度很高的，用模板生成更好维护。
"""

from __future__ import annotations

import datetime as dt

# ------------------------------------------------------------------
# 竞赛模板
#   signup / contest 用 (月, 日) 表示，按年份展开
#   drift 让不同年份的日期有一点自然漂移，避免数据假得不真实
# ------------------------------------------------------------------
COMPETITION_TEMPLATES: list[dict] = [
    {
        "name": "蓝桥杯全国软件和信息技术专业人才大赛",
        "category": "计算机",
        "level": "国家级",
        "organizer": "工业和信息化部人才交流中心",
        "signup": (11, 25),
        "contest": (4, 12),
        "target_grade": "本科/高职",
        "target_major": "不限",
        "url": "https://dasai.lanqiao.cn/",
        "description": "个人赛，分 C/C++、Java、Python 等组别；校赛—省赛—国赛三级赛制，省赛获奖可加综测分。",
    },
    {
        "name": "中国大学生计算机设计大赛",
        "category": "计算机",
        "level": "国家级",
        "organizer": "教育部高等学校计算机类专业教学指导委员会",
        "signup": (3, 1),
        "contest": (5, 18),
        "target_grade": "本科",
        "target_major": "计算机类/软件工程/数字媒体",
        "url": "https://jsjds.blcu.edu.cn/",
        "description": "团队赛（2-5 人），分软件应用、人工智能应用、数媒设计等大类，先省赛后国赛。",
    },
    {
        "name": "中国大学生程序设计竞赛（CCPC）",
        "category": "计算机",
        "level": "国家级",
        "organizer": "中国大学生程序设计竞赛组委会",
        "signup": (5, 10),
        "contest": (9, 20),
        "target_grade": "本科",
        "target_major": "计算机类",
        "url": "https://ccpc.io/",
        "description": "ACM 赛制，3 人一队，5 小时内解题，按通过题数与罚时排名，含金量高。",
    },
    {
        "name": "全国大学生信息安全竞赛",
        "category": "计算机",
        "level": "国家级",
        "organizer": "教育部高等学校网络空间安全专业教学指导委员会",
        "signup": (3, 15),
        "contest": (7, 8),
        "target_grade": "本科",
        "target_major": "网络空间安全/计算机类",
        "url": "http://www.ciscn.cn/",
        "description": "分作品赛与创新实践能力赛（CTF）。CTF 方向涵盖 Web、逆向、Pwn、密码、杂项。",
    },
    {
        "name": "中国高校计算机大赛·团体程序设计天梯赛",
        "category": "计算机",
        "level": "国家级",
        "organizer": "教育部高等学校计算机类专业教学指导委员会",
        "signup": (1, 10),
        "contest": (4, 20),
        "target_grade": "本科/高职",
        "target_major": "计算机类",
        "url": "https://gplt.patest.cn/",
        "description": "10 人一队，分基础、进阶、登顶三级题目，按团队总分排名。",
    },
    {
        "name": "全国大学生数学建模竞赛",
        "category": "数学",
        "level": "国家级",
        "organizer": "中国工业与应用数学学会",
        "signup": (5, 1),
        "contest": (9, 8),
        "target_grade": "本科/专科",
        "target_major": "不限",
        "url": "https://www.mcm.edu.cn/",
        "description": "3 人一队，72 小时完成建模、求解、论文写作，是国内规模最大的基础性学科竞赛。",
    },
    {
        "name": "全国大学生数学竞赛",
        "category": "数学",
        "level": "国家级",
        "organizer": "中国数学会",
        "signup": (9, 1),
        "contest": (11, 10),
        "target_grade": "本科",
        "target_major": "不限（数学类专业分组评奖）",
        "url": "https://www.cmathc.cn/",
        "description": "分数学类与非数学类，考试形式为笔试 3 小时，内容以高等数学、数学分析为主。",
    },
    {
        "name": "全国大学生英语竞赛（NECCS）",
        "category": "英语",
        "level": "国家级",
        "organizer": "高等学校大学外语教学研究会",
        "signup": (12, 15),
        "contest": (4, 14),
        "target_grade": "本科/研究生",
        "target_major": "不限",
        "url": "http://www.chinaneccs.cn/",
        "description": "分 A/B/C/D 四类，本科非英语专业报 C 类，含听力、词汇、阅读、翻译、写作。",
    },
    {
        "name": "外研社·国才杯全国大学生英语演讲大赛",
        "category": "英语",
        "level": "国家级",
        "organizer": "外语教学与研究出版社",
        "signup": (9, 10),
        "contest": (10, 25),
        "target_grade": "本科",
        "target_major": "不限",
        "url": "https://uchallenge.unipus.cn/",
        "description": "先校级初赛，再省级复赛、全国决赛；含定题演讲、即兴演讲与问答环节。",
    },
    {
        "name": "全国大学生电子设计竞赛",
        "category": "电子",
        "level": "国家级",
        "organizer": "教育部高等教育司、工业和信息化部人事教育司",
        "signup": (4, 1),
        "contest": (8, 10),
        "target_grade": "本科/高职",
        "target_major": "电子信息类/自动化类",
        "url": "https://nuedc.xjtu.edu.cn/",
        "description": "3 人一队，4 天 3 夜完成硬件设计与调试，奇数年举办全国赛。",
    },
    {
        "name": "中国国际大学生创新大赛（原「互联网+」）",
        "category": "创新创业",
        "level": "国家级",
        "organizer": "教育部",
        "signup": (4, 15),
        "contest": (8, 20),
        "target_grade": "本科/研究生/高职",
        "target_major": "不限",
        "url": "https://cy.ncss.cn/",
        "description": "分高教主赛道、青年红色筑梦之旅赛道、产业命题赛道；需提交商业计划书与路演。",
    },
    {
        "name": "「挑战杯」全国大学生课外学术科技作品竞赛",
        "category": "创新创业",
        "level": "国家级",
        "organizer": "共青团中央、中国科协、教育部",
        "signup": (3, 20),
        "contest": (6, 15),
        "target_grade": "本科/研究生",
        "target_major": "不限",
        "url": "https://www.tiaozhanbei.net/",
        "description": "分自然科学类学术论文、哲学社会科学类社会调查报告、科技发明制作三类。",
    },
]

# ------------------------------------------------------------------
# 考试模板：一年可能有 2 次（如 CET、NCRE）
# ------------------------------------------------------------------
EXAM_TEMPLATES: list[dict] = [
    {
        "name": "全国大学英语四级考试（CET-4）",
        "category": "英语",
        "sessions": [
            {"signup": (3, 20), "exam": (6, 14)},
            {"signup": (9, 20), "exam": (12, 14)},
        ],
        "fee": "30-50 元",
        "target_grade": "本科/研究生",
        "remark": "需先通过四级才能报考六级；四级成绩 425 分及以上视为通过。",
    },
    {
        "name": "全国大学英语六级考试（CET-6）",
        "category": "英语",
        "sessions": [
            {"signup": (3, 20), "exam": (6, 14)},
            {"signup": (9, 20), "exam": (12, 14)},
        ],
        "fee": "30-50 元",
        "target_grade": "本科/研究生",
        "remark": "报考需四级成绩达到 425 分；部分院校对刷分次数有限制。",
    },
    {
        "name": "全国计算机等级考试（NCRE）二级",
        "category": "计算机",
        "sessions": [
            {"signup": (12, 25), "exam": (3, 22)},
            {"signup": (6, 25), "exam": (9, 21)},
        ],
        "fee": "80-120 元",
        "target_grade": "不限",
        "remark": "二级含 MS Office、Python、C 语言等科目；证书长期有效。",
    },
    {
        "name": "计算机技术与软件专业技术资格考试（软考·中级）",
        "category": "计算机",
        "sessions": [
            {"signup": (3, 10), "exam": (5, 24)},
            {"signup": (8, 10), "exam": (11, 8)},
        ],
        "fee": "约 140 元/科",
        "target_grade": "不限",
        "remark": "软件设计师、网络工程师等中级资格含金量较高，部分城市可职称落户。",
    },
    {
        "name": "全国大学生英语竞赛（初赛）",
        "category": "英语",
        "sessions": [{"signup": (12, 15), "exam": (4, 14)}],
        "fee": "约 30 元",
        "target_grade": "本科/研究生",
        "remark": "初赛成绩优异者可进入决赛；获奖可计入综测。",
    },
    {
        "name": "普通话水平测试",
        "category": "其他",
        "sessions": [
            {"signup": (3, 5), "exam": (4, 18)},
            {"signup": (9, 5), "exam": (10, 18)},
        ],
        "fee": "约 50 元",
        "target_grade": "不限",
        "remark": "教师资格认定需二级乙等及以上；语文教师需二级甲等。",
    },
    {
        "name": "全国硕士研究生招生考试（初试）",
        "category": "学业",
        "sessions": [{"signup": (10, 8), "exam": (12, 21)}],
        "fee": "约 150 元",
        "target_grade": "大四/应届本科",
        "remark": "网上预报名 9 月底，正式报名 10 月，初试 12 月下旬，复试次年 3-4 月。",
    },
]

# 届次年份跨度：跟着「今天」走，保证永远有「正在报名」的数据可演示。
# 写死 [2021..2025] 的后果是——过两年跑起来全是「已结束」，演示直接垮掉。
_CURRENT_YEAR = dt.date.today().year
YEARS = list(range(_CURRENT_YEAR - 5, _CURRENT_YEAR + 1))

# ------------------------------------------------------------------
# 学生画像（演示用）
# ------------------------------------------------------------------
STUDENTS: list[dict] = [
    {
        "student_no": "20230101",
        "name": "陈乐正",
        "grade": "大三",
        "major": "计算机科学与技术",
        "college": "计算机学院",
        "interests": "算法,信息安全,人工智能",
        "gpa": 3.72,
    },
    {
        "student_no": "20240217",
        "name": "林一鸣",
        "grade": "大二",
        "major": "软件工程",
        "college": "计算机学院",
        "interests": "Web开发,创新创业",
        "gpa": 3.45,
    },
    {
        "student_no": "20250933",
        "name": "苏晓棠",
        "grade": "大一",
        "major": "数学与应用数学",
        "college": "数学学院",
        "interests": "数学建模,数据分析",
        "gpa": 3.88,
    },
]


def _date(year: int, month: int, day: int, drift: int = 0) -> dt.date:
    """构造日期，drift 用于制造自然的年份漂移（±3 天）。"""
    return dt.date(year, month, day) + dt.timedelta(days=drift)


def _align(signup: tuple[int, int], event: tuple[int, int], event_year: int, drift: int) -> tuple[dt.date, dt.date]:
    """让「报名」与「正式活动」落在正确的年份上。

    约定：模板里的 year 表示**正式活动（比赛/考试）发生的年份**。
    如果报名月份晚于活动月份（如 11 月报名、次年 4 月比赛），
    报名就应该落在前一年 —— 这正是学生最容易问错的地方。
    """
    event_date = _date(event_year, *event, drift)
    signup_year = event_year - 1 if signup[0] > event[0] else event_year
    signup_date = _date(signup_year, *signup, drift)
    return signup_date, event_date


def build_competitions() -> list[dict]:
    rows: list[dict] = []
    for template in COMPETITION_TEMPLATES:
        for index, year in enumerate(YEARS):
            drift = (index * 3) % 7 - 3  # -3..3 之间循环漂移
            signup_start, contest_start = _align(
                template["signup"], template["contest"], year, drift
            )
            rows.append(
                {
                    "name": template["name"],
                    "category": template["category"],
                    "level": template["level"],
                    "organizer": template["organizer"],
                    "year": year,
                    "signup_start": signup_start,
                    "signup_end": signup_start + dt.timedelta(days=75),
                    "contest_start": contest_start,
                    "contest_end": contest_start + dt.timedelta(days=7),
                    "target_grade": template["target_grade"],
                    "target_major": template["target_major"],
                    "official_url": template["url"],
                    "description": template["description"],
                }
            )
    return rows


def build_exams() -> list[dict]:
    rows: list[dict] = []
    for template in EXAM_TEMPLATES:
        for index, year in enumerate(YEARS):
            for session_index, session in enumerate(template["sessions"]):
                drift = (index + session_index) % 5 - 2
                signup_start, exam_date = _align(
                    session["signup"], session["exam"], year, drift
                )
                rows.append(
                    {
                        "name": template["name"],
                        "category": template["category"],
                        "year": year,
                        "signup_start": signup_start,
                        "signup_end": signup_start + dt.timedelta(days=20),
                        "exam_date": exam_date,
                        "fee": template["fee"],
                        "target_grade": template["target_grade"],
                        "remark": template["remark"],
                    }
                )
    return rows


if __name__ == "__main__":
    comps = build_competitions()
    exams = build_exams()
    print(f"竞赛 {len(comps)} 条，考试 {len(exams)} 条，学生 {len(STUDENTS)} 条")
    print("样例竞赛：", comps[0])
    print("样例考试：", exams[0])
