"""学生画像工具。

第三阶段「学业规划」需要知道「你是谁」，才能给出有针对性的建议。
真实系统里这里应该接学校的统一身份认证；
当前用一张演示表代替，字段结构是一样的。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from app import db


@tool
def get_student_profile(student_no: str = "", name: str = "") -> str:
    """查询学生的基本信息（年级、专业、学院、兴趣方向、GPA）。

    当学生说「我大二、计算机专业」时，其实信息已经够了，不必调这个工具；
    只有当学生提供了学号，或者需要确认学生身份时才调用。

    Args:
        student_no: 学号，例如「20240217」。
        name: 学生姓名。学号和姓名至少提供一个。
    """
    conditions: list[tuple[str, str, object]] = [
        ("student_no", "=", student_no.strip() if student_no else None),
        ("name", "=", name.strip() if name else None),
    ]
    where, params = db.build_where(conditions)
    if not params:
        return json.dumps(
            {"错误": "请至少提供学号或姓名"},
            ensure_ascii=False,
        )

    row = db.fetch_one(
        "SELECT student_no, name, grade, major, college, interests, gpa "
        f"FROM student{where} LIMIT 1",
        params,
    )
    if not row:
        return json.dumps(
            {"错误": "未找到该学生", "学号": student_no, "姓名": name}, ensure_ascii=False
        )
    return json.dumps({"学生信息": row}, ensure_ascii=False, default=str)
