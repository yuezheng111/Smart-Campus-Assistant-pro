"""建库建表 + 灌入结构化种子数据（竞赛 / 考试 / 学生）。

运行：python scripts/init_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app import config, db  # noqa: E402
from scripts import seed_data  # noqa: E402

SCHEMA_FILE = config.BASE_DIR / "db" / "schema.sql"


def split_sql(sql_text: str) -> list[str]:
    """把 .sql 文件按分号拆成一条条语句，跳过注释行。"""
    lines = []
    for raw in sql_text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("--"):
            continue
        lines.append(raw)
    body = "\n".join(lines)
    return [s.strip() for s in body.split(";") if s.strip()]


def create_schema() -> None:
    print(f"[1/3] 执行建表脚本 {SCHEMA_FILE.name} ...")
    statements = split_sql(SCHEMA_FILE.read_text(encoding="utf-8"))
    engine = db.get_server_engine()
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    print(f"      OK，共执行 {len(statements)} 条语句")


COMP_INSERT = """
INSERT INTO competition
    (name, category, level, organizer, year, signup_start, signup_end,
     contest_start, contest_end, target_grade, target_major, official_url, description)
VALUES
    (:name, :category, :level, :organizer, :year, :signup_start, :signup_end,
     :contest_start, :contest_end, :target_grade, :target_major, :official_url, :description)
"""

EXAM_INSERT = """
INSERT INTO exam
    (name, category, year, signup_start, signup_end, exam_date, fee, target_grade, remark)
VALUES
    (:name, :category, :year, :signup_start, :signup_end, :exam_date, :fee, :target_grade, :remark)
"""

STUDENT_INSERT = """
INSERT INTO student (student_no, name, grade, major, college, interests, gpa)
VALUES (:student_no, :name, :grade, :major, :college, :interests, :gpa)
"""


def seed_structured() -> None:
    print("[2/3] 灌入结构化数据 ...")
    competitions = seed_data.build_competitions()
    exams = seed_data.build_exams()
    students = seed_data.STUDENTS

    with db.get_engine().begin() as conn:
        conn.execute(text("TRUNCATE TABLE competition"))
        conn.execute(text("TRUNCATE TABLE exam"))
        conn.execute(text("TRUNCATE TABLE student"))
        conn.execute(text(COMP_INSERT), competitions)
        conn.execute(text(EXAM_INSERT), exams)
        conn.execute(text(STUDENT_INSERT), students)

    print(f"      竞赛 {len(competitions)} 条 / 考试 {len(exams)} 条 / 学生 {len(students)} 条")


def verify() -> None:
    print("[3/3] 校验 ...")
    for table in ("competition", "exam", "student"):
        row = db.fetch_one(f"SELECT COUNT(*) AS c FROM {table}")
        print(f"      {table:12s} -> {row['c']} 行")

    sample = db.fetch_all(
        "SELECT name, category, year, signup_start, contest_start "
        "FROM competition WHERE category = '计算机' ORDER BY year DESC LIMIT 3"
    )
    print("      样例（计算机类竞赛，最近 3 届）：")
    for r in sample:
        print(f"        - {r['year']} {r['name']}  报名 {r['signup_start']} / 比赛 {r['contest_start']}")


def main() -> None:
    print("=" * 64)
    print("校园智能助手 · 数据库初始化")
    print("=" * 64)
    create_schema()
    seed_structured()
    verify()
    print("\n完成。数据库：", config.MYSQL_DB)
    print()
    print("注意：schema.sql 会 DROP 并重建 document 表，所以本脚本会清空文件元数据。")
    print("     跑完本脚本后，必须接着执行 python run_ingest.py 重新灌入学校文件。")


if __name__ == "__main__":
    main()
