"""Tools 层冒烟测试：不经过 LLM，直接调用每个工具，确认返回结构正确。

运行：python scripts/test_tools.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools import (  # noqa: E402
    get_document_detail,
    get_student_profile,
    list_school_documents,
    query_competition,
    query_exam,
    recommend_competition,
    search_school_document,
)


def show(title: str, raw: str, preview: int = 380) -> None:
    data = json.loads(raw)
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    print(text[:preview] + ("\n  …(截断)" if len(text) > preview else ""))


def main() -> None:
    show("① RAG：国家奖学金评选条件（filter=奖学金）",
         search_school_document.invoke({"query": "国家奖学金评选条件", "doc_type": "奖学金"}))

    show("② 元数据列表：计算机学院的文件",
         list_school_documents.invoke({"college": "计算机学院"}))

    show("③ 竞赛查询：计算机类 2025 年",
         query_competition.invoke({"category": "计算机", "year": 2025}))

    show("④ 竞赛推荐：大二 / 软件工程 / Web开发,创新创业",
         recommend_competition.invoke(
             {"grade": "大二", "major": "软件工程", "interests": "Web开发,创新创业", "limit": 4}
         ))

    show("⑤ 考试查询：英语类",
         query_exam.invoke({"category": "英语", "keyword": "四六级", "limit": 4}))

    show("⑥ 学生画像：20240217",
         get_student_profile.invoke({"student_no": "20240217"}))

    first = json.loads(list_school_documents.invoke({"limit": 1})).get("文件清单") or []
    if first:
        show("⑦ 文件详情", get_document_detail.invoke({"doc_id": first[0]["id"]}))

    print("\n全部工具调用成功。")


if __name__ == "__main__":
    main()
