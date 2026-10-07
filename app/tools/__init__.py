"""工具层统一出口。

按 Agent 的路由分支把工具分了两组：
    DOC_TOOLS  —— 走 RAG / 文件元数据（第一条线）
    DATA_TOOLS —— 走 MySQL 结构化数据（第二条线）
这样 LangGraph 在「只查文件」的分支里就只会挂上 DOC_TOOLS，
模型没有机会误调竞赛库 —— 用工具可见性来约束行为，比在 prompt 里写
「请你不要调用 xxx」要可靠得多。
"""

from app.tools.competition import query_competition, recommend_competition
from app.tools.document import (
    get_document_detail,
    list_school_documents,
    search_school_document,
)
from app.tools.exam import query_exam
from app.tools.student import get_student_profile

DOC_TOOLS = [search_school_document, list_school_documents, get_document_detail]
DATA_TOOLS = [query_competition, recommend_competition, query_exam, get_student_profile]
ALL_TOOLS = DOC_TOOLS + DATA_TOOLS

__all__ = [
    "search_school_document",
    "list_school_documents",
    "get_document_detail",
    "query_competition",
    "recommend_competition",
    "query_exam",
    "get_student_profile",
    "DOC_TOOLS",
    "DATA_TOOLS",
    "ALL_TOOLS",
]
