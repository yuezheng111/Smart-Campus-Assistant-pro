"""MySQL 访问层。

统一 engine 的创建，并对外暴露三个安全查询函数。
所有值都通过 SQLAlchemy 的绑定参数传入，不做字符串拼接 —— 这是防 SQL 注入的关键。
"""

from __future__ import annotations

from typing import Any, Sequence
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app import config

_engine: Engine | None = None
_server_engine: Engine | None = None


def _build_url(database: str | None) -> str:
    pwd = quote_plus(config.MYSQL_PASSWORD)
    user = quote_plus(config.MYSQL_USER)
    db_part = f"/{database}" if database else ""
    return (
        f"mysql+pymysql://{user}:{pwd}@{config.MYSQL_HOST}:{config.MYSQL_PORT}"
        f"{db_part}?charset=utf8mb4"
    )


def get_engine() -> Engine:
    """连接到业务库 school_agent。"""
    global _engine
    if _engine is None:
        _engine = create_engine(
            _build_url(config.MYSQL_DB), pool_pre_ping=True, pool_recycle=3600
        )
    return _engine


def get_server_engine() -> Engine:
    """连接到 MySQL 服务器本身（不指定库），用于 CREATE DATABASE。"""
    global _server_engine
    if _server_engine is None:
        _server_engine = create_engine(
            _build_url(None), pool_pre_ping=True, pool_recycle=3600
        )
    return _server_engine


def fetch_all(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params or {}).mappings().all()
    return [dict(r) for r in rows]


def fetch_one(sql: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    rows = fetch_all(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: dict[str, Any] | None = None) -> int:
    with get_engine().begin() as conn:
        result = conn.execute(text(sql), params or {})
        return result.rowcount


def execute_many(sql: str, rows: Sequence[dict[str, Any]]) -> int:
    if not rows:
        return 0
    with get_engine().begin() as conn:
        conn.execute(text(sql), list(rows))
    return len(rows)


def build_where(conditions: Sequence[tuple[str, str, Any]]) -> tuple[str, dict[str, Any]]:
    """把过滤条件拼成参数化 WHERE 片段。

    conditions 里每一项是 (列名, 操作符, 值)：
        ("category", "=", "计算机")
        ("name", "like", "蓝桥杯")
        ("college", "in", ("计算机学院", "全校"))
        ("file_missing_at", "is_null", None)      # 值会被忽略

    列名和操作符由调用方在代码里写死（不是用户输入），值一律走绑定参数 ——
    这样即使值里混进了 `' OR 1=1 --`，也只会被当成一个普通字符串。
    """
    allowed_ops = {
        "=", ">", ">=", "<", "<=", "!=", "like", "in", "or_like",
        "is_null", "is_not_null",
    }
    clauses: list[str] = []
    params: dict[str, Any] = {}

    for index, (col, op, value) in enumerate(conditions):
        # IS NULL 没有「值」可比，所以必须放在下面「空值就跳过」之前判断，
        # 否则 `(列, 'is_null', None)` 会因为 value 是 None 被直接丢掉。
        if op == "is_null":
            clauses.append(f"`{col}` IS NULL")
            continue
        if op == "is_not_null":
            clauses.append(f"`{col}` IS NOT NULL")
            continue
        if value is None or value == "" or op not in allowed_ops:
            continue
        if op == "in":
            placeholders = []
            for sub_index, item in enumerate(value):
                key = f"w{index}_{sub_index}"
                placeholders.append(f":{key}")
                params[key] = item
            clauses.append(f"`{col}` IN ({', '.join(placeholders)})")
            continue
        if op == "or_like":
            # 同义词展开：「四六级」要能同时匹配到「四级」和「六级」两条记录
            parts = []
            for sub_index, item in enumerate(value):
                key = f"w{index}_{sub_index}"
                parts.append(f"`{col}` LIKE :{key}")
                params[key] = f"%{item}%"
            clauses.append("(" + " OR ".join(parts) + ")")
            continue

        key = f"w{index}"
        if op == "like":
            clauses.append(f"`{col}` LIKE :{key}")
            params[key] = f"%{value}%"
        else:
            clauses.append(f"`{col}` {op} :{key}")
            params[key] = value

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params
