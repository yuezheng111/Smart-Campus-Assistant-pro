"""管理端的「待确认变更」：模型只能提出，不能执行。

为什么执行能力不在工具集里
--------------------------
如果给模型一个 `update_document_meta(confirm=True)` 的工具，它完全可以先算 diff
再自己确认 —— 人工确认就变成了走过场。所以职责被切成两段：

    模型   → admin_propose_metadata_change()  → 写一条 pending 变更单（不动文件）
    人     → 前端看到 diff，点「确认」
    前端   → POST /api/admin/changes/{id}/apply → 真正落库 + 同步向量副本 + 记审计

这条边界是整套 HITL 设计里唯一不能妥协的地方：**决策和执行分处两个信任域**，
中间必须夹一次人的点击。撤销也是同理 —— 审计里存了旧值，才能一键回滚。
"""

from __future__ import annotations

import datetime as dt
import json
import re
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app import config, db
from app.logging_config import get_logger

log = get_logger(__name__)

# 允许模型提议修改的字段。刻意不含 file_path：
# 改路径是主键级变更，牵连磁盘文件、侧车 yaml、向量库归属，单独走 create_move_change，
# 不能让它在「改元数据」这条路上被顺手带出来。
EDITABLE_FIELDS = (
    "title",
    "doc_type",
    "college",
    "department",
    "doc_family",
    "year",
    "publish_date",
    "url",
)

# 文件名里不允许出现的字符（Windows 与 POSIX 的交集）
_UNSAFE_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

# 这些字段同时存在于向量库的 metadata 副本里，改完必须同步，否则检索过滤会对不上
_VECTOR_FIELDS = (
    "title", "doc_type", "college", "department",
    "doc_family", "year", "publish_date", "url",
)

_tables_ready = False


def _ensure_tables() -> None:
    """建变更单与审计表。

    用 CREATE TABLE IF NOT EXISTS 而不是写进 schema.sql：schema.sql 是
    DROP + CREATE 重建，跑一次会把 document 表清空。这里只补两张新表，
    绝不动已有的表。
    """
    global _tables_ready
    if _tables_ready:
        return
    with db.get_engine().begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS `admin_change` (
                `id`          CHAR(32) PRIMARY KEY,
                `doc_id`      INT NOT NULL,
                `file_path`   VARCHAR(512) DEFAULT NULL,
                `changes`     LONGTEXT,
                `before`      LONGTEXT,
                `status`      VARCHAR(16) NOT NULL DEFAULT 'pending',
                `operator`    VARCHAR(64) DEFAULT NULL,
                `created_at`  DATETIME DEFAULT CURRENT_TIMESTAMP,
                `applied_at`  DATETIME DEFAULT NULL,
                KEY `idx_status` (`status`),
                KEY `idx_doc` (`doc_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='管理端待确认变更单'
            """
        ))
        conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS `admin_audit` (
                `id`         BIGINT AUTO_INCREMENT PRIMARY KEY,
                `change_id`  CHAR(32) DEFAULT NULL,
                `doc_id`     INT DEFAULT NULL,
                `action`     VARCHAR(32) DEFAULT NULL,
                `before`     LONGTEXT,
                `after`      LONGTEXT,
                `operator`   VARCHAR(64) DEFAULT NULL,
                `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                KEY `idx_doc` (`doc_id`),
                KEY `idx_time` (`created_at`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='管理端操作审计'
            """
        ))
    _tables_ready = True


def _normalize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()[:10]
    return str(value).strip()


def create_change(
    doc_id: int,
    changes: dict[str, Any],
    operator: str = "admin",
) -> dict[str, Any]:
    """算 diff 并存一条 pending 变更单。**不修改 document 表。**"""
    _ensure_tables()

    allowed = {k: v for k, v in changes.items() if k in EDITABLE_FIELDS}
    dropped = sorted(set(changes) - set(allowed))
    if not allowed:
        return {"错误": "没有可修改的字段", "可改字段": list(EDITABLE_FIELDS),
                "被忽略的字段": dropped}

    row = db.fetch_one(
        f"SELECT id, file_path, {', '.join(EDITABLE_FIELDS)} "
        "FROM `document` WHERE id = :i",
        {"i": int(doc_id)},
    )
    if not row:
        return {"错误": f"没有 id={doc_id} 的文件", "提示": "先用 admin_list_documents 找到正确 id"}

    before: dict[str, Any] = {}
    diff: dict[str, Any] = {}
    for field, new_value in allowed.items():
        old = row.get(field)
        if _normalize(old) == _normalize(new_value):
            continue
        before[field] = _normalize(old)
        diff[field] = {"原值": _normalize(old), "新值": _normalize(new_value)}

    if not diff:
        return {"变更单": None, "说明": "拟修改的值与当前一致，没有需要确认的变更",
                "被忽略的字段": dropped}

    change_id = uuid.uuid4().hex
    # 列名一律加反引号：`before` 是 MySQL 保留字，不加会在 INSERT 的列清单处直接语法错误。
    db.execute(
        "INSERT INTO `admin_change` "
        "(`id`, `doc_id`, `file_path`, `changes`, `before`, `status`, `operator`) "
        "VALUES (:id, :doc_id, :file_path, :changes, :before, 'pending', :operator)",
        {
            "id": change_id,
            "doc_id": int(doc_id),
            "file_path": row.get("file_path"),
            "changes": json.dumps({k: _normalize(v) for k, v in allowed.items() if k in diff},
                                  ensure_ascii=False),
            "before": json.dumps(before, ensure_ascii=False),
            "operator": operator,
        },
    )
    log.info("生成变更单 %s | doc=%s | 字段=%s", change_id, doc_id, list(diff))

    return {
        "变更单": change_id,
        "文档": {"id": row.get("id"), "标题": row.get("title"), "file_path": row.get("file_path")},
        "变更内容": diff,
        "被忽略的字段": dropped,
        "说明": "这是**待确认**的变更，还没有写入。需要管理员在管理端核对 diff 后点击确认。",
        "确认接口": f"POST /api/admin/changes/{change_id}/apply",
    }


def _abs(rel_path: str) -> Path:
    candidate = Path(str(rel_path or "").strip().replace("\\", "/"))
    return candidate if candidate.is_absolute() else config.BASE_DIR / candidate


def _safe_filename(raw: str) -> str:
    """浏览器/模型给的文件名不可信：只取主干名，再过滤非法字符。"""
    name = Path(str(raw or "")).name
    stem, suffix = Path(name).stem, Path(name).suffix
    clean = _UNSAFE_NAME.sub("_", stem).strip(" .") or "file"
    return f"{clean}{suffix}"


def create_move_change(
    doc_id: int,
    new_filename: str,
    operator: str = "admin",
) -> dict[str, Any]:
    """提出「给文件改名」。只生成变更单，不动磁盘。

    限制只做**同目录改名**，不允许跨目录：loader.discover_paths 只扫 data/raw 一层
    （iterdir 不递归），把文件挪到子目录里，它就从知识库里"消失"了 ——
    表现是元数据表有记录、向量库有片段，但原件对账显示缺失。
    真要支持目录层级，得先改 discover_paths 的扫描规则，不是在这里放开。
    """
    _ensure_tables()

    row = db.fetch_one(
        "SELECT id, title, file_path FROM `document` WHERE id = :i", {"i": int(doc_id)}
    )
    if not row:
        return {"错误": f"没有 id={doc_id} 的文件", "提示": "先用 admin_list_documents 找到正确 id"}

    old_rel = str(row.get("file_path") or "").strip().replace("\\", "/")
    if not old_rel:
        return {"错误": "这份文件没有记录 file_path，无法改名"}

    old_abs = _abs(old_rel)
    clean = _safe_filename(new_filename)
    if not Path(clean).suffix:
        return {"错误": "新文件名缺少后缀", "提示": f"原后缀是 {old_abs.suffix}"}
    if Path(clean).suffix.lower() != old_abs.suffix.lower():
        return {
            "错误": "后缀必须与原文件一致",
            "原后缀": old_abs.suffix,
            "新后缀": Path(clean).suffix,
            "说明": "改后缀会让解析器换人，等于换了一份文件，不属于改名",
        }

    new_rel = (Path(old_rel).parent / clean).as_posix()
    if new_rel == old_rel:
        return {"变更单": None, "说明": "新文件名与原文件名一致，不需要改名"}
    if _abs(new_rel).exists():
        return {"错误": f"目标已存在：{new_rel}", "提示": "换一个名字，或先处理掉同名文件"}
    if not old_abs.is_file():
        return {"错误": f"原件不在磁盘上：{old_rel}", "提示": "先把它放回 data/raw 再改名"}

    change_id = uuid.uuid4().hex
    db.execute(
        "INSERT INTO `admin_change` "
        "(`id`, `doc_id`, `file_path`, `changes`, `before`, `status`, `operator`) "
        "VALUES (:id, :doc_id, :file_path, :changes, :before, 'pending', :operator)",
        {
            "id": change_id,
            "doc_id": int(doc_id),
            "file_path": old_rel,
            "changes": json.dumps({"file_path": new_rel}, ensure_ascii=False),
            "before": json.dumps({"file_path": old_rel}, ensure_ascii=False),
            "operator": operator,
        },
    )
    log.info("生成改名变更单 %s | doc=%s | %s -> %s", change_id, doc_id, old_rel, new_rel)
    return {
        "变更单": change_id,
        "文档": {"id": row.get("id"), "标题": row.get("title")},
        "变更内容": {"file_path": {"原值": old_rel, "新值": new_rel}},
        "说明": "这是**待确认**的改名，磁盘文件还没动。需要管理员在管理端确认。",
        "确认接口": f"POST /api/admin/changes/{change_id}/apply",
    }


def _apply_move(change: dict[str, Any], new_rel: str, operator: str) -> dict[str, Any]:
    """执行改名。顺序：磁盘 → MySQL → 向量库副本 → 侧车 yaml。

    顺序不是随便定的：磁盘先动，失败时数据库还指着旧路径，对账能看出"原件缺失"，
    可重试；反过来先改数据库，磁盘一失败就变成"库里指向一个不存在的文件"，
    学生端搜得到、点开 404，比前者难查得多。

    三步各自幂等：磁盘已挪动、库已更新、副本已同步，任一环节重跑都不会出错。
    """
    from app.rag import loader
    from app.rag.vectorstore import update_metadata_by_file_path

    change_id = str(change["id"])
    doc_id = int(change["doc_id"])
    old_rel = str(change.get("file_path") or "").strip().replace("\\", "/")
    new_rel = str(new_rel).strip().replace("\\", "/")
    old_abs, new_abs = _abs(old_rel), _abs(new_rel)

    if old_abs == new_abs:
        raise ValueError("新旧路径相同")

    moved_on_disk = False
    if new_abs.is_file() and not old_abs.is_file():
        pass  # 上次已经搬过，只剩数据库没同步 —— 幂等重跑
    elif not old_abs.is_file():
        raise ValueError(f"原件不在磁盘上：{old_rel}")
    elif new_abs.exists():
        raise ValueError(f"目标已存在，拒绝覆盖：{new_rel}")
    else:
        old_abs.replace(new_abs)
        moved_on_disk = True

    try:
        with db.get_engine().begin() as conn:
            conn.execute(
                text("UPDATE `document` SET `file_path` = :p WHERE `id` = :i"),
                {"p": new_rel, "i": doc_id},
            )
            conn.execute(
                text(
                    "UPDATE `admin_change` SET status = 'applied', applied_at = NOW() "
                    "WHERE id = :c"
                ),
                {"c": change_id},
            )
    except Exception:
        # 数据库没改成 → 把磁盘恢复回去，保证"要么都别动"
        if moved_on_disk and new_abs.is_file() and not old_abs.exists():
            new_abs.replace(old_abs)
        raise

    patched = 0
    warning = ""
    try:
        patched = update_metadata_by_file_path(old_rel, {"file_path": new_rel})
        if patched == 0:
            warning = "向量库里没有这份文件的片段，改名后仍需重新灌库"
    except Exception as exc:
        warning = f"向量库同步失败：{exc}。检索与原件路径会不一致，需要重跑同步。"
        log.warning("改名 %s 同步向量失败：%s", change_id, exc)

    sidecar_moved = False
    old_side = loader.sidecar_path(old_abs)
    new_side = loader.sidecar_path(new_abs)
    if old_side.is_file():
        try:
            if new_side.exists():
                new_side.unlink()
            old_side.replace(new_side)
            sidecar_moved = True
        except OSError as exc:
            warning = (warning + " " if warning else "") + f"侧车元数据改名失败：{exc}"

    db.execute(
        "INSERT INTO `admin_audit` "
        "(`change_id`, `doc_id`, `action`, `before`, `after`, `operator`) "
        "VALUES (:c, :d, 'move_file', :b, :a, :o)",
        {
            "c": change_id,
            "d": doc_id,
            "b": json.dumps({"file_path": old_rel}, ensure_ascii=False),
            "a": json.dumps({"file_path": new_rel}, ensure_ascii=False),
            "o": operator,
        },
    )
    log.info(
        "改名完成 | doc=%s | %s -> %s | 向量 %d | 侧车 %s",
        doc_id, old_rel, new_rel, patched, sidecar_moved,
    )
    return {
        "变更单": change_id,
        "文档 id": doc_id,
        "旧路径": old_rel,
        "新路径": new_rel,
        "向量副本同步条数": patched,
        "侧车元数据已改名": sidecar_moved,
        "提示": warning or "已生效",
    }


def _refresh_chunk_titles(
    file_path: str,
    old_title: str,
    new_title: str,
) -> tuple[int, str]:
    """标题变了，切片正文里的【旧标题】前缀要跟着换。

    为什么必须重算向量而不是只改 metadata：切分时把标题拼进了 page_content
    （`【标题】\\n正文`），那是为了让 embedding 也"看见"标题 —— 只拿一段
    「二、申请条件……」，向量模型不知道这是哪份文件，召回会明显变差。
    所以标题既是展示字段，也是向量的一部分，改了就得重新嵌入。

    代价只有这一份文件的切片，不是全量重灌。

    写入顺序是「先加新的、再删旧的」：反过来先删后加，中间一旦失败，
    这份文件就彻底从检索里消失了；先加后删最坏也只是短暂重复，
    重复能被检索发现、能被重灌修好，丢失不能。
    """
    from langchain_core.documents import Document

    from app.rag.vectorstore import get_vectorstore

    store = get_vectorstore()
    try:
        found = store.get(where={"file_path": file_path}, include=["documents", "metadatas"])
    except Exception as exc:
        return 0, f"读取切片失败，标题前缀未刷新：{exc}"

    ids = list(found.get("ids") or [])
    documents = list(found.get("documents") or [])
    metadatas = list(found.get("metadatas") or [])
    if not ids:
        return 0, "向量库里没有这份文件的切片，无需刷新标题前缀"

    prefix = f"【{old_title}】"
    rebuilt = []
    for text, meta in zip(documents, metadatas):
        body = text.split("\n", 1)[1] if text.startswith(prefix) and "\n" in text else text
        rebuilt.append(
            Document(
                page_content=f"【{new_title}】\n{body}",
                metadata={**dict(meta or {}), "title": new_title},
            )
        )

    try:
        store.add_documents(rebuilt)
    except Exception as exc:
        return 0, f"重新嵌入失败，标题前缀仍是旧的：{exc}"

    warning = ""
    try:
        store.delete(ids=ids)
    except Exception as exc:
        warning = f"旧切片删除失败，可能出现短暂重复：{exc}"
        log.warning("刷新标题后删除旧切片失败 %s：%s", file_path, exc)
    return len(rebuilt), warning


def get_change(change_id: str) -> dict[str, Any] | None:
    _ensure_tables()
    row = db.fetch_one("SELECT * FROM `admin_change` WHERE id = :i", {"i": change_id})
    if not row:
        return None
    return {
        **row,
        "changes": json.loads(row.get("changes") or "{}"),
        "before": json.loads(row.get("before") or "{}"),
    }


def apply_change(change_id: str, operator: str = "admin") -> dict[str, Any]:
    """确认后真正落库：MySQL 正本 → 向量库副本 → 审计。

    顺序不能反。先改 MySQL：它是正本，如果这一步失败，向量库一个字都没动，
    状态干净；反过来先改向量库，一旦后面失败，检索过滤就会和正本对不上，
    表现成「列表里是 A 学院、检索时按 A 学院却搜不到」。
    """
    change = get_change(change_id)
    if not change:
        raise LookupError(f"没有 id={change_id} 的变更单")
    if change.get("status") != "pending":
        raise ValueError(f"变更单状态是 {change.get('status')}，不能重复执行")

    doc_id = int(change["doc_id"])
    changes: dict[str, Any] = change["changes"] or {}
    if not changes:
        raise ValueError("变更单里没有有效字段")

    # 改路径是另一条链路（磁盘 + 数据库 + 向量 + 侧车），不能和改字段混在一起跑
    if "file_path" in changes:
        return _apply_move(change, str(changes["file_path"]), operator)

    assignments = ", ".join(f"`{k}` = :{k}" for k in changes)
    params: dict[str, Any] = {k: (v if v != "" else None) for k, v in changes.items()}
    params["id"] = doc_id
    with db.get_engine().begin() as conn:
        conn.execute(
            text(f"UPDATE `document` SET {assignments} WHERE id = :id"), params
        )
        conn.execute(
            text("UPDATE `admin_change` SET status = 'applied', applied_at = NOW() WHERE id = :i"),
            {"i": change_id},
        )

    vector_path = str(change.get("file_path") or "")
    vector_changes = {k: v for k, v in changes.items() if k in _VECTOR_FIELDS}
    patched = 0
    warning = ""
    if vector_path and vector_changes:
        try:
            from app.rag.vectorstore import update_metadata_by_file_path

            # Chroma 只收标量：year 必须是 int，其余是 str
            normalized = {
                k: (int(v) if k == "year" and str(v).strip() else v)
                for k, v in vector_changes.items()
            }
            patched = update_metadata_by_file_path(vector_path, normalized)
        except Exception as exc:
            warning = f"向量库副本同步失败：{exc}。检索过滤可能与列表不一致，需要重跑同步。"
            log.warning("变更单 %s 同步向量失败：%s", change_id, exc)

    db.execute(
        "INSERT INTO `admin_audit` "
        "(`change_id`, `doc_id`, `action`, `before`, `after`, `operator`) "
        "VALUES (:c, :d, 'update_meta', :b, :a, :o)",
        {
            "c": change_id,
            "d": doc_id,
            "b": json.dumps(change.get("before") or {}, ensure_ascii=False),
            "a": json.dumps(changes, ensure_ascii=False),
            "o": operator,
        },
    )

    refreshed = 0
    if "title" in changes:
        old_title = str((change.get("before") or {}).get("title") or "")
        refreshed, note = _refresh_chunk_titles(
            vector_path, old_title, str(changes["title"])
        )
        if note:
            warning = (warning + " " if warning else "") + note

    log.info("变更单 %s 已执行 | doc=%s | 向量同步 %d 个 chunk", change_id, doc_id, patched)
    return {
        "变更单": change_id,
        "文档 id": doc_id,
        "已修改字段": list(changes),
        "向量副本同步条数": patched,
        "重新嵌入切片数": refreshed,
        "提示": warning or "已生效",
        "可回滚": bool(change.get("before")),
        "原值": change.get("before"),
    }


def list_changes(limit: int = 20) -> list[dict[str, Any]]:
    _ensure_tables()
    rows = db.fetch_all(
        "SELECT id, doc_id, status, operator, created_at, applied_at "
        "FROM `admin_change` ORDER BY created_at DESC LIMIT :l",
        {"l": max(1, min(int(limit), 100))},
    )
    return rows
