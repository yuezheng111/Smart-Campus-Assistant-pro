"""灌库编排：把「文件」变成「可检索的知识」。

这是整个系统**唯一的写入入口**，两个场景共用同一份逻辑：

    全量重建（run_ingest.py）      data/raw 里的全部文件，清空后重写
    增量上传（管理后台上传接口）    只处理刚上传的那一份

为什么要共用？如果两条路各写一套，"什么算需要更新"的判断标准迟早会漂移，
最后出现「命令行跑出来的结果和界面上的结果不一样」这种最难查的问题。

写入顺序与幂等性
----------------
MySQL 和 Chroma 是两个独立存储，没有跨库事务。所以这里不追求
"要么全成功、要么全失败"，而是追求 **可重跑**：每一步都是
"按 file_path 覆盖"，重跑任意多次结果都一样。中途断了，再跑一次即可回到一致状态。
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import bindparam, text

from app import config, db
from app.logging_config import get_logger
from app.rag import loader, splitter, vectorstore

log = get_logger(__name__)

_has_hash_column: bool | None = None
_has_lifecycle_columns: bool | None = None

# 内容侧字段：从文件（或侧车）读出来，重建时覆盖
_CONTENT_COLUMNS = (
    "doc_no", "title", "department", "doc_type", "college", "publish_date",
    "year", "file_path", "url", "summary",
)

# 状态侧字段：只在数据库里存在，重建时从旧记录回填、绝不覆盖
_STATE_COLUMNS = ("status", "superseded_by", "superseded_at")

# 这一列也属于状态侧，但重建时的规矩**相反**：不回填，直接置 NULL。
# 因为「这份文件正在被处理」本身就等于「原件好好地躺在 data/raw 里」，
# 没有比这更可靠的证据。落点在 sync_metadata。
_MISSING_COLUMN = "file_missing_at"

# 生命周期扩展列（内容侧 + 状态侧一起加进表）
_LIFECYCLE_COLUMNS = (
    "doc_family", "version", "effective_from", "effective_to",
) + _STATE_COLUMNS + (_MISSING_COLUMN,)

_insert_sql_cache: dict[tuple[bool, bool], str] = {}


def _insert_sql(use_hash: bool, use_lifecycle: bool) -> str:
    """按数据库实际有哪些列，拼出对应的 INSERT 语句。

    schema.sql 是 DROP + CREATE 重建表，所以「库比代码旧」是常态
    （没重跑 init_db.py 就缺新列）。与其在每一步写 if/else 拼两套语句，
    不如按列存在与否生成一条 —— 列少的时候自动降级，不让整条管线停摆。
    """
    key = (use_hash, use_lifecycle)
    cached = _insert_sql_cache.get(key)
    if cached:
        return cached

    columns = list(_CONTENT_COLUMNS)
    if use_hash:
        columns.append("content_hash")
    if use_lifecycle:
        columns.extend(_LIFECYCLE_COLUMNS)

    sql = (
        f"INSERT INTO document ({', '.join(columns)}) "
        f"VALUES ({', '.join(':' + c for c in columns)})"
    )
    _insert_sql_cache[key] = sql
    return sql


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    """流式算文件哈希。

    基于**原始字节**而不是解析后的文本：同一个 PDF 在不同版本的 pypdf 下
    可能抽出不一样的文字，用它算哈希会让「文件没改」被误判成「改过了」，
    白白重算一次 embedding —— 也就是白花钱。字节是稳定的，文本不是。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _supports_content_hash() -> bool:
    """数据库里有没有 content_hash 列。

    schema.sql 是 DROP + CREATE 重建表，没重跑过 init_db.py 的库会缺这一列。
    这里探测一次并缓存结果：缺了就退回「不带哈希」的写入，
    而不是让整条灌库管线直接报错停摆。
    """
    global _has_hash_column
    if _has_hash_column is None:
        try:
            _has_hash_column = bool(
                db.fetch_one("SHOW COLUMNS FROM `document` LIKE 'content_hash'")
            )
        except Exception as exc:
            log.warning("探测 content_hash 列失败：%s", exc)
            _has_hash_column = False
        if not _has_hash_column:
            log.warning(
                "document 表缺少 content_hash 列，本次不记录内容哈希。"
                "重跑 python scripts/init_db.py 可补齐。"
            )
    return _has_hash_column


def _supports_lifecycle() -> bool:
    """数据库里有没有生命周期字段（status / file_missing_at）。

    同样是「库可能比代码旧」的兼容处理：缺列时降级成「所有文件都当现行」，
    而不是让整条灌库管线直接报错停摆。

    刻意要求**两列都在**才算数：只探测 status 是不够的 —— 加了新列却没重跑
    init_db.py 的库，会出现「有 status、没 file_missing_at」的半吊子状态，
    那时 INSERT 的列列表照旧带上新列，直接撞 SQL 错误。宁可整体降级。
    """
    global _has_lifecycle_columns
    if _has_lifecycle_columns is None:
        try:
            columns = {
                str(row.get("Field") or "")
                for row in db.fetch_all("SHOW COLUMNS FROM `document`")
            }
            _has_lifecycle_columns = {"status", _MISSING_COLUMN} <= columns
        except Exception as exc:
            log.warning("探测生命周期字段失败：%s", exc)
            _has_lifecycle_columns = False
        if not _has_lifecycle_columns:
            log.warning(
                "document 表缺少 status / %s 等生命周期列，本次不维护文件状态。"
                "重跑 python scripts/init_db.py 可补齐。",
                _MISSING_COLUMN,
            )
    return _has_lifecycle_columns


def _delete_paths(conn: Any, paths: list[str]) -> int:
    """按 file_path 删掉旧记录。增量写入的「先删后插」。"""
    if not paths:
        return 0
    stmt = text("DELETE FROM document WHERE file_path IN :paths").bindparams(
        bindparam("paths", expanding=True)
    )
    return conn.execute(stmt, {"paths": paths}).rowcount


def _mark_all_gone(conn: Any, today: dt.date) -> int:
    """把「还标着原件在」的记录统一标成原件缺失。

    全量重建时用它替代「按路径比差集」。看起来粗暴 —— 把全部记录都标上，
    再让本次真正处理过的那些行「删掉重插」自然恢复成 NULL —— 但它换来一个
    很实际的好处：**不需要一条超长的 IN 列表**。上万份文件时，
    `WHERE file_path NOT IN (一万个路径)` 拼出来的 SQL 能有几百 KB，
    而这里始终是一条不带参数的 UPDATE。

    顺序不能反：必须在删除本次要重写的行**之前**执行。
    先删的话，那批行已经不在了，「没被处理 = 原件消失」的判断就失效了。
    """
    stmt = text(
        f"UPDATE document SET {_MISSING_COLUMN} = :today "
        f"WHERE {_MISSING_COLUMN} IS NULL"
    )
    return conn.execute(stmt, {"today": today}).rowcount


def load_states(file_paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    """读出这些文件在数据库里已有的「状态侧」字段。

    为什么必须先读再删：这批文件的写入方式是「先删旧行、再插新行」，
    而 status / superseded_by 这些值**只有数据库里有**（侧车文件里不写）。
    删之前不读出来，一份已废止的文件跑一次全量重建就会复活成现行。

    这就是「内容侧 / 状态侧」分工的落点：

        内容侧（title/doc_type/doc_family…）→ 文件是真相源，重建时覆盖
        状态侧（status/superseded_by…）    → 数据库是真相源，重建时回填
    """
    paths = list(dict.fromkeys(str(p) for p in file_paths if p))
    if not paths or not _supports_lifecycle():
        return {}

    stmt = text(
        "SELECT file_path, status, superseded_by, superseded_at "
        "FROM document WHERE file_path IN :paths"
    ).bindparams(bindparam("paths", expanding=True))
    try:
        with db.get_engine().begin() as conn:
            rows = conn.execute(stmt, {"paths": paths}).mappings().all()
    except Exception as exc:
        log.warning("读取文件旧状态失败，本次按默认状态写入：%s", exc)
        return {}
    return {str(row["file_path"]): dict(row) for row in rows}


def sync_metadata(
    docs: list[loader.LoadedDoc],
    hashes: dict[str, str] | None = None,
    *,
    full: bool = False,
    purge: bool = False,
    states: dict[str, dict[str, Any]] | None = None,
) -> int:
    """把文件元数据写进 MySQL，供前端列表和「按条件筛选」使用。

    full=True  整批重写，并把「这次没被处理」的记录标成原件缺失（保留，不删）
    full=False 只替换这批文件对应的行（对应「增量上传」）
    purge=True 真正的从零重建：连原件已消失的记录一起清掉（full 的加强版）

    这里刻意不用 TRUNCATE：TRUNCATE 在 MySQL 里是隐式提交，一旦表被清空
    之后的插入又失败，整张表就空了，没有任何回滚机会。改用 DELETE 让它
    留在同一个事务里 —— 要么一起生效，要么一起回退。

    states 是这批文件原有的状态侧字段，由调用方在删除之前用 load_states()
    读出并传进来。没传（或库里本来就没这条记录）时按默认状态 active 处理。
    """
    hashes = hashes or {}
    states = states or {}
    use_hash = _supports_content_hash()
    use_lifecycle = _supports_lifecycle()

    rows: list[dict[str, Any]] = []
    for doc in docs:
        row: dict[str, Any] = {
            "doc_no": doc.doc_no,
            "title": doc.title,
            "department": doc.department,
            "doc_type": doc.doc_type,
            "college": doc.college,
            "publish_date": doc.publish_date,
            "year": doc.year,
            "file_path": doc.file_path,
            "url": doc.url,
            "summary": loader.make_summary(doc),
        }
        if use_hash:
            row["content_hash"] = hashes.get(doc.file_path)
        if use_lifecycle:
            # 内容侧 —— 来自文件本身
            row["doc_family"] = doc.doc_family or None
            row["version"] = doc.version
            row["effective_from"] = doc.effective_from
            row["effective_to"] = doc.effective_to
            # 状态侧 —— 来自旧记录；没有旧记录才落到默认值
            old = states.get(doc.file_path) or {}
            row["status"] = old.get("status") or "active"
            row["superseded_by"] = old.get("superseded_by")
            row["superseded_at"] = old.get("superseded_at")
            # 这一列**不回填**：能走到这里，说明文件刚刚被成功解析过，
            # 原件必然还在磁盘上。回填反而会把「原件已回来」误判成「仍然缺失」。
            row["file_missing_at"] = None
        rows.append(row)

    with db.get_engine().begin() as conn:
        if purge:
            # 从零重建：一切推倒重来，孤儿记录也不留
            conn.execute(text("DELETE FROM document"))
        else:
            if full and use_lifecycle and _MISSING_COLUMN:
                # 全量重建但**保留**孤儿：先把所有「还标着原件在」的记录标成缺失，
                # 本次真正处理过的那些行紧接着会「删掉重插」自然恢复成 NULL。
                # 这样原件已消失的文件只会退出检索，不会被从库里抹掉。
                marked = _mark_all_gone(conn, dt.date.today())
                if marked:
                    log.info("全量重建：%d 条记录标记为原件缺失（保留，不删除）", marked)
            if rows:
                _delete_paths(conn, [r["file_path"] for r in rows])
        if rows:
            conn.execute(text(_insert_sql(use_hash, use_lifecycle)), rows)
    return len(rows)


def _unchanged_files(
    docs: list[loader.LoadedDoc],
    hashes: dict[str, str],
) -> set[str]:
    """找出「内容和档案都没变」的文件，这些可以整个跳过。

    为什么要把「元数据」也一起比：如果只比内容哈希，会出现一个反直觉的
    坏情况 —— 管理员改错了学院，重新上传把学院改对了，但因为 PDF 字节
    没变，系统判定「没变化」直接跳过，于是错误一直留在库里。
    元数据变了就必须重写一遍。

    注意这里**只比内容侧**，不比 status。原因见 sync_metadata：
    status 由数据库说了算，本来就不该被「文件变没变」牵着走 ——
    把它放进比较，反而会让废止操作看起来像是「文件变了」。

    这个判断只在增量场景用。全量重建本来就该按文件重来，不查这个。
    """
    if not _supports_content_hash():
        return set()

    use_lifecycle = _supports_lifecycle()
    columns = [
        "content_hash", "title", "doc_type", "college", "department",
        "doc_no", "publish_date", "url",
    ]
    if use_lifecycle:
        columns += ["doc_family", "version", "effective_from", "effective_to"]
    sql = f"SELECT {', '.join(columns)} FROM document WHERE file_path = :p"

    def as_date(value: Any) -> str:
        return value.isoformat() if value else ""

    unchanged: set[str] = set()
    for doc in docs:
        new_hash = hashes.get(doc.file_path)
        if not new_hash:
            continue
        row = db.fetch_one(sql, {"p": doc.file_path})
        if not row or str(row.get("content_hash") or "") != new_hash:
            continue

        same_meta = (
            str(row.get("title") or "") == doc.title
            and str(row.get("doc_type") or "") == doc.doc_type
            and str(row.get("college") or "") == doc.college
            and str(row.get("department") or "") == doc.department
            and str(row.get("doc_no") or "") == doc.doc_no
            and str(row.get("url") or "") == doc.url
            and str(row.get("publish_date") or "") == as_date(doc.publish_date)
        )
        if same_meta and use_lifecycle:
            same_meta = (
                str(row.get("doc_family") or "") == doc.doc_family
                and int(row.get("version") or 0) == int(doc.version or 0)
                and str(row.get("effective_from") or "") == as_date(doc.effective_from)
                and str(row.get("effective_to") or "") == as_date(doc.effective_to)
            )
        if same_meta:
            unchanged.add(doc.file_path)

    return unchanged


def _missing_paths() -> list[str]:
    """列出所有被标记为「原件已缺失」的文件路径。

    全量重建写向量库时要用：这批文件的片段必须原地保留、只改标记，
    不能跟着「先删后写」的节奏被清掉。
    """
    if not _supports_lifecycle():
        return []
    try:
        rows = db.fetch_all(
            f"SELECT file_path FROM document WHERE {_MISSING_COLUMN} IS NOT NULL"
        )
    except Exception as exc:
        log.warning("读取原件缺失清单失败：%s", exc)
        return []
    return [str(r["file_path"]) for r in rows if r.get("file_path")]


def ingest_paths(
    paths: Iterable[Path],
    *,
    full: bool = False,
    purge: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """处理指定的一批文件：解析 → 切分 → 写 MySQL → 写向量库。

    full=True   整批重写；原件已消失的文件会被标记保留，而不是清掉
    full=False  先按 file_path 删掉这些文件的旧片段再写（增量，幂等）

    purge=True  真正的从零重建：元数据表与向量库一起清空重来。
                换向量库、或怀疑库里全是脏数据时才用它 —— 它会把
                「原件已消失」那部分留档一并丢掉，所以默认不开。
    """
    if purge:
        full = True
    path_list = [Path(p) for p in paths]

    def say(message: str) -> None:
        if verbose:
            print(message)
        log.info(message)

    say(f"[1/4] 解析 {len(path_list)} 个文件 ...")
    docs = loader.load_paths(path_list)
    if not docs:
        say("      没有可处理的文档（格式不支持、内容为空，或全是扫描件）")
        return {
            "文档数": 0,
            "chunk数": 0,
            "跳过数": 0,
            "向量库总数": vectorstore.count(),
            "全量": full,
            "文件": [],
        }
    say(f"      加载 {len(docs)} 份文档")

    hashes: dict[str, str] = {}
    for path in path_list:
        try:
            hashes[loader.relative_path(path)] = file_sha256(path)
        except OSError as exc:
            log.warning("计算文件哈希失败 %s：%s", path.name, exc)

    # 增量场景下，内容和档案都一模一样的文件直接跳过 ——
    # 跳过就等于不调 embedding 接口，也就是不花钱。
    skipped = 0
    if not full:
        unchanged = _unchanged_files(docs, hashes)
        if unchanged:
            docs = [doc for doc in docs if doc.file_path not in unchanged]
            skipped = len(unchanged)
            say(f"      {skipped} 份文件的内容与档案都没变 → 跳过，不重算向量")
        if not docs:
            say("      没有需要更新的文件，结束")
            return {
                "文档数": 0,
                "chunk数": 0,
                "跳过数": skipped,
                "向量库总数": vectorstore.count(),
                "全量": full,
                "文件": [],
            }

    say(f"[2/4] 切分（chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}）...")
    chunks = splitter.split_documents(docs)
    say(f"      得到 {len(chunks)} 个 chunk")

    # 「现行 / 已废止」是运行时状态，文件里读不出来，只在数据库里。
    # 所以必须在删旧记录**之前**读出来，写入时再回填 ——
    # 否则一份已废止的文件跑一次全量重建就会复活成现行。
    states = load_states(doc.file_path for doc in docs)

    say("[3/4] 写入 MySQL document 表 ...")
    written = sync_metadata(docs, hashes, full=full, purge=purge, states=states)
    say(f"      写入 {written} 条文件元数据")

    # 向量库里的 status 是**副本**，它存在的唯一理由是检索时必须能预先过滤
    # —— 过滤要发生在取 top_k 之前，否则会召回到「一堆已废止文件里最相似的
    # 那几条」，把现行文件挤掉。副本从正本（数据库）同步过来。
    if states:
        patched = 0
        for chunk in chunks:
            state = states.get(str(chunk.metadata.get("file_path") or "")) or {}
            status = state.get("status")
            if status:
                chunk.metadata["status"] = status
                patched += 1
        if patched:
            say(f"      {patched} 个 chunk 沿用数据库里的状态（含已废止）")

    say(f"[4/4] 写入向量库 {config.CHROMA_DIR} ...")
    if purge:
        # 从零重建：整个集合丢掉重来。原件已消失的那部分片段也一并清掉 ——
        # 这正是 purge 与默认全量的区别，所以要显式开。
        vectorstore.reset_vectorstore()
    else:
        removed = vectorstore.delete_many([doc.file_path for doc in docs])
        if removed:
            say(f"      清理旧 chunk {removed} 个（同一文件反复处理不会堆积）")
        if full:
            # 全量重建时，原件已消失的文件的片段要**留下**，只把标记改成缺失。
            # 不这么做的话，它们会在下面 add_documents 之前凭空消失 ——
            # 结果是元数据表里留着记录、向量库里没有片段，反而退化成「搜不到」
            # 的坏状态，比直接删掉还难查。
            gone = _missing_paths()
            if gone:
                patched = 0
                for path in gone:
                    patched += vectorstore.update_metadata_by_file_path(
                        path, {"file_missing": 1}
                    )
                say(f"      保留原件已消失的 {len(gone)} 份文件的片段，"
                    f"标记为缺失（{patched} 个 chunk，未删除）")
    vectorstore.add_documents(chunks, verbose=verbose)

    total = vectorstore.count()
    say(f"      向量库现有 {total} 条记录")

    return {
        "文档数": len(docs),
        "chunk数": len(chunks),
        "跳过数": skipped,
        "向量库总数": total,
        "全量": full,
        "文件": [doc.file_path for doc in docs],
    }


VALID_STATUSES = ("active", "superseded")


def set_status(
    doc_id: int,
    status: str,
    superseded_by: int | None = None,
) -> dict[str, Any]:
    """废止 / 撤销废止一份文件。

    这个操作要同时落到两个存储：

        MySQL  正本 —— 状态是文档级的业务信息，还要能互相引用，只有关系库能做
        Chroma 副本 —— 检索过滤必须发生在取 top_k 之前，所以每个 chunk 都要带状态

    两边没有分布式事务，所以这里不追求「要么全成功、要么全失败」，而是让每一步
    都是**幂等的「设成某个值」**：中途失败，重跑一次就回到一致状态。

    刻意不动 effective_to：它属于内容侧（侧车预先写明这份文件何时失效），
    而 status 属于状态侧。混在一起，重建时就会出现「文件说的」和「管理员说的」
    互相打架的局面。
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"未知状态 {status!r}，只支持 {VALID_STATUSES}")
    if not _supports_lifecycle():
        raise RuntimeError(
            "document 表缺少生命周期字段，请先执行 python scripts/init_db.py"
        )

    row = db.fetch_one(
        "SELECT id, title, file_path, status FROM document WHERE id = :i",
        {"i": int(doc_id)},
    )
    if not row:
        raise LookupError(f"没有找到 id={doc_id} 的文件")

    file_path = str(row["file_path"] or "")
    today = dt.date.today()

    with db.get_engine().begin() as conn:
        if status == "superseded":
            conn.execute(
                text(
                    "UPDATE document SET status = 'superseded', "
                    "superseded_by = :by, superseded_at = :at WHERE id = :i"
                ),
                {"by": superseded_by, "at": today, "i": int(doc_id)},
            )
        else:
            # 撤销废止要把「被谁替代」一并清掉，否则会留下一个悬空引用：
            # 检索时拿到它，也不知道该往哪指。
            conn.execute(
                text(
                    "UPDATE document SET status = 'active', "
                    "superseded_by = NULL, superseded_at = NULL WHERE id = :i"
                ),
                {"i": int(doc_id)},
            )

    patched = vectorstore.update_metadata_by_file_path(file_path, {"status": status})
    log.info(
        "状态变更 | id=%s | %s -> %s | file=%s | 向量副本=%d 个 chunk",
        doc_id, row.get("status"), status, file_path, patched,
    )
    return {
        "id": int(doc_id),
        "title": row["title"],
        "file_path": file_path,
        "原状态": str(row.get("status") or "active"),
        "新状态": status,
        "向量副本条数": patched,
    }


# ---- 原件缺失：对账与处置 ------------------------------------------------
#
# 这一块处理的是「文件从 data/raw 里消失了」。
#
# 处置动作刻意设计成 **只改标记、不删任何东西**：
#   · 元数据表里的行   → 保留，只把 file_missing_at 写上日期
#   · 向量库里的片段   → 保留，只把 file_missing 改成 1
#
# 为什么不物理删除？三条理由，一条比一条实际：
#   1. 「原件被删了」不等于「内容作废」—— 学生还要查往年文件；
#   2. 片段里存着正文，本身就是一份可用的备份，原件误删时它是救命的；
#   3. 标记可逆且免费 —— 恢复只是改一个字段，而重算向量要真金白银。
#
# 反过来，**扫描永远只读**。把处置做成自动执行是危险的：data/raw 的路径一旦
# 配错，或者它被换成网盘同步目录（同步中断时整个目录看起来就是「文件全没了」），
# 一次自动执行就能把整个知识库标记成缺失，学生端瞬间什么都搜不到。
# 所以这里只提供「扫描 → 报告 → 人工确认 → 执行」四步，不提供一步到底。


_ORPHAN_SELECT = (
    "SELECT id, title, doc_type, college, publish_date, file_path, status, "
    "superseded_at, " + _MISSING_COLUMN + " "
    "FROM document ORDER BY publish_date DESC, id DESC"
)


def _brief(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "title": row.get("title") or "",
        "doc_type": row.get("doc_type") or "",
        "college": row.get("college") or "",
        "publish_date": str(row["publish_date"]) if row.get("publish_date") else None,
        "file_path": str(row.get("file_path") or ""),
        "废止状态": "已废止" if row.get("superseded_at") else "现行",
        "原因": reason,
    }


def find_orphans(directory: Path | None = None) -> dict[str, Any]:
    """只读对账：列出「磁盘 / 元数据表 / 向量库」三者之间的差异，不改任何东西。

    五个方向，含义和修法各不相同：

        待标记      原件已从磁盘消失，但记录还标着「原件在」
                    → 执行标记，让它退出检索（内容一个字都不删）
        可恢复      原件已回到磁盘，但记录还标着「原件缺失」
                    → 执行恢复，重新可检索
        未登记      磁盘上有，但元数据表里没有
                    → 用管理台的上传功能补录
        向量库多出  向量库里有片段，元数据表里没有对应记录
                    → 上次灌库半途失败留下的，建议重跑全量重建
        向量库缺少  元数据表里有记录，向量库里却没有片段
                    → 这份文件搜不到，重新灌它一次即可

    注意「可恢复」这一项：它是「原件被放回来了」的提示，
    靠 file_missing_at 非空 + 磁盘上文件存在这两个条件同时成立来判断。
    """
    target_dir = Path(directory) if directory else config.DATA_DIR
    if not _supports_lifecycle():
        raise RuntimeError(
            "document 表缺少生命周期字段，无法对账。"
            "请先执行 python scripts/init_db.py"
        )

    disk_paths = {loader.relative_path(p) for p in loader.discover_paths(target_dir)}
    rows = [dict(r) for r in db.fetch_all(_ORPHAN_SELECT)]

    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = str(row.get("file_path") or "")
        if path:
            by_path[path] = row

    pending = [
        _brief(row, "原件已不在磁盘上，但仍标着「原件在」")
        for path, row in by_path.items()
        if not row.get(_MISSING_COLUMN) and path not in disk_paths
    ]
    recoverable = [
        _brief(row, "原件已回到磁盘，但仍标着「原件缺失」")
        for path, row in by_path.items()
        if row.get(_MISSING_COLUMN) and path in disk_paths
    ]
    unregistered = [
        {"file_path": path, "原因": "磁盘上有这个文件，但元数据表里没有记录"}
        for path in sorted(disk_paths - set(by_path))
    ]

    vector_only: list[dict[str, Any]] = []
    vector_lacking: list[dict[str, Any]] = []
    try:
        vector_paths = vectorstore.list_file_paths()
        vector_only = [
            {"file_path": path, "原因": "向量库里有片段，但元数据表里没有记录"}
            for path in sorted(vector_paths - set(by_path))
        ]
        vector_lacking = [
            _brief(row, "元数据表里有记录，但向量库里没有片段（这份文件搜不到）")
            for path, row in by_path.items()
            if path not in vector_paths
        ]
    except Exception as exc:
        log.warning("读取向量库文件清单失败，本次跳过向量库方向的对账：%s", exc)

    summary = {
        "待标记": len(pending),
        "可恢复": len(recoverable),
        "未登记": len(unregistered),
        "向量库多出": len(vector_only),
        "向量库缺少": len(vector_lacking),
    }
    log.info("对账扫描完成 | 源目录=%s | %s", target_dir, summary)
    return {
        "源目录": str(target_dir),
        "统计": summary,
        "待标记": pending,
        "可恢复": recoverable,
        "未登记": unregistered,
        "向量库多出": vector_only,
        "向量库缺少": vector_lacking,
    }


def _rows_by_ids(ids: list[int]) -> list[dict[str, Any]]:
    """按 id 批量取出行。id 是管理端传进来的，所以只读固定几列。

    占位符名是动态拼的，但**值仍然走绑定参数** —— 这一点不能省：
    管理端传来的 id 同样是不可信输入，拼进 SQL 就等于开了注入口子。
    这里不用 bindparam(expanding=True)，是因为 db.fetch_all 内部还会再包一层
    text()，而 text() 只接受字符串。
    """
    if not ids:
        return []
    placeholders = ", ".join(f":id{i}" for i in range(len(ids)))
    params = {f"id{i}": int(value) for i, value in enumerate(ids)}
    return db.fetch_all(
        f"SELECT id, title, file_path FROM document WHERE id IN ({placeholders})",
        params,
    )


def mark_missing(doc_ids: Iterable[int]) -> dict[str, Any]:
    """把一批文件标记为「原件缺失」：退出默认检索，但内容一个字都不删。

    标记要落在两处，和「废止」是同一套路数：

        MySQL  正本 —— 文档级事实，还要参与筛选与统计
        Chroma 副本 —— 检索过滤必须发生在取 top_k 之前，见 app/rag/retriever.py

    两边没有跨库事务，但每一步都是幂等的「设成某个值」，
    中途失败重跑一次就能回到一致状态。
    """
    ids = [int(i) for i in dict.fromkeys(doc_ids)]
    if not ids:
        return {"标记数": 0, "向量副本": 0, "明细": []}
    if not _supports_lifecycle():
        raise RuntimeError(
            "document 表缺少生命周期字段，无法标记。"
            "请先执行 python scripts/init_db.py"
        )

    rows = _rows_by_ids(ids)
    if not rows:
        raise LookupError("没有找到任何匹配的文件 id")

    stmt = text(
        f"UPDATE document SET {_MISSING_COLUMN} = :today WHERE id IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    with db.get_engine().begin() as conn:
        conn.execute(
            stmt, {"today": dt.date.today(), "ids": [r["id"] for r in rows]}
        )

    patched = 0
    for row in rows:
        patched += vectorstore.update_metadata_by_file_path(
            str(row["file_path"] or ""), {"file_missing": 1}
        )

    log.info("标记原件缺失 | %d 份文件 | 向量副本 %d 个 chunk", len(rows), patched)
    return {
        "标记数": len(rows),
        "向量副本": patched,
        "明细": [
            {"id": r["id"], "title": r["title"], "file_path": r["file_path"]}
            for r in rows
        ],
    }


def restore_files(doc_ids: Iterable[int]) -> dict[str, Any]:
    """把「原件已回来」的文件恢复成可检索。

    会**先确认原件真的回到了磁盘上**，不在就跳过。这一步不是多余的谨慎：
    恢复只是把标记清掉，而向量片段一直都在 —— 原件没回来就恢复的话，
    这份文件立刻能被检索到，但学生点开「查看原文」是 404。
    宁可拒绝，也不要造出这种「搜得到、打不开」的状态。
    """
    ids = [int(i) for i in dict.fromkeys(doc_ids)]
    if not ids:
        return {"恢复数": 0, "跳过": [], "向量副本": 0, "明细": []}
    if not _supports_lifecycle():
        raise RuntimeError(
            "document 表缺少生命周期字段，无法恢复。"
            "请先执行 python scripts/init_db.py"
        )

    rows = _rows_by_ids(ids)
    if not rows:
        raise LookupError("没有找到任何匹配的文件 id")

    root = config.BASE_DIR.resolve()
    ready: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        rel = str(row["file_path"] or "").strip().replace("\\", "/")
        candidate = Path(rel)
        target = candidate if candidate.is_absolute() else root / candidate
        if rel and target.is_file():
            ready.append(row)
        else:
            skipped.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "原因": "原件不在磁盘上，先把它放回 data/raw 再恢复",
                }
            )

    if ready:
        stmt = text(
            f"UPDATE document SET {_MISSING_COLUMN} = NULL WHERE id IN :ids"
        ).bindparams(bindparam("ids", expanding=True))
        with db.get_engine().begin() as conn:
            conn.execute(stmt, {"ids": [r["id"] for r in ready]})

    patched = 0
    for row in ready:
        patched += vectorstore.update_metadata_by_file_path(
            str(row["file_path"] or ""), {"file_missing": 0}
        )

    log.info(
        "恢复原件缺失 | 成功 %d 份 | 跳过 %d 份 | 向量副本 %d",
        len(ready), len(skipped), patched,
    )
    return {
        "恢复数": len(ready),
        "跳过": skipped,
        "向量副本": patched,
        "明细": [
            {"id": r["id"], "title": r["title"], "file_path": r["file_path"]}
            for r in ready
        ],
    }
