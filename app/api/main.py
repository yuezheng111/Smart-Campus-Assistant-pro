"""FastAPI 后端。

接口设计原则：
- /api/chat 是唯一「会思考」的接口（走 LangGraph，消耗 LLM）；
- 其余 /api/* 都是纯查询接口（只读 MySQL / 向量库），前端切换标签页时用，
  不消耗 LLM，响应快。
这样设计的好处：用户随便点筛选器不会烧钱，只有真正提问才调用模型。

启动：python -m app.api.main
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, db
from app.agent.graph import run_agent
from app.ingest import (
    file_sha256,
    find_orphans,
    ingest_paths,
    mark_missing,
    restore_files,
    set_status,
)
from app.logging_config import get_logger
from app.rag import loader
from app.rag.retriever import search_documents
from app.tools import (
    get_document_detail,
    get_student_profile,
    list_school_documents,
    query_competition,
    query_exam,
    recommend_competition,
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

log = get_logger(__name__)

app = FastAPI(title="校园智能助手 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    """给每个业务请求记一行：方法、路径、状态码、耗时。

    只记 /api 与 /health。静态资源（css / js / 图片）的请求不记 ——
    否则打开一次页面就刷出几十行噪音，真正有用的信息反而被埋掉。
    """
    path = request.url.path
    if not (path.startswith("/api") or path == "/health"):
        return await call_next(request)

    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("请求异常 %s %s", request.method, path)
        raise
    cost_ms = (time.perf_counter() - started) * 1000
    log.info("HTTP %s %s -> %s (%.0fms)", request.method, path, response.status_code, cost_ms)
    return response


# ----------------------------------------------------------------------
# 请求模型
# ----------------------------------------------------------------------


class Turn(BaseModel):
    role: str = Field(description="user 或 assistant")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="学生的问题")
    history: list[Turn] = Field(default_factory=list, description="最近几轮对话（可选）")


class RecommendRequest(BaseModel):
    grade: str = Field(min_length=1, description="年级，如「大二」")
    major: str = ""
    interests: str = ""
    limit: int = 6


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------


def _loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"原始结果": raw}


# 单个文件最多返回多少字符（防止误读到超大文件把响应撑爆）
MAX_RAW_CHARS = 40_000


def _split_front_matter(raw: str) -> tuple[str, str]:
    """切开 Markdown 的 YAML 头与正文。

    这些文档的元数据已经进了 MySQL，前端看原文时不需要再显示一遍，
    所以直接把头部剥掉。这里只做切分，不解析 YAML。
    """
    text = raw.lstrip("\ufeff")
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end == -1:
        return "", text
    return text[3:end].strip(), text[end + 4:].lstrip("\n")


# ----------------------------------------------------------------------
# 接口
# ----------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    missing = config.missing_api_keys()
    vector_count = None
    try:
        from app.rag.vectorstore import count

        vector_count = count()
    except Exception:
        pass
    db_ok = True
    try:
        db.fetch_one("SELECT 1 AS ok")
    except Exception:
        db_ok = False

    return {
        "状态": "ok" if (db_ok and not missing) else "degraded",
        "MySQL": "connected" if db_ok else "error",
        "向量库文档块数": vector_count,
        "缺失的APIKey": missing,
        "对话模型": config.CHAT_MODEL,
        "向量模型": config.EMBEDDING_MODEL,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    """Agent 主入口：走完整的 LangGraph 工作流。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    question = req.question.strip()
    started = time.perf_counter()
    try:
        result = run_agent(
            question,
            [turn.model_dump() for turn in req.history][-6:],
        )
    except Exception as exc:
        # 以前这里把堆栈整个丢掉了，只留下「类型: 消息」，出错时根本不知道
        # 是哪一行炸的。现在完整堆栈进日志；返回给前端的仍是简短信息，
        # 免得把内部路径、SQL 之类的细节暴露出去。
        log.exception("Agent 执行失败 | question=%s", question)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    cost_ms = (time.perf_counter() - started) * 1000
    # 这是日志里最有价值的一行：一问一答到底走了哪条路、命中多少、花了多久。
    log.info(
        "chat | route=%s | 来源=%d | 工具调用=%d | %.0fms | q=%s",
        result.get("route", ""),
        len(result.get("sources") or []),
        len(result.get("data_records") or []),
        cost_ms,
        question,
    )

    return {
        "answer": result.get("answer", ""),
        "route": result.get("route", ""),
        "plan": result.get("plan", {}),
        "sources": result.get("sources", []),
        "documents": result.get("doc_documents", []),
        "trace": result.get("trace", []),
        "tool_calls": result.get("data_records", []),
    }


@app.get("/api/search")
def search(
    q: str = Query(min_length=1, description="检索语句"),
    doc_type: str = "",
    college: str = "",
    year: int = 0,
    top_k: int = 5,
    include_superseded: bool = False,
) -> dict[str, Any]:
    """纯 RAG 检索（不经过 LLM，用于前端的「文件检索」标签页）。

    默认只搜现行文件；include_superseded=True 时把已废止的旧版本一并纳入，
    用于「往年政策」这类查询。
    """
    result = search_documents(
        q,
        k=max(1, min(top_k, 20)),
        doc_type=doc_type or None,
        college=college or None,
        year=year or None,
        include_superseded=include_superseded,
    )

    # 向量库里只存了 file_path，没有 MySQL 的自增 id。
    # 这里补一次映射，前端才能用同一个 id 去取文件详情。
    docs = result["documents"]
    try:
        id_map = {
            row["file_path"]: row["id"]
            for row in db.fetch_all("SELECT id, file_path FROM document")
        }
        for doc in docs:
            doc["id"] = id_map.get(doc.get("file_path"))
    except Exception:
        for doc in docs:
            doc.setdefault("id", None)

    return {
        "query": q,
        "fallback": result["fallback"],
        "filter": result["filter_used"],
        "documents": docs,
        "hits": result["hits"],
    }


@app.get("/api/documents")
def documents(
    doc_type: str = "",
    college: str = "",
    year: int = 0,
    department: str = "",
    keyword: str = "",
    limit: int = 30,
    include_superseded: bool = False,
) -> dict[str, Any]:
    """学生端的文件列表：默认只列现行文件。"""
    return _loads(
        list_school_documents.invoke(
            {
                "doc_type": doc_type,
                "college": college,
                "year": year,
                "department": department,
                "keyword": keyword,
                "limit": limit,
                "include_superseded": include_superseded,
            }
        )
    )


@app.get("/api/documents/{doc_id}")
def document_detail(doc_id: int) -> dict[str, Any]:
    return _loads(get_document_detail.invoke({"doc_id": doc_id}))


def _resolve_document_file(doc_id: int) -> tuple[dict[str, Any], Path]:
    """按 id 查出文件记录，并把它安全地解析成本地绝对路径。

    安全：只允许读项目目录内的文件。file_path 虽然来自我们自己的数据库，
    但把「数据库字段当路径用」本身就是危险动作，所以这里仍然做一次
    目录逃逸校验 —— 万一哪天有人往表里塞了 `../../etc/passwd`，也读不出去。
    """
    row = db.fetch_one(
        "SELECT id, title, doc_no, department, doc_type, college, publish_date, "
        "file_path, url FROM document WHERE id = :doc_id",
        {"doc_id": doc_id},
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"没有找到 id={doc_id} 的文件")

    rel = str(row.get("file_path") or "").strip().replace("\\", "/")
    if not rel:
        raise HTTPException(status_code=404, detail="该文件没有登记本地路径")

    root = config.BASE_DIR.resolve()
    candidate = Path(rel)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()

    if not target.is_relative_to(root):
        raise HTTPException(status_code=403, detail="拒绝访问项目目录以外的文件")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在：{rel}")

    row["file_path"] = rel
    return row, target


def _doc_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_no": row["doc_no"],
        "department": row["department"],
        "doc_type": row["doc_type"],
        "college": row["college"],
        "publish_date": str(row["publish_date"]) if row["publish_date"] else None,
        "url": row["url"],
        "file_path": row["file_path"],
    }


@app.get("/api/documents/{doc_id}/raw")
def document_raw(doc_id: int) -> dict[str, Any]:
    """返回学校文件的原文正文，供前端「查看原文」使用。

    YAML 头会被剥掉：元数据前端已经单独显示，正文里再来一遍是噪音。
    PDF / docx 是二进制，不能当文本直接读（会抛 UnicodeDecodeError），
    所以交给 loader 去解析 —— 和灌库时用的是同一套逻辑，看到的内容一致。
    """
    row, target = _resolve_document_file(doc_id)

    if target.suffix.lower() in {".md", ".txt"}:
        raw = target.read_text(encoding="utf-8", errors="replace")
        _, body = _split_front_matter(raw)
    else:
        loaded = loader.load_one(target)
        body = loaded.content if loaded else ""

    return {
        "id": row["id"],
        "title": row["title"],
        "meta": _doc_meta(row),
        "size": len(body),
        "truncated": len(body) > MAX_RAW_CHARS,
        "content": body[:MAX_RAW_CHARS],
    }


# 下载时的 MIME 类型。之前一律返回 text/markdown，导致下载 PDF 后
# 浏览器把它当文本打开（乱码），而不是保存原件。
_MEDIA_TYPES = {
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@app.get("/api/documents/{doc_id}/download")
def document_download(doc_id: int) -> FileResponse:
    """下载学校文件原件（保留原始扩展名）。"""
    row, target = _resolve_document_file(doc_id)
    filename = f"{row['title']}{target.suffix or '.md'}"
    media = _MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
    return FileResponse(path=target, media_type=media, filename=filename)


@app.get("/api/competitions")
def competitions(
    category: str = "",
    level: str = "",
    year: int = 0,
    grade: str = "",
    keyword: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    return _loads(
        query_competition.invoke(
            {
                "category": category,
                "level": level,
                "year": year,
                "grade": grade,
                "keyword": keyword,
                "limit": limit,
            }
        )
    )


@app.get("/api/exams")
def exams(
    category: str = "",
    year: int = 0,
    grade: str = "",
    keyword: str = "",
    upcoming_only: bool = False,
    limit: int = 50,
) -> dict[str, Any]:
    return _loads(
        query_exam.invoke(
            {
                "category": category,
                "year": year,
                "grade": grade,
                "keyword": keyword,
                "upcoming_only": upcoming_only,
                "limit": limit,
            }
        )
    )


@app.post("/api/recommend")
def recommend(req: RecommendRequest) -> dict[str, Any]:
    return _loads(recommend_competition.invoke(req.model_dump()))


@app.get("/api/students/{student_no}")
def student(student_no: str) -> dict[str, Any]:
    return _loads(get_student_profile.invoke({"student_no": student_no}))


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    def count_of(sql: str) -> int:
        row = db.fetch_one(sql)
        return int(row["c"]) if row else 0

    # 学生端看到的「文件数」只该数**真正可用**的文件：
    #   · 已被新版替代的旧通知 → 不该算，会让学生以为它是可用资料
    #   · 原件已被移出库的留档 → 不该算，学生点「查看原文」会直接 404
    # 这两列都是后加进表的，所以逐级降级：库比代码旧时也不会让统计接口崩掉。
    live_conds: list[str] = []
    try:
        columns = {
            str(r.get("Field") or "")
            for r in db.fetch_all("SHOW COLUMNS FROM `document`")
        }
        if "status" in columns:
            live_conds.append("status = 'active'")
        if "file_missing_at" in columns:
            live_conds.append("file_missing_at IS NULL")
    except Exception:
        live_conds = []

    live = (" WHERE " + " AND ".join(live_conds)) if live_conds else ""
    live_and = (" AND " + " AND ".join(live_conds)) if live_conds else ""

    return {
        "文件数": count_of(f"SELECT COUNT(*) AS c FROM document{live}"),
        "竞赛记录": count_of("SELECT COUNT(*) AS c FROM competition"),
        "考试记录": count_of("SELECT COUNT(*) AS c FROM exam"),
        "竞赛类别": [
            r["category"]
            for r in db.fetch_all(
                "SELECT DISTINCT category FROM competition WHERE category IS NOT NULL ORDER BY category"
            )
        ],
        "文件类型": [
            r["doc_type"]
            for r in db.fetch_all(
                "SELECT DISTINCT doc_type FROM document WHERE doc_type IS NOT NULL"
                + live_and
            )
        ],
    }


# ----------------------------------------------------------------------
# 管理后台：上传学校文件
#
# 面向对象是管理员（1~2 人、低频），不是学生 —— 所以入口做成独立页面，
# 不出现在学生界面的导航里。
#
# ⚠️ 目前没有身份校验。这意味着「能访问到这个服务的人都能往知识库里灌文件」，
#    而知识库投毒是 RAG 系统的典型攻击面（上传一份假通知，Agent 就会照着答）。
#    在补上认证之前，服务只跑 127.0.0.1，不要暴露到局域网或公网。
# ----------------------------------------------------------------------

# Windows 文件名里不允许出现的字符 + 控制字符
_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename(raw: str) -> str:
    """把浏览器传来的文件名清理成安全的文件名。

    浏览器给的文件名不可信：可能夹带路径分隔符（试图往上级目录写），
    也可能带 Windows 不允许的字符。这里只取主干名再过滤。
    后缀由调用方保证合法，原样保留。
    """
    name = Path(raw or "").name  # 去掉任何目录部分
    stem, suffix = Path(name).stem, Path(name).suffix
    clean_stem = _UNSAFE_NAME.sub("_", stem).strip(" .") or "upload"
    return f"{clean_stem}{suffix}"


def _pick_target(name: str, digest: str) -> Path:
    """决定文件最终落在哪里。同名但内容不同的文件绝不静默覆盖。"""
    target = config.DATA_DIR / name
    if not target.exists():
        return target
    try:
        if file_sha256(target) == digest:
            return target  # 内容完全一样，就是同一份，直接覆盖
    except OSError:
        pass
    stem, suffix = target.stem, target.suffix
    index = 2
    while True:
        candidate = config.DATA_DIR / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        try:
            if file_sha256(candidate) == digest:
                return candidate
        except OSError:
            pass
        index += 1


@app.post("/api/admin/upload")
async def admin_upload(
    file: UploadFile = File(..., description="要入库的学校文件"),
    title: str = Form(""),
    doc_type: str = Form(""),
    college: str = Form(""),
    department: str = Form(""),
    doc_no: str = Form(""),
    publish_date: str = Form(""),
    url: str = Form(""),
) -> dict[str, Any]:
    """上传一份文件并立刻入库。

    只处理这一份，不碰库里已有的其他文件 —— 所以是秒级完成，
    也只花这一份的 embedding 钱。
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in loader.SUPPORTED_SUFFIXES:
        supported = " / ".join(sorted(loader.SUPPORTED_SUFFIXES))
        raise HTTPException(
            status_code=400,
            detail=f"不支持 {suffix or '无后缀'} 文件，仅支持：{supported}",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="上传的文件是空的")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    name = _safe_filename(filename)
    target = _pick_target(name, digest)
    target.write_bytes(payload)

    # 元数据写成侧车文件，而不是只写数据库：
    # 这样「文件 + 侧车」是一个自包含的整体，将来全量重建时不依赖数据库状态。
    sidecar = loader.write_sidecar(
        target,
        {
            "title": title.strip(),
            "doc_type": doc_type.strip(),
            "college": college.strip(),
            "department": department.strip(),
            "doc_no": doc_no.strip(),
            "publish_date": publish_date.strip(),
            "url": url.strip(),
        },
    )

    # md / txt 自带 YAML 头时，文件里的值优先于表单 —— 提醒管理员一声，
    # 免得他改了表单却发现不生效。
    note = ""
    if suffix in {".md", ".txt"}:
        head = payload[:400].decode("utf-8", errors="ignore").lstrip("\ufeff").lstrip()
        if head.startswith("---"):
            note = "该文件自带 YAML 头，其中已写的字段优先于表单填写的同名项"

    try:
        result = ingest_paths([target], full=False, verbose=False)
    except Exception as exc:
        log.exception("上传后入库失败 | file=%s", target.name)
        raise HTTPException(
            status_code=500,
            detail=f"文件已保存，但入库失败：{type(exc).__name__}: {exc}",
        ) from exc

    unchanged = int(result.get("跳过数") or 0)
    if unchanged and not result["文档数"]:
        note = (note + "；" if note else "") + "文件内容与档案都没变，已跳过（没有重算向量，也没花钱）"

    log.info(
        "upload | file=%s | 文档=%d chunk=%d 跳过=%d | 向量库=%d | %s",
        loader.relative_path(target),
        result["文档数"],
        result["chunk数"],
        unchanged,
        result["向量库总数"],
        note or "-",
    )

    return {
        # 全部跳过也算成功：文件已经是最新的，本来就没有要做的事
        "ok": result["文档数"] > 0 or unchanged > 0,
        "file_path": loader.relative_path(target),
        "sidecar": loader.relative_path(sidecar),
        "sha256": digest,
        "文档数": result["文档数"],
        "chunk数": result["chunk数"],
        "跳过数": unchanged,
        "向量库总数": result["向量库总数"],
        "note": note,
    }


# ---- 文件状态管理 ----------------------------------------------------
#
# 这一块体现的是「废止 ≠ 删除」：废止只把状态改成 superseded，文件、正文、
# 向量片段都原样保留。所以「去年的政策怎么规定的」仍然答得出来，而撤销废止
# 只是把状态改回去 —— 不用重算 embedding，也就不会花钱。


@app.get("/api/admin/documents")
def admin_documents(
    status: str = "",
    missing: str = "",
    keyword: str = "",
    limit: int = 200,
) -> dict[str, Any]:
    """管理端文件清单，**包含已废止和原件已消失的文件**。

    这是和学生端列表的关键区别：管理员要能看到全貌。一份文件被废止、
    或者原件被移出了 data/raw，它并不会从库里消失，只是退出学生端的检索 ——
    管理端要是也把它们藏起来，管理员就没有地方去撤销、恢复了。

    missing 筛选用字符串而不是布尔：空=全部，'1'=只看原件缺失，'0'=只看原件在。
    """
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if status:
        clauses.append("status = :status")
        params["status"] = status
    if missing == "1":
        clauses.append("file_missing_at IS NOT NULL")
    elif missing == "0":
        clauses.append("file_missing_at IS NULL")
    if keyword:
        clauses.append("(title LIKE :kw OR doc_no LIKE :kw)")
        params["kw"] = f"%{keyword}%"
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params["limit"] = max(1, min(int(limit), 500))

    try:
        rows = db.fetch_all(
            "SELECT id, doc_no, title, department, doc_type, college, publish_date, "
            "year, file_path, status, doc_family, version, superseded_by, "
            "superseded_at, file_missing_at "
            f"FROM document{where} ORDER BY publish_date DESC, id DESC LIMIT :limit",
            params,
        )
        counts = {
            str(r["status"]): int(r["c"])
            for r in db.fetch_all(
                "SELECT status, COUNT(*) AS c FROM document GROUP BY status"
            )
        }
        missing_row = db.fetch_one(
            "SELECT COUNT(*) AS c FROM document WHERE file_missing_at IS NOT NULL"
        )
        missing_count = int(missing_row["c"]) if missing_row else 0
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="读取失败：数据库可能缺少生命周期字段，请先执行 python scripts/init_db.py",
        ) from exc

    return {
        "数量": len(rows),
        "状态统计": counts,
        "原件缺失数": missing_count,
        "文件清单": rows,
    }


def _apply_status(doc_id: int, status: str, superseded_by: int | None) -> dict[str, Any]:
    """把 set_status 抛出的异常翻译成合适的 HTTP 状态码。"""
    try:
        result = set_status(doc_id, status, superseded_by)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("状态变更失败 | id=%s -> %s", doc_id, status)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return {"ok": True, **result}


@app.post("/api/admin/documents/{doc_id}/supersede")
def admin_supersede(doc_id: int, superseded_by: int = 0) -> dict[str, Any]:
    """废止一份文件：默认不再参与检索，但历史依然可查。

    superseded_by 可选。填上「被哪份新文件替代」的 id 之后，
    学生问起时就能顺着这条线告诉他「这份已作废，现行的是哪一份」。
    """
    return _apply_status(doc_id, "superseded", superseded_by or None)


@app.post("/api/admin/documents/{doc_id}/restore")
def admin_restore(doc_id: int) -> dict[str, Any]:
    """撤销废止，恢复为现行。只改一个字段，不重算向量。"""
    return _apply_status(doc_id, "active", None)


# ---- 原件缺失：对账 --------------------------------------------------
#
# 文件从 data/raw 被删掉（或被人手工移走）之后，库里仍然留着它的档案和检索
# 片段 —— 这是**有意为之**，不是遗漏：
#   · 学生可能还要查往年文件，「原件被删了」不等于「内容作废」；
#   · 片段里存着正文，本身就是一份可用的备份，原件误删时它是救命的。
#
# 但不做任何处置的话，它会一直以「现行」的身份混在检索结果里，
# 学生点开「查看原文」却是 404。所以要有对账。
#
# 流程刻意做成四步：扫描 → 报告 → 管理员确认 → 标记。
# 扫描永远只读，处置只改标记、不删内容，随时可以撤销。


class IdsRequest(BaseModel):
    ids: list[int] = Field(default_factory=list, description="要处置的文件 id 列表")


@app.get("/api/admin/orphans")
def admin_orphans() -> dict[str, Any]:
    """对账扫描：列出磁盘 / 元数据表 / 向量库三者之间的差异。**只读**。"""
    try:
        return find_orphans()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("对账扫描失败")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/admin/orphans/mark")
def admin_mark_missing(req: IdsRequest) -> dict[str, Any]:
    """把选中的文件标记为「原件缺失」：退出检索，但内容一个字都不删。"""
    try:
        result = mark_missing(req.ids)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("标记原件缺失失败")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return {"ok": True, **result}


@app.post("/api/admin/orphans/restore")
def admin_restore_missing(req: IdsRequest) -> dict[str, Any]:
    """把原件已放回磁盘的文件恢复成可检索；原件不在的会被跳过并说明原因。"""
    try:
        result = restore_files(req.ids)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("恢复原件缺失失败")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return {"ok": True, **result}


@app.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    """管理员上传页。

    刻意做成独立页面，而不是学生界面的第 6 个标签页：
    入库存档是管理员的活，不应该出现在学生的导航里。
    """
    page = WEB_DIR / "admin.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="管理页面不存在")
    return FileResponse(page, media_type="text/html; charset=utf-8")


# ----------------------------------------------------------------------
# 前端静态资源
# 挂载在最后：Starlette 按注册顺序匹配，这样 /api/* 与 /health 会先命中，
# 剩下的路径才交给静态目录（html=True 让 / 返回 index.html）。
# 用相对路径引用 css/js，所以直接双击 index.html 也能打开。
# ----------------------------------------------------------------------

if not WEB_DIR.is_dir():
    raise RuntimeError(f"前端目录不存在：{WEB_DIR}")

app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    missing = config.missing_api_keys()
    if missing:
        print(f"⚠️  缺少 API Key：{', '.join(missing)}，/api/chat 会失败")
    print("学生端：http://127.0.0.1:8000")
    print("管理端：http://127.0.0.1:8000/admin  （上传文件，无认证，勿对外暴露）")
    print(f"日志：  {config.LOG_DIR / 'app.log'}")
    log.info("服务启动")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
