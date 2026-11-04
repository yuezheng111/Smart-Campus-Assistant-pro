"""对话附件：管理员在聊天框里上传的文件，先落临时区，不直接入库。

为什么附件不直接进 data/raw
--------------------------
聊天里发一份文件，多半是"帮我看看这份怎么写""照这个格式生成一份"，
属于**这一轮对话的上下文**，不是"我要把它放进知识库"。
两者混在一起，会出现"随手发了一份草稿，结果被全校学生搜到"这种事。
真要入库，走管理台的「文件管理」或明确让助手处理，那是另一条链路。

解析结果按会话隔离存放，进程重启后仍然能读（落盘而不是放内存），
但**不参与检索**，只有本轮对话里的工具能读到。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from app import config
from app.logging_config import get_logger
from app.rag import loader

log = get_logger(__name__)

UPLOAD_ROOT = config.VAR_DIR / "uploads"

# 数量与体积上限：管理端一次对话发 5 份、单份 10MB，足够处理通知/表格/说明文档。
# 再大就不是"聊天附件"而是"批量导入"，应该走管理台上传。
MAX_FILES = 5
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024

# 解析后保留的正文上限。超过就截断并在结果里标注 —— 附件是上下文，不是数据库。
MAX_TEXT_CHARS = 200_000

_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def _clean_session(session_id: str | None) -> str:
    raw = str(session_id or "").strip()
    raw = re.sub(r"[^0-9a-zA-Z_-]", "", raw)[:48]
    return raw or "default"


def session_dir(session_id: str | None = None) -> Path:
    path = UPLOAD_ROOT / _clean_session(session_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def supported_suffixes() -> list[str]:
    return sorted(loader.SUPPORTED_SUFFIXES)


def _safe_name(raw: str) -> str:
    name = Path(str(raw or "")).name
    stem, suffix = Path(name).stem, Path(name).suffix
    clean = _UNSAFE.sub("_", stem).strip(" .") or "upload"
    return f"{clean}{suffix.lower()}"


def _meta_path(directory: Path, name: str) -> Path:
    return directory / f"{name}.meta.json"


def _text_path(directory: Path, name: str) -> Path:
    return directory / f"{name}.parsed.txt"


def _unique_name(directory: Path, name: str) -> str:
    """同名文件不覆盖：追加 _2、_3。附件之间互不干扰比"省一个文件名"重要。"""
    if not (directory / name).exists():
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    index = 2
    while (directory / f"{stem}_{index}{suffix}").exists():
        index += 1
    return f"{stem}_{index}{suffix}"


def save_upload(filename: str, payload: bytes, session_id: str | None = None) -> dict[str, Any]:
    """保存并解析一份附件。返回**逐文件**的结果，失败不影响同批其他文件。"""
    directory = session_dir(session_id)
    raw_name = str(filename or "")
    suffix = Path(raw_name).suffix.lower()

    if not payload:
        return {"name": raw_name, "ok": False, "error": "文件是空的"}
    if suffix not in loader.SUPPORTED_SUFFIXES:
        return {
            "name": raw_name,
            "ok": False,
            "error": f"不支持 {suffix or '无后缀'}，目前只支持 "
                     + " / ".join(supported_suffixes()),
        }
    if len(payload) > MAX_FILE_BYTES:
        return {
            "name": raw_name,
            "ok": False,
            "error": f"超过单份上限 {MAX_FILE_BYTES // (1024 * 1024)}MB",
        }

    name = _unique_name(directory, _safe_name(raw_name))
    target = directory / name
    target.write_bytes(payload)

    digest = hashlib.sha256(payload).hexdigest()
    try:
        doc = loader.load_one(target)
    except Exception as exc:
        log.warning("附件解析失败 %s：%s", name, exc)
        return {"name": name, "ok": False, "error": f"解析失败：{type(exc).__name__}: {exc}"}

    if doc is None or not doc.content.strip():
        return {
            "name": name,
            "ok": False,
            "error": "没解析出文字（扫描件需要先 OCR，目前不支持）",
        }

    text = doc.content
    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        text = text[:MAX_TEXT_CHARS]
    _text_path(directory, name).write_text(text, encoding="utf-8")

    meta = {
        "name": name,
        "chars": len(text),
        "truncated": truncated,
        "bytes": len(payload),
        "sha256": digest,
        "suffix": suffix,
        "title": doc.title,
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _meta_path(directory, name).write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    log.info("附件已保存 | %s | %d 字符 | %d 字节", name, len(text), len(payload))
    return {"name": name, "ok": True, **meta}


def list_attachments(session_id: str | None = None) -> list[dict[str, Any]]:
    directory = session_dir(session_id)
    items: list[dict[str, Any]] = []
    for meta_file in sorted(directory.glob("*.meta.json")):
        try:
            items.append(json.loads(meta_file.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return items


def read_attachment(
    name: str,
    offset: int = 0,
    max_chars: int = 6000,
    session_id: str | None = None,
) -> dict[str, Any]:
    """按名读取附件的解析正文，支持分段（offset/max_chars）。"""
    directory = session_dir(session_id)
    available = [item["name"] for item in list_attachments(session_id)]
    if not available:
        return {"错误": "当前会话还没有上传任何附件"}

    target = _safe_name(name)
    if target not in available:
        # 允许只给主名（不带后缀）来匹配
        matched = [n for n in available if Path(n).stem == Path(target).stem]
        if len(matched) != 1:
            return {"错误": f"找不到附件 {name}", "可用附件": available}
        target = matched[0]

    try:
        text = _text_path(directory, target).read_text(encoding="utf-8")
    except OSError as exc:
        return {"错误": f"读取附件失败：{exc}"}

    offset = max(0, int(offset))
    max_chars = max(200, min(int(max_chars), 20000))
    chunk = text[offset: offset + max_chars]
    meta = {}
    try:
        meta = json.loads(_meta_path(directory, target).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass

    return {
        "name": target,
        "总字数": len(text),
        "本次范围": f"{offset}-{offset + len(chunk)}",
        "还有更多": offset + len(chunk) < len(text),
        "下次 offset": offset + len(chunk) if offset + len(chunk) < len(text) else None,
        "标题": meta.get("title"),
        "正文": chunk,
    }
