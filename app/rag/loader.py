"""文档加载：把 data/raw 下的学校文件读成「正文 + 元数据」。

支持格式：.md / .txt（带 YAML front-matter）、.pdf、.docx。

设计要点
--------
RAG 的检索质量，一半取决于切分，另一半取决于**元数据**。
所以这里不只返回纯文本，而是把 title / department / doc_type / college /
publish_date / year 一起抽出来 —— 后面检索时可以先用元数据做硬过滤，
再用向量做语义排序，这一招能显著减少「答非所问」。
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from app import config
from app.logging_config import get_logger

log = get_logger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx"}

# 侧车元数据文件后缀：22_通知.pdf 旁边的 22_通知.pdf.meta.yaml
# 存在的理由见 _load_sidecar 的注释。
SIDECAR_SUFFIX = ".meta.yaml"

# 从文件名里提取日期的兜底规则，例如 05_考试_2024-03-18_xxx.md
_DATE_IN_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def relative_path(path: Path) -> str:
    """统一存「相对项目根」的 POSIX 路径。

    只存文件名是不够的：库里拿到 `xx.md` 之后没人知道它在哪个目录，
    前端「打开原文」就无从下手。存相对路径，谁都能照着定位。
    """
    try:
        rel = path.resolve().relative_to(config.BASE_DIR.resolve())
    except ValueError:
        # 项目目录之外的文件（比如用户从别处挂进来的），存绝对路径
        return path.resolve().as_posix()
    return rel.as_posix()


@dataclass
class LoadedDoc:
    """一份学校文件：正文 + 元数据。

    这里只承载**内容侧**信息（「这份文件是什么」）。至于
    「这份文件现在还有效吗」（status / superseded_by），不属于文件本身，
    由数据库单独管 —— 见 app/ingest.py 里「状态侧回填」那一段。
    """

    file_path: str
    content: str
    title: str = ""
    doc_no: str = ""
    department: str = ""
    doc_type: str = "其他"
    college: str = "全校"
    publish_date: dt.date | None = None
    year: int | None = None
    url: str = ""
    # 生命周期 · 内容侧：描述文件本身，一次确定后不再变
    doc_family: str = ""
    version: int | None = None
    effective_from: dt.date | None = None
    effective_to: dt.date | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """转成 Chroma 能接受的元数据（只允许 str / int / float / bool）。

        status 这里给的是一份**占位默认值**（都当现行），真正生效的值由
        ingest 从数据库读出来后覆盖 —— 因为「现行还是废止」是运行时会变的
        状态，文件里读不出来。检索时要靠这个字段做硬过滤，所以每个 chunk
        都必须带它，不能缺。

        file_missing 同理，而且**必须以 0 显式写进去**：它是「原件还在不在
        data/raw」的标记。新写入的 chunk 原件当然在，所以恒为 0；等到有人
        把原件从磁盘上删掉，ingest 才会把它的副本改成 1。这里不能省略 ——
        Chroma 的 where 对「字段不存在」的匹配行为不明确，缺字段的 chunk
        会在 `file_missing = 0` 这次过滤里被漏掉，表现为「文件凭空搜不到」。
        """
        return {
            "file_path": self.file_path,
            "title": self.title,
            "doc_no": self.doc_no,
            "department": self.department,
            "doc_type": self.doc_type,
            "college": self.college,
            "publish_date": self.publish_date.isoformat() if self.publish_date else "",
            "year": int(self.year or 0),
            "url": self.url,
            "doc_family": self.doc_family,
            "version": int(self.version or 0),
            "status": "active",
            "file_missing": 0,
        }


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """拆出 YAML front-matter。没有头部就返回空字典 + 原文。"""
    if not text.lstrip().startswith("---"):
        return {}, text
    stripped = text.lstrip()
    end = stripped.find("\n---", 3)
    if end == -1:
        return {}, text
    header = stripped[3:end]
    body = stripped[end + 4:]
    try:
        meta = yaml.safe_load(header) or {}
    except yaml.YAMLError:
        return {}, text
    return (meta if isinstance(meta, dict) else {}), body


def _parse_date(value: Any) -> dt.date | None:
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日"):
            try:
                return dt.datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
    return None


def _parse_int(value: Any) -> int | None:
    """侧车里的 version 可能是 3、'3' 或 '第3版' 这类写法，能救就救，救不了返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group())
    return None


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def _infer_from_filename(path: Path) -> dict[str, Any]:
    """没有 front-matter 时，退化为从文件名猜元数据。"""
    meta: dict[str, Any] = {}
    match = _DATE_IN_NAME.search(path.name)
    if match:
        try:
            meta["publish_date"] = dt.date(*(int(g) for g in match.groups()))
        except ValueError:
            pass
    return meta


def sidecar_path(path: Path) -> Path:
    """侧车文件的位置：在同名文件后面追加 .meta.yaml。"""
    return path.with_name(path.name + SIDECAR_SUFFIX)


def _load_sidecar(path: Path) -> dict[str, Any]:
    """读同名侧车元数据文件。

    为什么需要它
    ------------
    PDF / docx 是二进制，塞不进 YAML front-matter，而 doc_type / college
    这些字段是检索时做「硬过滤」的依据，缺了就等于这个文件按学院/类型
    筛选时永远失配。把元数据放在文件**外面**的同名 yaml 里，还有个额外好处：
    「文件 + 侧车」自成一个完整的资料单元 —— 数据库清空重灌时，元数据跟着
    文件一起回来，不会因为「只存在数据库里」而丢失。
    """
    path = sidecar_path(path)
    if not path.is_file():
        return {}
    try:
        meta = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("侧车元数据读取失败 %s：%s", path.name, exc)
        return {}
    if not isinstance(meta, dict):
        log.warning("侧车元数据格式不对（应为键值对）%s", path.name)
        return {}
    return meta


def write_sidecar(path: Path, meta: dict[str, Any]) -> Path:
    """把元数据写成侧车文件。

    空值会被过滤掉：留下一堆 `department: ''` 只会让 yaml 变吵，
    而 loader 那边本来就有「空值取默认值」的兜底。
    """
    clean = {k: v for k, v in meta.items() if v not in (None, "", 0)}
    target = sidecar_path(path)
    target.write_text(
        yaml.safe_dump(clean, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target


def load_one(path: Path) -> LoadedDoc | None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return None

    if suffix == ".pdf":
        raw = _read_pdf(path)
    elif suffix == ".docx":
        raw = _read_docx(path)
    else:
        raw = path.read_text(encoding="utf-8", errors="ignore")

    front_meta, body = _split_front_matter(raw)

    # 元数据优先级：文件自述（front-matter）> 侧车 yaml > 文件名里的日期 > 默认值。
    # 用「合并」而不是「二选一」，是为了让两者互补 —— 比如 md 的头部只写了标题，
    # 学院信息仍然可以从侧车里补上。同名字段以文件自述为准。
    meta = {**_load_sidecar(path), **front_meta}
    if not meta:
        meta = _infer_from_filename(path)

    publish_date = _parse_date(meta.get("publish_date"))
    doc = LoadedDoc(
        file_path=relative_path(path),
        content=body.strip(),
        title=str(meta.get("title") or path.stem),
        doc_no=str(meta.get("doc_no") or ""),
        department=str(meta.get("department") or ""),
        doc_type=str(meta.get("doc_type") or "其他"),
        college=str(meta.get("college") or "全校"),
        publish_date=publish_date,
        year=publish_date.year if publish_date else None,
        url=str(meta.get("url") or ""),
        doc_family=str(meta.get("doc_family") or ""),
        version=_parse_int(meta.get("version")),
        effective_from=_parse_date(meta.get("effective_from")),
        effective_to=_parse_date(meta.get("effective_to")),
    )
    return doc


def discover_paths(directory: Path) -> list[Path]:
    """列出目录下受支持的文档路径（只列出，不解析）。

    单独抽出来是为了让「全量重建」和「增量上传」共用同一套文件发现规则 ——
    两边的判断标准必须一致，否则迟早出现「命令行看不到、界面上看得到」
    这种最难排查的错位。
    """
    if not directory.exists():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def load_paths(paths: Iterable[Path]) -> list[LoadedDoc]:
    """解析指定的这批文件。空内容的会被跳过，但会留下日志。"""
    docs: list[LoadedDoc] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            doc = load_one(path)
        except Exception as exc:  # 单份文件坏掉不该拖垮整批
            log.exception("解析失败，已跳过 %s：%s", path.name, exc)
            continue
        if doc is None:
            continue
        if not doc.content:
            # 以前这里是静默丢弃，结果是「明明放进去了却搜不到」。
            # 最常见的原因是扫描版 PDF（图片型），pypdf 抽不出文字。
            log.warning("内容为空，已跳过（扫描件需先做 OCR）：%s", path.name)
            continue
        docs.append(doc)
    return docs


def load_all(directory: Path) -> list[LoadedDoc]:
    return load_paths(discover_paths(directory))


def make_summary(doc: LoadedDoc, limit: int = 140) -> str:
    """一句话摘要。

    这里刻意用「正文首段截断」而不是再调一次 LLM：
    灌库阶段调 20 次 LLM 既慢又贵，而元数据表里的 summary 只用于列表展示，
    真正的语义理解交给向量检索就够了。
    """
    text = re.sub(r"\s+", " ", doc.content).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


if __name__ == "__main__":
    from app import config

    loaded = load_all(config.DATA_DIR)
    print(f"共加载 {len(loaded)} 份文档\n")
    for d in loaded[:3]:
        print(f"- {d.title}")
        print(f"  部门={d.department} 类型={d.doc_type} 学院={d.college} 日期={d.publish_date}")
        print(f"  摘要={make_summary(d)}\n")
