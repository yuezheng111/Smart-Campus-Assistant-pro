"""解析器统一接口。

所有格式的解析器都实现 `BaseParser`，对外只暴露 `parse()` 一个方法，
返回 `ParsedDocument`。下游（Chunker / Chroma / 检索）只认这个返回值。

模板方法的意义
--------------
`parse()` 在父类里做三件所有解析器都该做、但很容易有人漏做的事：

    1. 算 document_id / 文件大小 / sha256   —— 每个解析器各写一遍必然漂移
    2. **异常兜底**：任何异常都转成 parse_status=FAILED + fatal error，
       而不是往上抛。这样"某一份文件坏了"不会拖垮整批入库
    3. 补齐 title / file_name / file_type / file_path

子类只实现 `_parse()`，专心处理自己那种格式的难点。

注意：兜底不等于吞异常。异常信息会**完整记进 errors 和日志**，
管理端能看到"这份文件为什么没进库"，而不是像现在这样只留一条 WARNING。
"""

from __future__ import annotations

import datetime as dt
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from app import config
from app.logging_config import get_logger
from app.parsing._heuristics import title_from_filename
from app.parsing.models import (
    FileType,
    ParsedDocument,
    ParseError,
    ParseStatus,
    document_id_for,
)

log = get_logger(__name__)


def relative_path(path: Path) -> str:
    """相对项目根的 POSIX 路径。

    与 `app.rag.loader.relative_path` 保持同一口径，但这里**不 import 它**：
    解析器不该依赖 RAG 层，否则将来单独抽出去做解析服务时会被拖住。
    两边都存相对路径，是因为「只存文件名」会让前端打开原文时无从定位。
    """
    try:
        return path.resolve().relative_to(config.BASE_DIR.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    """按原始字节算哈希。

    不按解析后的文本算：同一份 PDF 换个 pypdf 版本抽出来的字可能不同，
    用它算哈希会把"文件没改"误判成"改过了"，白算一遍 embedding。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ParseOptions:
    """解析选项。

    assets_dir 为空时**不落盘图片**，只保留 `embedded://...` 形式的内嵌引用。
    这样跑测试不会往 var/ 里塞一堆 png；真要 OCR 时（阶段四）再传目录。
    """

    assets_dir: Path | None = None
    include_images: bool = True
    max_pages: int | None = 500
    sidecar: dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    """解析器基类。子类必须给出 name / version / file_types 并实现 _parse。"""

    name: str = "base"
    version: str = "0.1.0"
    file_types: tuple[FileType, ...] = ()

    # ---- 子类实现 ----

    @abstractmethod
    def _parse(
        self, path: Path, options: ParseOptions, document_id: str
    ) -> ParsedDocument:
        """真正干活的地方。抛异常是允许的，父类会转成 FAILED。"""

    # ---- 对外入口 ----

    def parse(
        self, path: str | Path, options: ParseOptions | None = None
    ) -> ParsedDocument:
        target = Path(path)
        opts = options or ParseOptions()
        rel = relative_path(target)
        document_id = document_id_for(rel)

        if not target.is_file():
            return self._failure(document_id, target, f"文件不存在：{target}")
        if target.stat().st_size == 0:
            return self._failure(document_id, target, "文件是空的（0 字节）")

        try:
            doc = self._parse(target, opts, document_id)
        except Exception as exc:  # 单份文件坏掉不该拖垮整批
            log.exception("解析失败 | %s | parser=%s", rel, self.name)
            return self._failure(
                document_id, target, f"{type(exc).__name__}: {exc}", rel=rel
            )

        self._finalize(doc, target, rel, document_id, opts)
        if doc.parse_status is ParseStatus.FAILED:
            log.warning("解析失败 | %s | %s", rel, doc.errors[0].message if doc.errors else "")
        elif doc.parse_status in (ParseStatus.EMPTY, ParseStatus.PARTIAL):
            log.warning(
                "解析不完整 | %s | 状态=%s 块数=%d 错误=%d",
                rel, doc.parse_status.value, len(doc.blocks), len(doc.errors),
            )
        return doc

    # ---- 内部 ----

    def _failure(
        self,
        document_id: str,
        path: Path,
        message: str,
        *,
        rel: str | None = None,
        stage: str = "parse",
    ) -> ParsedDocument:
        return ParsedDocument(
            document_id=document_id,
            file_name=path.name,
            file_type=self.file_types[0] if self.file_types else FileType.UNKNOWN,
            title=title_from_filename(path.stem),
            metadata={"file_path": rel or relative_path(path)},
            blocks=[],
            parser_name=self.name,
            parser_version=self.version,
            parse_status=ParseStatus.FAILED,
            errors=[ParseError(stage=stage, message=message, fatal=True)],
        )

    def _finalize(
        self,
        doc: ParsedDocument,
        path: Path,
        rel: str,
        document_id: str,
        options: ParseOptions,
    ) -> None:
        """补齐解析器不必重复填的公共字段。

        刻意用赋值而不是重建对象：重建会触发一次完整校验，把解析器
        特意留下的 PARTIAL + 空 blocks 之类组合重新判一遍，意义不大且更慢。
        """
        doc.document_id = document_id
        doc.file_name = path.name
        # 解析器没找到标题时会拿 path.stem 顶上（`title=title or path.stem`），
        # 所以这里除了"空标题"还要接住"标题就是文件名"的情况，否则归档前缀
        # `38_助学金_2026-09-22_` 会跟着每个 chunk 走一路。
        # 先留一份解析器给的原始标题：下面判断"要不要用侧车标题"时，
        # 需要知道的是**解析器读没读到标题**，而不是兜底之后的那个值。
        parsed_title = doc.title
        if not doc.title.strip() or doc.title == path.stem:
            doc.title = title_from_filename(path.stem)
        if doc.file_type is FileType.UNKNOWN and self.file_types:
            doc.file_type = self.file_types[0]

        meta = doc.metadata
        if not meta.file_path:
            meta.file_path = rel
        if meta.file_size is None:
            try:
                meta.file_size = path.stat().st_size
            except OSError:
                meta.file_size = None
        if meta.sha256 is None:
            try:
                meta.sha256 = file_sha256(path)
            except OSError as exc:
                log.warning("计算文件哈希失败 %s：%s", path.name, exc)
        if not meta.mime_type:
            meta.mime_type = MIME_BY_SUFFIX.get(path.suffix.lower(), "")

        # 侧车元数据只补「解析器没填」的字段，不覆盖解析结果。
        # 理由：侧车里写的是人给的档案信息，解析器读到的是文件事实，
        # 两者冲突时应该保留文件事实，同时不让档案字段缺着。
        #
        # 「解析器填没填」**不能看值是否为空**。档案字段大多有非空默认值
        # （college 默认「全校」、doc_type 默认「其他」），按空值判断的话
        # 侧车永远写不进去 —— 管理员在上传表单里选的学院会被静默丢掉，
        # 而且丢得毫无痕迹，看起来像是表单没生效。
        # 正确判据是 pydantic 的 `model_fields_set`：只有解析器显式赋过值的
        # 字段才在集合里，没赋值的（即仍是默认值）允许侧车来填。
        sidecar = options.sidecar or {}
        explicit = set(getattr(meta, "model_fields_set", None) or ())
        for key, value in sidecar.items():
            if value in (None, ""):
                continue
            if key == "title":
                # title 挂在 doc 上而不是 metadata 上，得单独接。
                # 只在「解析器没从文件里读出标题」时才用侧车的：
                # md 的 front-matter 标题、PDF 的字号推断标题都算文件事实，
                # 优先级高于侧车（沿用老 loader 的 front-matter > 侧车）。
                if not parsed_title.strip() or parsed_title == path.stem:
                    doc.title = str(value).strip()
                continue
            if key not in type(meta).model_fields or key in explicit:
                continue
            try:
                # 先按字段类型校验再赋值。裸 `setattr` 会**绕过** pydantic 的验证，
                # 于是侧车里写的 `publish_date: 2025-12-15` 会以字符串形式存进
                # 本该是 date 的字段：后面任何一处 `model_dump()` 都会报
                # PydanticSerializationUnexpectedValue（实测 40 份带侧车的文件
                # 报了 37 次），而直接调 `.isoformat()` 的地方会当场崩。
                annotation = type(meta).model_fields[key].annotation
                setattr(meta, key, TypeAdapter(annotation).validate_python(value))
            except Exception:  # 类型对不上就跳过，不因一条元数据弄挂整份解析
                log.warning("侧车字段 %s=%r 与模型类型不匹配，已忽略", key, value)

        if doc.parsed_at.tzinfo is None:
            doc.parsed_at = doc.parsed_at.replace(tzinfo=dt.timezone.utc)

    # ---- 图片/附件落盘 ----

    def save_asset(
        self,
        blob: bytes,
        filename: str,
        options: ParseOptions,
        document_id: str,
    ) -> str:
        """把内嵌资源写到 `assets_dir/<document_id>/`，返回相对项目根的路径。

        按 document_id 分目录：一份文件 30 张图全平铺在一个目录里，
        既没法清理也没法对应回原文。同名文件追加 _2、_3，不覆盖。
        """
        if options.assets_dir is None:
            raise ValueError("assets_dir 未配置，调用方应改用 embedded:// 引用")
        target_dir = Path(options.assets_dir) / document_id
        target_dir.mkdir(parents=True, exist_ok=True)
        name = Path(filename).name or "asset.bin"
        target = target_dir / name
        index = 2
        while target.exists():
            target = target_dir / f"{Path(name).stem}_{index}{Path(name).suffix}"
            index += 1
        target.write_bytes(blob)
        return relative_path(target)


MIME_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".md": "text/markdown",
    ".txt": "text/plain",
}
