"""解析器分发：按**真实文件类型**选解析器。

为什么不能只看后缀
------------------
后缀是文件名的一部分，改个名字就能骗过去。`.pdf` 里塞个 docx、或者把扫描件
改名成 `.pdf` 都是真实会发生的操作（用户导出文件时随手改）。所以这里先用
文件头做一次复核：

    %PDF            → PDF
    PK\\x03\\x04 + 包内 word/ ppt/ xl/ → OOXML 三兄弟
    D0CF11E0        → 老式 OLE 复合文档（.doc/.xls/.ppt，本项目不支持，要明确告知）
    其他            → 回落到后缀判断（md/txt 这类纯文本没有 magic）

分发只决定「用哪个解析器」，不负责兜底异常 —— 那是 BaseParser.parse 的事。
"""

from __future__ import annotations

import importlib
import zipfile
from pathlib import Path

from app.logging_config import get_logger
from app.parsing.base import BaseParser, ParseOptions, relative_path
from app.parsing.models import FileType, ParsedDocument, ParseError, ParseStatus, document_id_for

log = get_logger(__name__)

# 解析器注册表：格式 → (模块路径, 类名)
# 用字符串而不是直接 import，是为了让「没装 python-pptx」只影响 pptx，
# 不影响 PDF/DOCX 的解析 —— 导入期就失败的话整个包都没法用。
_PARSER_REGISTRY: dict[FileType, tuple[str, str]] = {
    FileType.PDF: ("app.parsing.pdf_parser", "PDFParser"),
    FileType.DOCX: ("app.parsing.docx_parser", "DOCXParser"),
    FileType.PPTX: ("app.parsing.pptx_parser", "PPTXParser"),
    FileType.XLSX: ("app.parsing.xlsx_parser", "XLSXParser"),
    FileType.MARKDOWN: ("app.parsing.text_parser", "TextParser"),
    FileType.TEXT: ("app.parsing.text_parser", "TextParser"),
}

SUPPORTED_TYPES = frozenset(_PARSER_REGISTRY)

#: 能入库的后缀（含点）。上传接口拿它做白名单 —— 和新链路支持的格式同源，
#: 免得"解析器加了 pptx、上传白名单忘了加"这种两处各写一份的老问题。
SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".pptx", ".xlsx", ".md", ".txt"})

_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


class UnsupportedFileType(Exception):
    """检测到了类型，但没有对应解析器。带上人话解释，别只说「不支持」。"""

    def __init__(self, message: str, file_type: FileType = FileType.UNKNOWN) -> None:
        super().__init__(message)
        self.file_type = file_type


def detect_file_type(path: str | Path) -> FileType:
    """按文件头 + 包结构判定真实格式，后缀只作兜底。"""
    target = Path(path)
    suffix_type = FileType.from_suffix(target.suffix)

    try:
        with target.open("rb") as handle:
            head = handle.read(8192)
    except OSError as exc:
        log.warning("读取文件头失败 %s：%s", target.name, exc)
        return suffix_type

    if head.startswith(b"%PDF"):
        return FileType.PDF

    if head.startswith(_OLE_MAGIC):
        # .doc / .xls / .ppt 的老格式。明确说清楚，别让它掉进"未知格式"里
        # 让人以为是文件坏了 —— 实际是格式太老。
        return FileType.UNKNOWN

    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        try:
            with zipfile.ZipFile(target) as archive:
                names = archive.namelist()
        except zipfile.BadZipFile:
            log.warning("zip 结构损坏，按后缀处理：%s", target.name)
            return suffix_type
        if any(name.startswith("word/") for name in names):
            return FileType.DOCX
        if any(name.startswith("ppt/") for name in names):
            return FileType.PPTX
        if any(name.startswith("xl/") for name in names):
            return FileType.XLSX
        return suffix_type

    # 纯文本没有 magic，只能信后缀
    if suffix_type in (FileType.MARKDOWN, FileType.TEXT):
        return suffix_type

    return suffix_type


def get_parser(file_type: FileType) -> BaseParser:
    """取某个格式的解析器实例。解析器本身无状态，用完即弃。"""
    entry = _PARSER_REGISTRY.get(file_type)
    if entry is None:
        raise UnsupportedFileType(_unsupported_message(file_type), file_type)
    module_path, class_name = entry
    module = importlib.import_module(module_path)
    parser_cls = getattr(module, class_name)
    return parser_cls()


def _unsupported_message(file_type: FileType) -> str:
    supported = " / ".join(sorted(t.value for t in SUPPORTED_TYPES))
    if file_type is FileType.UNKNOWN:
        return (
            f"无法识别的文件格式（可能是 .doc/.xls/.ppt 老格式，或文件已损坏）。"
            f"目前支持：{supported}"
        )
    return f"暂不支持 {file_type.value} 格式，目前支持：{supported}"


def parse_file(
    path: str | Path,
    options: ParseOptions | None = None,
    *,
    file_type: FileType | None = None,
) -> ParsedDocument:
    """解析一份文件，**任何情况都不抛异常**，失败也返回 ParsedDocument。

    这一点很重要：批量入库时最怕的就是「一个坏文件把整批带崩」。
    调用方只需要看 `parse_status`，不需要写 try/except。
    """
    target = Path(path)
    detected = file_type or detect_file_type(target)
    try:
        parser = get_parser(detected)
    except UnsupportedFileType as exc:
        rel = relative_path(target)
        log.warning("不支持的文件类型 | %s | %s", rel, exc)
        return ParsedDocument(
            document_id=document_id_for(rel),
            file_name=target.name,
            file_type=detected,
            title=target.stem,
            metadata={"file_path": rel},
            blocks=[],
            parser_name="none",
            parser_version="0.0.0",
            parse_status=ParseStatus.FAILED,
            errors=[ParseError(stage="detect", message=str(exc), fatal=True)],
        )
    return parser.parse(target, options)
