"""多格式文档解析包。

对外只暴露统一中间结构；具体解析器（PDF/DOCX/PPTX/XLSX）在阶段三加入，
所以这里**不 import 任何解析实现**，避免导入 `app.parsing` 就顺带把
pypdf / python-docx 全加载进来。
"""

from app.parsing.models import (
    BlockType,
    DocumentBlock,
    DocumentMetadata,
    FileType,
    ParseError,
    ParsedDocument,
    ParseStatus,
    SourceLocation,
    TableCell,
    TableData,
    document_id_for,
    new_block_id,
)

__all__ = [
    "BlockType",
    "DocumentBlock",
    "DocumentMetadata",
    "FileType",
    "ParseError",
    "ParsedDocument",
    "ParseStatus",
    "SourceLocation",
    "TableCell",
    "TableData",
    "document_id_for",
    "new_block_id",
]
