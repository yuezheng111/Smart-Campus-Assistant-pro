"""文本切分。

中文切分和英文不一样：不能按空格切。这里按「段落 → 句号 → 分号 → 逗号」
逐级退化，尽量保证一个 chunk 是一段语义完整的话。

另外一个容易被忽略的点：**给每个 chunk 补上标题**。
检索时如果只拿到「二、申请条件 1. 具有中华人民共和国国籍……」这样一段，
向量模型很难知道这是哪份文件的。补上标题后召回准确率明显提升。
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.rag.loader import LoadedDoc

# 中文友好的分隔符，按优先级从高到低
CHINESE_SEPARATORS = [
    "\n\n",   # 段落
    "\n",     # 换行
    "。",      # 句号
    "；",      # 分号
    "！",      # 感叹号
    "？",      # 问号
    "，",      # 逗号
    "、",      # 顿号
    " ",
    "",
]


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=CHINESE_SEPARATORS,
        length_function=len,
        keep_separator=True,
    )


def split_documents(docs: list[LoadedDoc]) -> list[Document]:
    splitter = get_splitter()
    chunks: list[Document] = []

    for doc in docs:
        base_meta = doc.to_metadata()
        pieces = splitter.split_text(doc.content)
        for index, piece in enumerate(pieces):
            text = piece.strip()
            if len(text) < 20:  # 太短的碎片（如页码、空行）直接丢弃
                continue
            metadata = {**base_meta, "chunk_index": index, "chunk_total": len(pieces)}
            chunks.append(
                Document(
                    page_content=f"【{doc.title}】\n{text}",
                    metadata=metadata,
                )
            )

    return chunks


if __name__ == "__main__":
    from app.rag.loader import load_all

    docs = load_all(config.DATA_DIR)
    chunks = split_documents(docs)
    print(f"{len(docs)} 份文档 -> {len(chunks)} 个 chunk")
    print(f"平均每份 {len(chunks) / max(len(docs), 1):.1f} 个 chunk")
    print("\n--- 第一个 chunk 示例 ---")
    print(chunks[0].page_content[:300])
    print("\n--- 元数据 ---")
    print(chunks[0].metadata)
