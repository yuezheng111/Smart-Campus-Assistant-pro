"""LLM 客户端工厂。

对话用 DeepSeek，Embedding 用 DashScope —— 两个供应商各取所长。
把创建逻辑收在一处，避免每个模块各自 new 一个客户端（连接池浪费、参数不一致）。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_deepseek import ChatDeepSeek

from app import config


@lru_cache(maxsize=4)
def get_llm(temperature: float | None = None, max_tokens: int | None = None) -> ChatDeepSeek:
    """返回一个 DeepSeek 对话模型实例（按 temperature 缓存）。"""
    kwargs: dict = {
        "model": config.CHAT_MODEL,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "timeout": 120,
        "max_retries": 2,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatDeepSeek(**kwargs)


def check_keys() -> list[str]:
    return config.missing_api_keys()
