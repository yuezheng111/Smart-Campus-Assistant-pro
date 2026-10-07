"""全局配置：路径、MySQL、LLM / Embedding。

配置来源优先级：进程环境变量 > .env 文件 > 默认值。
另外在 Windows 上做了一次注册表兜底（见 _load_windows_env）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
VAR_DIR = BASE_DIR / "var"
CHROMA_DIR = VAR_DIR / "chroma"
LOG_DIR = VAR_DIR / "logs"

load_dotenv(BASE_DIR / ".env")


def _load_windows_env() -> None:
    """把注册表里的 API Key 补进当前进程。

    Windows 上用「系统属性 → 环境变量」设置的变量，只对之后新启动的进程生效。
    这里主动读一次注册表，省去每次开终端都要手动 set 的麻烦。
    """
    if sys.platform != "win32":
        return
    import winreg

    roots = (
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
        (winreg.HKEY_CURRENT_USER, r"Environment"),
    )
    for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
        if os.environ.get(name):
            continue
        for root, sub in roots:
            try:
                with winreg.OpenKey(root, sub) as key:
                    value, _ = winreg.QueryValueEx(key, name)
            except OSError:
                continue
            if value:
                os.environ[name] = value
                break


_load_windows_env()

# ---------- MySQL ----------
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "school_agent")

# ---------- LLM ----------
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# ---------- RAG ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "5"))
COLLECTION_NAME = "school_documents"


def missing_api_keys() -> list[str]:
    """返回当前缺失的 API Key 名称，用于启动时给出友好提示。"""
    needed = []
    if not os.getenv("DEEPSEEK_API_KEY"):
        needed.append("DEEPSEEK_API_KEY")
    if not os.getenv("DASHSCOPE_API_KEY"):
        needed.append("DASHSCOPE_API_KEY")
    return needed


if __name__ == "__main__":
    print(f"BASE_DIR      = {BASE_DIR}")
    print(f"DATA_DIR      = {DATA_DIR}")
    print(f"CHROMA_DIR    = {CHROMA_DIR}")
    print(f"MySQL         = {MYSQL_USER}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}")
    print(f"CHAT_MODEL    = {CHAT_MODEL}")
    print(f"EMBEDDING     = {EMBEDDING_MODEL}")
    print(f"缺失的 API Key = {missing_api_keys() or '无'}")
