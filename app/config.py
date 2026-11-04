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

# ---------- 入库流水线 ----------
# legacy：老链路（pdf/docx 抽成一个大字符串 → 按 600 字切）
# v2    ：阶段六链路（多格式解析 → 清洗 → 结构感知切分 → 稳定 ID 入库）
#
# 默认 v2。留这个开关不是"两个都要维护"，而是为了**能立刻退回去**：
# 新链路一旦在生产上表现异常，改一个环境变量重启即可恢复，不用回滚代码。
PARSING_PIPELINE = os.getenv("PARSING_PIPELINE", "v2").strip().lower()

# 解析/切分的并发线程数。向量化是公网往返（慢），多线程有意义；
# 但写 Chroma 始终串行（SQLite 单写者），所以调太大只会压满 DashScope 配额。
INGEST_WORKERS = int(os.getenv("INGEST_WORKERS", "4"))
# 写库失败的重试次数（只对"可能是网络抖动"的步骤生效）。
INGEST_RETRIES = int(os.getenv("INGEST_RETRIES", "2"))

# 检索相关度下限。低于它说明「召回的东西大概率对不上问题」。
#
# 注意分数的实际范围是 [-1, 1] 而不是 [0, 1]：Chroma 算的是 1 - cosine 距离，
# 而 cosine 距离上限是 2，所以完全无关的内容会落到负数（实测乱码查询 -0.048）。
#
# 0.2 来自 6 条样本的粗略观察，**不是标定结果**：
#     强相关 0.66–0.74 | 弱相关 0.52 | 知识库外 0.17–0.20 | 乱码 -0.05
# 样本太少，正式标定要用 evals 的负样本集跑分数分布再定这个值。
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.2"))

# ---------- 向量库运行模式 ----------
# embedded：进程内嵌（默认）。python 进程独占 var/chroma，查询走进程内存里的 HNSW 索引。
#           单进程够快，但**索引是启动时的快照**：别的进程写进去的数据，本进程 query 看不到。
#           所以 embedded 模式下只能开 1 个 worker。
# server  ：连独立的 Chroma 服务（`chroma run`）。索引在服务端，客户端只是 HTTP 连接，
#           不占内存、不存快照，因此可以开任意多个 worker，且入库进程写完后
#           web worker 立刻能搜到（不用重启）。多进程部署必须用这个模式。
CHROMA_MODE = os.getenv("CHROMA_MODE", "embedded").strip().lower()
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))

# web 进程数。只影响日志分文件与连接池预算，真正的 worker 数由 uvicorn 启动参数决定。
WEB_WORKERS = int(os.getenv("WEB_WORKERS", "1"))


def chroma_is_server() -> bool:
    return CHROMA_MODE in ("server", "http", "client")


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
    print(f"CHROMA_MODE   = {CHROMA_MODE}" + (
        f"（{CHROMA_HOST}:{CHROMA_PORT}）" if chroma_is_server() else "（进程内嵌，仅单 worker）"))
    print(f"WEB_WORKERS   = {WEB_WORKERS}")
    print(f"MySQL         = {MYSQL_USER}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}")
    print(f"CHAT_MODEL    = {CHAT_MODEL}")
    print(f"EMBEDDING     = {EMBEDDING_MODEL}")
    print(f"缺失的 API Key = {missing_api_keys() or '无'}")
