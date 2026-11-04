"""日志：一次请求留一行，出错留完整堆栈。

为什么用标准库 logging 而不是 loguru？
—— 日志属于基础设施，少一个第三方依赖就少一处版本风险。

落点与策略
----------
- 文件 var/logs/app.log 记全量（含堆栈、耗时、路由、命中数），按天轮转，留 14 天；
- 控制台只输出 WARNING 以上，避免把 uvicorn 的启动信息冲散；
- 第三方库（httpx / chromadb 等）压到 WARNING，否则一条向量查询能刷屏十几行。

多进程（WEB_WORKERS > 1）时会改成每进程一个文件 `app.p<pid>.log`
------------------------------------------------------------
原因不是「文件太乱」，是 `TimedRotatingFileHandler` 在 Windows 上多进程会直接报错：
午夜轮转时几个进程同时 rename 同一个文件，而 Windows 不允许改名一个被别人
打开着的文件 —— 结果是 `PermissionError`，且日志就此中断。
多进程同时 write 同一个文件句柄也会让单行内容撕裂（两个进程各写半行）。

每进程一个文件把这两件事都绕开了：没有任何跨进程共享的句柄。
代价是排查要 `grep var/logs/app.p*.log`，这是可接受的。

（为什么不用 QueueHandler + 父进程监听：uvicorn 的 worker 在 Windows 上是
spawn 出来的独立进程，跨进程队列要走 multiprocessing.Queue，worker 被强杀时
容易在写入端死锁。为一个日志文件引入死锁风险，不划算。）

这套东西要回答三个问题就够了：
    谁在什么时候问了什么？走了哪条路？哪一步慢 / 错了？
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from app import config

LOG_FILE = config.LOG_DIR / "app.log"

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"
_FORMAT_MP = "%(asctime)s | %(levelname)-7s | p%(process)-6d | %(name)-20s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_BACKUP_DAYS = 14

_configured = False


def multi_process() -> bool:
    return config.WEB_WORKERS > 1


def current_log_file() -> Path:
    """当前进程实际写入的日志文件。多进程时带 pid。"""
    if multi_process():
        return config.LOG_DIR / f"app.p{os.getpid()}.log"
    return LOG_FILE


def setup_logging(level: int = logging.INFO, console: bool = True) -> None:
    """初始化根 logger。重复调用只会生效一次，可以随便调。"""
    global _configured
    if _configured:
        return

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FORMAT_MP if multi_process() else _FORMAT,
                                  datefmt=_DATEFMT)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # delay=True：进程没写日志就不创建文件，避免空文件堆在 var/logs 里
    file_handler = logging.handlers.TimedRotatingFileHandler(
        current_log_file(),
        when="midnight",
        backupCount=_BACKUP_DAYS,
        encoding="utf-8",
        delay=True,
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)
        root.addHandler(console_handler)

    # 第三方库太吵：一条向量查询能刷十几行，把真正有用的信息埋掉。
    # httpx2 是新版 httpx 用的 logger 名，和 httpx 不是父子关系，得单独列。
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "urllib3",
        "chromadb",
        "openai",
        "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """取一个带名字的 logger。名字用 __name__ 传进来，日志里就能看出模块。"""
    setup_logging()
    return logging.getLogger(name)


if __name__ == "__main__":
    setup_logging()
    log = get_logger(__name__)
    target = current_log_file()
    log.info("日志模块自检：这条应该出现在 %s", target)
    log.warning("warning 级别会同时出现在控制台")
    print(f"日志文件：{target}（多进程={multi_process()}）")
