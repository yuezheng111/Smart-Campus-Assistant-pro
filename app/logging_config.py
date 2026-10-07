"""日志：一次请求留一行，出错留完整堆栈。

为什么用标准库 logging 而不是 loguru？
—— 日志属于基础设施，少一个第三方依赖就少一处版本风险。

落点与策略
----------
- 文件 var/logs/app.log 记全量（含堆栈、耗时、路由、命中数），按天轮转，留 14 天；
- 控制台只输出 WARNING 以上，避免把 uvicorn 的启动信息冲散；
- 第三方库（httpx / chromadb 等）压到 WARNING，否则一条向量查询能刷屏十几行。

这套东西要回答三个问题就够了：
    谁在什么时候问了什么？走了哪条路？哪一步慢 / 错了？
"""

from __future__ import annotations

import logging
import logging.handlers
import sys

from app import config

LOG_FILE = config.LOG_DIR / "app.log"

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_BACKUP_DAYS = 14

_configured = False


def setup_logging(level: int = logging.INFO, console: bool = True) -> None:
    """初始化根 logger。重复调用只会生效一次，可以随便调。"""
    global _configured
    if _configured:
        return

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # delay=True：进程没写日志就不创建文件，避免空文件堆在 var/logs 里
    file_handler = logging.handlers.TimedRotatingFileHandler(
        LOG_FILE,
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
    log.info("日志模块自检：这条应该出现在 %s", LOG_FILE)
    log.warning("warning 级别会同时出现在控制台")
    print(f"日志文件：{LOG_FILE}")
