"""启动 web 服务，支持多 worker。

为什么不用 `python -m app.api.main` 或直接 `uvicorn app.api.main:app`？
—— 因为多 worker 有三个前置条件必须成立，任何一个不成立都会导致
「服务能起来，但行为是错的」，而且错得很隐蔽。这个脚本就是来挡这个的。

前置条件
--------
1. **向量库必须是 server 模式**（CHROMA_MODE=server）。
   嵌入式模式下每个 worker 各持一份 HNSW 索引快照，结果是：
   管理员上传成功 → 学生搜不到 → 重启服务又好了。这种 bug 很难查。

2. **Chroma 服务必须已经活着**。
   否则 worker 起来后第一次检索才失败，错误被包在工具返回里，
   看起来像「知识库空了」而不是「连不上向量库」。

3. **WEB_WORKERS 必须等于实际 worker 数**。
   日志要按 pid 分文件、连接池要按 worker 数分配额度，两者都读这个值。
   如果不一致 → 日志抢文件、连接数超 MySQL 上限。

用法
----
    # 单进程（默认，行为和以前完全一样）
    python scripts/serve.py

    # 先起向量库服务（另开一个终端，常驻）
    chroma run --path var/chroma --host 127.0.0.1 --port 8001 --allow-reset

    # 再起 4 个 worker
    CHROMA_MODE=server python scripts/serve.py --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.logging_config import get_logger  # noqa: E402

log = get_logger(__name__)


def chroma_alive(timeout: float = 2.0) -> tuple[bool, str]:
    """探测 Chroma 服务是否活着。不引 chromadb 客户端，避免探测本身很重。"""
    url = f"http://{config.CHROMA_HOST}:{config.CHROMA_PORT}/api/v2/heartbeat"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                return True, ""
            return False, f"HTTP {resp.status}"
    except urllib.error.URLError as exc:
        return False, str(getattr(exc, "reason", exc))
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def check(workers: int) -> list[str]:
    """返回阻断启动的错误列表。空列表表示可以启动。"""
    errors: list[str] = []

    if workers > 1 and not config.chroma_is_server():
        errors.append(
            f"CHROMA_MODE={config.CHROMA_MODE}，但要求 --workers {workers}。\n"
            "   嵌入式向量库下每个 worker 各持一份索引快照，写入互不可见：\n"
            "   管理员上传成功、学生搜不到、重启又好 —— 这种问题不值得去查。\n"
            "   两个选择：\n"
            "     a) 单进程：python scripts/serve.py\n"
            "     b) 起向量库服务后：CHROMA_MODE=server python scripts/serve.py "
            f"--workers {workers}"
        )

    if config.chroma_is_server():
        ok, reason = chroma_alive()
        if not ok:
            errors.append(
                f"连不上向量库 {config.CHROMA_HOST}:{config.CHROMA_PORT}（{reason}）。\n"
                "   先起服务：chroma run --path var/chroma "
                f"--host {config.CHROMA_HOST} --port {config.CHROMA_PORT} --allow-reset"
            )

    missing = config.missing_api_keys()
    if missing:
        # 缺 Key 不阻断启动：静态页面和 /api/documents 仍可用，只有 /api/chat 会失败。
        log.warning("缺少 API Key：%s，/api/chat 会失败", "、".join(missing))

    size, overflow = (10, 5) if workers <= 4 else (
        max(3, 40 // workers), max(1, 10 // workers))
    total = (size + overflow) * workers
    if total > 100:
        log.warning("%d worker × %d 连接 = %d，接近 MySQL max_connections(151)",
                    workers, size + overflow, total)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="启动校园智助 web 服务")
    parser.add_argument("--workers", type=int, default=config.WEB_WORKERS,
                        help="worker 进程数（默认读 WEB_WORKERS，即 1）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="热重载，仅单 worker 可用")
    args = parser.parse_args()

    workers = max(1, args.workers)

    if args.reload and workers > 1:
        print("✗ --reload 与多 worker 互斥（uvicorn 的限制）", file=sys.stderr)
        return 2

    errors = check(workers)
    if errors:
        print("✗ 无法启动：", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    # 关键：必须在启动 uvicorn 之前设置。
    # 每个 worker 是 spawn 出来的独立进程，会重新 import config 并读这个环境变量 ——
    # 日志分文件、连接池额度都依赖它。设晚了子进程就还是旧值。
    os.environ["WEB_WORKERS"] = str(workers)

    import uvicorn

    print(f"学生端：http://{args.host}:{args.port}")
    print(f"管理端：http://{args.host}:{args.port}/admin  （无认证，勿对外暴露）")
    print(f"向量库：{config.CHROMA_MODE}"
          + (f" @ {config.CHROMA_HOST}:{config.CHROMA_PORT}"
             if config.chroma_is_server() else "（嵌入，单 worker）"))
    print(f"worker：{workers}")
    print(f"日志：  {config.LOG_DIR}" + (
        "\\app.p<pid>.log（每进程一个文件）" if workers > 1 else "\\app.log"))

    uvicorn.run(
        "app.api.main:app",  # 多 worker 必须传导入字符串，不能传 app 对象
        host=args.host,
        port=args.port,
        workers=workers,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    # Windows 上 spawn 会重新 import 本模块，没有这个守卫会无限递归
    raise SystemExit(main())
