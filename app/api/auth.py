"""管理端鉴权。

两条规则，按部署形态自动切换：

    ADMIN_TOKEN 没配  → 只允许本机回环（127.0.0.1 / ::1）访问，其他来源一律 403
    ADMIN_TOKEN 配了  → 必须带 X-Admin-Token 请求头，并用常量时间比较

为什么默认是「本机回环」而不是直接拒绝：管理台目前就是本机自用，
一上来强制令牌会让人打不开页面，于是又去注释掉鉴权 —— 那才是真的不安全。
默认只开回环，等于把「公网 8000 端口裸奔」这个最危险的状态先关掉；
要对外开放时配一个 ADMIN_TOKEN 即可，不需要改代码。
"""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _configured_token() -> str:
    return os.getenv("ADMIN_TOKEN", "").strip()


def require_admin(request: Request) -> None:
    token = _configured_token()
    client_host = getattr(request.client, "host", "") if request.client else ""

    if not token:
        if client_host not in _LOOPBACK:
            raise HTTPException(
                status_code=403,
                detail="管理端未配置 ADMIN_TOKEN，仅允许本机访问。"
                       "如需远程访问，请设置环境变量 ADMIN_TOKEN 并带上 X-Admin-Token 请求头。",
            )
        return

    provided = (
        request.headers.get("X-Admin-Token")
        or request.query_params.get("token")
        or ""
    )
    if not hmac.compare_digest(provided, token):
        raise HTTPException(status_code=401, detail="管理端令牌无效")


def is_admin_configured() -> bool:
    return bool(_configured_token())
