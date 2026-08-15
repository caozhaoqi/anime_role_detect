#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""API 网关路由表：路径 → 下游服务的**单一事实源**。

取代原先硬编码的 ``ROUTE_TABLE`` 列表与 ``SERVICES`` 字典两份真相源（加服务时改一处漏一处）。

- 路由规则通过 ``src.core.service_registry`` 的 ``register_route`` **声明式注册**，
  由 ``register_default_routes()`` 在 import 时装配；t2i 等独立服务也可在自己的模块里
  调用 ``register_route`` 自注册（见 ``src/services/t2i_service/router.py``），无需回改本文件。
- 下游服务根 URL 由调用方（``app.py`` 的 ``SERVICES`` 注册表）注入，路由表本身不硬编码任何
  ``config.*_URL``。
- 顺序即优先级，首个匹配生效；``match_default`` 必须最后注册。``register_route`` 按名称去重，
  重复注册后者覆盖前者（幂等，支持自注册与默认注册并存）。
"""
from __future__ import annotations

from typing import Dict, Tuple

from src.core.service_registry import registry


def register_default_routes() -> None:
    """装配网关默认路由（非 t2i 的稳态路由）。幂等，可重复调用。"""
    # t2i 服务在 src/services/t2i_service/router.py 自注册；此处保留同名兜底，确保
    # 即使 t2i 模块未被 import，网关仍能把 t2i/ 流量正确转发（去重后仅保留一条）。
    registry.register_route(
        name="t2i", service="t2i", match_prefix=["t2i/"], strip="t2i/",
        template="{base}/api/t2i/{path}",
    )
    registry.register_route(
        name="search", service="search",
        match_prefix=["search/image", "search/build-index", "search/stats"],
        template="{base}/api/{path}",
    )
    registry.register_route(
        name="video", service="multimedia", match_prefix=["video/"], strip="video/",
        template="{base}/video/{path}",
    )
    registry.register_route(
        name="classify-multi-role", service="model", match_exact=["classify/multi-role"],
        template="{base}/api/model/detect-multiple",
    )
    registry.register_route(
        name="classify", service="model", match_prefix=["classify"],
        template="{base}/api/{path}",
    )
    registry.register_route(
        name="model-health", service="model", match_exact=["model"],
        template="{base}/api/health",
    )
    registry.register_route(
        name="model-sub", service="model", match_prefix=["model/"], strip="model/",
        template="{base}/api/model/{path}",
    )
    registry.register_route(
        name="default", service="api", match_default=True,
        template="{base}/api/{path}",
    )


def resolve_route(path: str, service_urls: Dict[str, dict]) -> Tuple[str, str]:
    """根据请求路径解析出 (下游服务名, 目标 URL)。

    Args:
        path: 去掉 ``/api/`` 前缀后的请求路径（与 ``proxy_request`` 的 ``path`` 一致）。
        service_urls: 服务注册表，形如 ``SERVICES``，每项含 ``"url"`` 键。

    Returns:
        ``(service_name, target_url)``
    """
    return registry.resolve_route(path, service_urls)


# import 时装配默认路由（幂等）
register_default_routes()
