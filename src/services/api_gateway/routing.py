#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""API 网关路由表：路径 → 下游服务的**单一事实源**。

取代 ``app.py:proxy_request`` 中原先硬编码的 ``if/elif`` 分支，
以及原 ``SERVICES`` 字典里隐含的路由知识 —— 二者曾是双份真相源，
加服务时改一处漏一处就会漂移。

设计要点：
- 本模块为**纯函数**，不依赖任何 Web 框架 / 网络 / 配置，**可直接单测**。
- 下游服务的根 URL 由调用方（``app.py`` 的 ``SERVICES`` 注册表）注入，
  路由表本身不硬编码任何 ``config.*_URL``。
- ``ROUTE_TABLE`` 顺序即优先级，首个匹配生效。
- ``match_exact``：path 精确相等；``match_prefix``：path.startswith；
  ``match_default``：兜底（应放在最后）。
- 命中后按 ``strip`` 去除 path 前缀，再套用 ``template``
  （``{base}``=服务根 URL，``{path}``=处理后的 path）。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

ROUTE_TABLE: List[Dict] = [
    {
        "name": "search",
        "service": "search",
        "match_prefix": ["search/image", "search/build-index", "search/stats"],
        "template": "{base}/api/{path}",
    },
    {
        "name": "video",
        "service": "multimedia",
        "match_prefix": ["video/"],
        "strip": "video/",
        "template": "{base}/video/{path}",
    },
    {
        "name": "classify-multi-role",
        "service": "model",
        "match_exact": ["classify/multi-role"],
        "template": "{base}/api/model/detect-multiple",
    },
    {
        "name": "classify",
        "service": "model",
        "match_prefix": ["classify"],
        "template": "{base}/api/{path}",
    },
    {
        "name": "model-health",
        "service": "model",
        "match_exact": ["model"],
        "template": "{base}/api/health",
    },
    {
        "name": "model-sub",
        "service": "model",
        "match_prefix": ["model/"],
        "strip": "model/",
        "template": "{base}/api/model/{path}",
    },
    {
        "name": "default",
        "service": "api",
        "match_default": True,
        "template": "{base}/api/{path}",
    },
]


def resolve_route(path: str, service_urls: Dict[str, dict]) -> Tuple[str, str]:
    """根据请求路径解析出 (下游服务名, 目标 URL)。

    Args:
        path: 去掉 ``/api/`` 前缀后的请求路径（与 ``proxy_request`` 的 ``path`` 一致）。
        service_urls: 服务注册表，形如 ``SERVICES``，每项含 ``"url"`` 键。

    Returns:
        ``(service_name, target_url)``
    """
    for rule in ROUTE_TABLE:
        matched = False
        if rule.get("match_exact") and path in rule["match_exact"]:
            matched = True
        elif rule.get("match_prefix") and any(
            path.startswith(p) for p in rule["match_prefix"]
        ):
            matched = True
        elif rule.get("match_default"):
            matched = True
        if not matched:
            continue

        base = service_urls[rule["service"]]["url"]
        if rule.get("strip"):
            rest = path[len(rule["strip"]):]
            url = rule["template"].format(base=base, path=rest)
        else:
            url = rule["template"].format(base=base, path=path)
        return rule["service"], url

    # 兜底（理论上 default 规则已覆盖所有情况）
    return "api", f"{service_urls.get('api', {}).get('url', '')}/api/{path}"
