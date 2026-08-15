#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一服务注册表：用声明式 ``register_*(...)`` 取代网关里手写的两份真相源
(``SERVICES`` 字典 + ``ROUTE_TABLE`` 列表)。

设计目标（借鉴 HCM 项目的 ``@open_service`` 装饰器自动发现思路，但做减法，
避免引入元数据引擎复杂度）：

- 各微服务在自己模块里调用 ``register_service`` / ``register_route`` **自注册**，
  网关启动时只需 ``resolve_route(path, service_urls)``，无需再手工同步两份配置。
- 路由按注册顺序匹配，首个命中生效；``match_default`` 必须最后注册。
- 命名去重：同名 ``register_route`` 后者覆盖前者，避免重复注册。
- 纯标准库，可被主 ``.venv`` 与 ``t2i-mac`` venv 同时 import（无第三方依赖）。
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple


class ServiceRegistry:
    def __init__(self) -> None:
        self._services: Dict[str, dict] = {}
        self._routes: List[dict] = []
        self._lock = threading.Lock()

    # ---- 服务注册 ----
    def register_service(
        self,
        name: str,
        url: str,
        prefix: str,
        docs_path: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> None:
        """注册一个下游服务（名称 + 根 URL + 路径前缀）。"""
        with self._lock:
            self._services[name] = {
                "url": url,
                "prefix": prefix,
                "name": display_name or name,
                "docs_path": docs_path,
            }

    # ---- 路由注册（装饰器式自发现）----
    def register_route(
        self,
        *,
        name: str,
        service: str,
        match_prefix: Optional[List[str]] = None,
        match_exact: Optional[List[str]] = None,
        match_default: bool = False,
        strip: Optional[str] = None,
        template: str,
    ) -> None:
        """注册一条路由规则。同名覆盖，保证幂等（重复 import 不重复追加）。

        去重采用**原地替换**而非移除后追加，以保留原注册顺序——
        否则 default 等通用规则会被重排到前面，抢先匹配 t2i/ 等具体前缀。
        """
        with self._lock:
            new_rule = {
                "name": name,
                "service": service,
                "match_prefix": match_prefix,
                "match_exact": match_exact,
                "match_default": match_default,
                "strip": strip,
                "template": template,
            }
            for i, r in enumerate(self._routes):
                if r.get("name") == name:
                    self._routes[i] = new_rule  # 原地替换，保留位置
                    break
            else:
                self._routes.append(new_rule)  # 首次注册，追加末尾

    # ---- 查询 ----
    def get_services(self) -> Dict[str, dict]:
        return dict(self._services)

    def get_routes(self) -> List[dict]:
        return list(self._routes)

    # ---- 路由解析（与历史 resolve_route 语义一致）----
    def resolve_route(self, path: str, service_urls: Dict[str, dict]) -> Tuple[str, str]:
        for rule in self._routes:
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
        return "api", f"{service_urls.get('api', {}).get('url', '')}/api/{path}"


# 进程级单例：各服务模块 import 时向它注册，网关读取它。
registry = ServiceRegistry()
