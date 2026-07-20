#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向后兼容模块 — 已合并到 src.config

请直接从统一配置模块导入:
    from src.config import SERVICES, SERVICE_GROUPS, get_service_by_name

此文件保留仅为兼容旧代码，将在后续版本移除。
"""

from src.config import (
    SERVICES,
    SERVICE_GROUPS,
    get_service_by_name,
    get_services_by_group,
    list_all_services,
    get_service_port,
)

__all__ = [
    "SERVICES",
    "SERVICE_GROUPS",
    "get_service_by_name",
    "get_services_by_group",
    "list_all_services",
    "get_service_port",
]
