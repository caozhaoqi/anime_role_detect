#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向后兼容模块 — 已合并到 src.config

请直接从统一配置模块导入:
    from src.config import get_device, configure_device

此文件保留仅为兼容旧代码，将在后续版本移除。
"""

from src.config import get_device, configure_device

__all__ = ["get_device", "configure_device"]
