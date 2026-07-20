#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向后兼容模块 — 已合并到 src.config

请直接从统一配置模块导入:
    from src.config import config, project_config

此文件保留仅为兼容旧代码，将在后续版本移除。
"""

from src.config import config as _config
from src.config import _UnifiedConfig

# 保持旧接口: from src.config.config import config
config = _config

__all__ = ["config"]
