#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anime Role Detect 核心包
提供技能仓库、版本管理、工作流引擎和性能监控功能
"""

from . import store, version, workflow, monitoring, api, cli

__all__ = ['store', 'version', 'workflow', 'monitoring', 'api', 'cli']
__version__ = '1.0.0'