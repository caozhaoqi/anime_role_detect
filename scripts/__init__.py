#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动漫角色检测项目主模块
提供统一的入口和API接口
"""

import os
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导出公共模块
from .common import (
    # 下载工具
    setup_logger,
    download_image,
    is_valid_image_url,
    load_urls_from_file,
    load_local_hashes,
    DownloadStats,
    DownloadConfig,
    # 通知工具
    ProgressNotifier,
    # 数据库工具
    ImageDatabase
)

__all__ = [
    'setup_logger',
    'download_image',
    'is_valid_image_url',
    'load_urls_from_file',
    'load_local_hashes',
    'DownloadStats',
    'DownloadConfig',
    'ProgressNotifier',
    'ImageDatabase',
    'PROJECT_ROOT'
]
