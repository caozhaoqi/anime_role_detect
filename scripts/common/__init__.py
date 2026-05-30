#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共工具模块包
提供项目共用的工具函数和类
"""

from .download_utils import (
    DEFAULT_HEADERS,
    USER_AGENTS,
    ALLOWED_EXTENSIONS,
    BLOCKED_DOMAINS,
    setup_logger,
    get_random_user_agent,
    is_valid_image_url,
    get_filename_from_url,
    hash_url,
    compute_file_hash,
    download_image,
    load_urls_from_file,
    load_local_hashes,
    DownloadStats,
    DownloadConfig,
)

from .notification_utils import (
    NotificationConfig,
    FeishuNotifier,
    TelegramNotifier,
    CompositeNotifier,
    NullNotifier,
    ProgressNotifier,
)

from .database_utils import ImageDatabase

__all__ = [
    # download_utils
    "DEFAULT_HEADERS",
    "USER_AGENTS",
    "ALLOWED_EXTENSIONS",
    "BLOCKED_DOMAINS",
    "setup_logger",
    "get_random_user_agent",
    "is_valid_image_url",
    "get_filename_from_url",
    "hash_url",
    "compute_file_hash",
    "download_image",
    "load_urls_from_file",
    "load_local_hashes",
    "DownloadStats",
    "DownloadConfig",
    # notification_utils
    "NotificationConfig",
    "FeishuNotifier",
    "TelegramNotifier",
    "CompositeNotifier",
    "NullNotifier",
    "ProgressNotifier",
    # database_utils
    "ImageDatabase",
]
