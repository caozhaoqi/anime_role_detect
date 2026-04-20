#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存服务
负责管理缓存相关操作
"""

import threading
from src.core.logging.global_logger import get_logger

logger = get_logger("cache_service")

# 延迟初始化缓存管理器
cache_manager = None
CacheManager = None

# 添加线程锁，用于保护缓存管理器的初始化过程
cache_init_lock = threading.Lock()

# 模型缓存字典，支持版本管理
model_cache = {}

# 模型版本信息
model_versions = {}

# 图像变换缓存
transform_cache = {}


def init_cache_manager():
    """
    初始化缓存管理器
    """
    global cache_manager, CacheManager
    with cache_init_lock:
        if cache_manager is None:
            # 延迟导入CacheManager
            if CacheManager is None:
                from src.utils.cache_manager import CacheManager
            # 优化缓存配置：减少内存缓存大小，缩短过期时间，避免内存占用过高
            cache_manager = CacheManager(
                max_memory_size=100,  # 减少内存缓存大小
                max_file_size=500,    # 减少文件缓存大小
                default_ttl=1800      # 缩短默认过期时间到30分钟
            )
            logger.info("缓存管理器初始化成功")


def get_cache_manager():
    """
    获取缓存管理器实例
    
    Returns:
        缓存管理器实例
    """
    global cache_manager
    if cache_manager is None:
        init_cache_manager()
    return cache_manager


def get_image_transform():
    """
    获取图像变换对象（缓存）
    
    Returns:
        图像变换对象
    """
    # 延迟导入PyTorch和相关库
    import torchvision.transforms as transforms
    
    global transform_cache
    transform_key = "default"
    
    if transform_key not in transform_cache:
        transform_cache[transform_key] = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform_cache[transform_key]
