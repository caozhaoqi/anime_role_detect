#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存服务
负责管理缓存相关操作
"""

import threading
from src.core.logging.global_logger import get_logger
from .cache_factory import get_cache_factory

logger = get_logger("cache_service")

# 延迟初始化缓存工厂
cache_factory = None

# 添加线程锁，用于保护缓存工厂的初始化过程
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
    global cache_factory
    with cache_init_lock:
        if cache_factory is None:
            # 初始化缓存工厂
            cache_factory = get_cache_factory()
            logger.info("缓存工厂初始化成功")


def get_cache_manager():
    """
    获取缓存管理器实例
    
    Returns:
        缓存管理器实例
    """
    global cache_factory
    if cache_factory is None:
        init_cache_manager()
    return cache_factory


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


def get_cache_stats():
    """
    获取缓存统计信息
    
    Returns:
        缓存统计信息
    """
    cache_factory = get_cache_manager()
    return cache_factory.get_stats()
