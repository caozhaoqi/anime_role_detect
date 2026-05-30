#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片缓存助手
用于生成图片哈希和缓存管理
"""

import hashlib
from typing import Optional, Tuple
from src.core.logging.global_logger import get_logger

logger = get_logger("cache_helper")


def generate_image_hash(content: bytes, model_name: str = "default") -> str:
    """
    生成图片内容的哈希值

    Args:
        content: 图片二进制内容
        model_name: 使用的模型名称

    Returns:
        str: 哈希值字符串
    """
    hash_obj = hashlib.sha256()
    hash_obj.update(content)
    hash_obj.update(model_name.encode("utf-8"))
    return hash_obj.hexdigest()


def get_cache_key(image_hash: str, prefix: str = "classify") -> str:
    """
    生成缓存键

    Args:
        image_hash: 图片哈希值
        prefix: 缓存键前缀

    Returns:
        str: 缓存键
    """
    return f"{prefix}:{image_hash}"


def get_classify_cache_key(content: bytes, model_name: str, options: dict = None) -> str:
    """
    生成分类结果的缓存键

    Args:
        content: 图片二进制内容
        model_name: 模型名称
        options: 额外选项

    Returns:
        str: 缓存键
    """
    hash_str = generate_image_hash(content, model_name)
    if options:
        options_str = str(sorted(options.items()))
        hash_str = hashlib.sha256((hash_str + options_str).encode("utf-8")).hexdigest()
    return get_cache_key(hash_str, "classify:result")


async def check_cache_result(
    cache_manager, content: bytes, model_name: str, options: dict = None
) -> Tuple[bool, Optional[dict]]:
    """
    检查缓存是否存在

    Args:
        cache_manager: 缓存管理器
        content: 图片内容
        model_name: 模型名称
        options: 额外选项

    Returns:
        Tuple[bool, Optional[dict]]: (是否命中缓存, 缓存结果)
    """
    if cache_manager is None:
        return False, None

    try:
        cache_key = get_classify_cache_key(content, model_name, options)
        cached_result = cache_manager.get(cache_key)

        if cached_result is not None:
            logger.info(f"缓存命中: {cache_key[:20]}...")
            return True, cached_result

        logger.debug(f"缓存未命中: {cache_key[:20]}...")
        return False, None

    except Exception as e:
        logger.error(f"缓存检查失败: {e}")
        return False, None


async def save_to_cache(
    cache_manager,
    content: bytes,
    model_name: str,
    result: dict,
    options: dict = None,
    ttl: int = 3600,
) -> bool:
    """
    保存结果到缓存

    Args:
        cache_manager: 缓存管理器
        content: 图片内容
        model_name: 模型名称
        result: 要缓存的结果
        options: 额外选项
        ttl: 过期时间（秒）

    Returns:
        bool: 是否成功
    """
    if cache_manager is None:
        return False

    try:
        cache_key = get_classify_cache_key(content, model_name, options)
        success = cache_manager.set(cache_key, result, ttl)

        if success:
            logger.info(f"结果已缓存: {cache_key[:20]}..., TTL={ttl}s")
        else:
            logger.warning(f"缓存保存失败: {cache_key[:20]}...")

        return success

    except Exception as e:
        logger.error(f"缓存保存失败: {e}")
        return False
