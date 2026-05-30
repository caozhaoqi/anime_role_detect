#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存工厂
管理不同的缓存实现
"""

import os
from typing import Optional, Any
from src.core.logging.global_logger import get_logger
from src.services.cache_service.redis_cache import get_redis_cache
from src.utils.cache_manager import CacheManager

logger = get_logger("cache_factory")


class CacheFactory:
    """
    缓存工厂
    """

    def __init__(self):
        """
        初始化缓存工厂
        """
        # 获取配置
        self.use_redis = os.environ.get("USE_REDIS", "false").lower() == "true"

        # 初始化缓存实例
        self.redis_cache = get_redis_cache() if self.use_redis else None
        self.local_cache = CacheManager()

        # 缓存策略配置
        self.cache_strategy = os.environ.get("CACHE_STRATEGY", "local_first")

        logger.info(f"缓存工厂初始化完成，策略: {self.cache_strategy}, Redis: {self.use_redis}")

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存值或None
        """
        if self.cache_strategy == "redis_first" and self.redis_cache:
            # 优先使用Redis
            value = self.redis_cache.get(key)
            if value is not None:
                logger.debug(f"Redis缓存命中: {key}")
                # 同步到本地缓存
                self.local_cache.set(value, key)
                return value

            # Redis未命中，尝试本地缓存
            value = self.local_cache.get(key)
            if value is not None:
                logger.debug(f"本地缓存命中: {key}")
                # 同步到Redis
                self.redis_cache.set(key, value)
                return value

        elif self.cache_strategy == "local_first":
            # 优先使用本地缓存
            value = self.local_cache.get(key)
            if value is not None:
                logger.debug(f"本地缓存命中: {key}")
                # 同步到Redis
                if self.redis_cache:
                    self.redis_cache.set(key, value)
                return value

            # 本地缓存未命中，尝试Redis
            if self.redis_cache:
                value = self.redis_cache.get(key)
                if value is not None:
                    logger.debug(f"Redis缓存命中: {key}")
                    # 同步到本地缓存
                    self.local_cache.set(value, key)
                    return value

        # 缓存未命中
        logger.debug(f"缓存未命中: {key}")
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        设置缓存

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）

        Returns:
            是否成功
        """
        # 设置本地缓存
        self.local_cache.set(value, key, ttl)

        # 设置Redis缓存
        if self.redis_cache:
            self.redis_cache.set(key, value, ttl)

        return True

    def delete(self, key: str) -> bool:
        """
        删除缓存

        Args:
            key: 缓存键

        Returns:
            是否成功
        """
        # 删除本地缓存
        self.local_cache.delete(key)

        # 删除Redis缓存
        if self.redis_cache:
            self.redis_cache.delete(key)

        return True

    def clear(self, pattern: str = "*") -> int:
        """
        清除缓存

        Args:
            pattern: 键模式

        Returns:
            清除的键数量
        """
        # 清除本地缓存
        self.local_cache.clear()

        # 清除Redis缓存
        if self.redis_cache:
            return self.redis_cache.clear(pattern)

        return 0

    def get_stats(self) -> dict:
        """
        获取缓存统计信息

        Returns:
            统计信息
        """
        stats = {"local_cache": self.local_cache.get_stats(), "strategy": self.cache_strategy}

        if self.redis_cache:
            stats["redis_cache"] = self.redis_cache.get_stats()

        return stats


# 全局缓存工厂实例
_cache_factory = None


def get_cache_factory():
    """
    获取缓存工厂实例

    Returns:
        CacheFactory实例
    """
    global _cache_factory
    if _cache_factory is None:
        _cache_factory = CacheFactory()
    return _cache_factory
