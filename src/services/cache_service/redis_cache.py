#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis缓存服务
提供分布式缓存支持
"""

import os
import json
import redis
from typing import Optional, Any
from src.core.logging.global_logger import get_logger

logger = get_logger("redis_cache")


class RedisCache:
    """
    Redis缓存实现
    """

    def __init__(self):
        """
        初始化Redis连接
        """
        try:
            self.redis_client = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                db=int(os.environ.get("REDIS_DB", 0)),
                decode_responses=True,
            )
            # 测试连接
            self.redis_client.ping()
            self.available = True
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            self.redis_client = None
            self.available = False

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存值或None
        """
        if not self.available:
            return None

        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Redis获取失败: {e}")
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
        if not self.available:
            return False

        try:
            self.redis_client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
            return True
        except Exception as e:
            logger.error(f"Redis设置失败: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        删除缓存

        Args:
            key: 缓存键

        Returns:
            是否成功
        """
        if not self.available:
            return False

        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis删除失败: {e}")
            return False

    def clear(self, pattern: str = "*") -> int:
        """
        清除缓存

        Args:
            pattern: 键模式

        Returns:
            清除的键数量
        """
        if not self.available:
            return 0

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis清除失败: {e}")
            return 0

    def exists(self, key: str) -> bool:
        """
        检查键是否存在

        Args:
            key: 缓存键

        Returns:
            是否存在
        """
        if not self.available:
            return False

        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Redis检查失败: {e}")
            return False

    def get_stats(self) -> dict:
        """
        获取Redis状态

        Returns:
            状态信息
        """
        if not self.available:
            return {"available": False, "error": "Redis连接失败"}

        try:
            info = self.redis_client.info()
            return {
                "available": True,
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
            }
        except Exception as e:
            logger.error(f"Redis获取状态失败: {e}")
            return {"available": False, "error": str(e)}

    def _calculate_hit_rate(self, info: dict) -> float:
        """
        计算命中率

        Args:
            info: Redis信息

        Returns:
            命中率
        """
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0


# 全局Redis缓存实例
_redis_cache = None


def get_redis_cache():
    """
    获取Redis缓存实例

    Returns:
        RedisCache实例
    """
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache
