#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis缓存服务
提供分布式缓存支持，Redis不可用时降级到进程内LRU本地缓存
"""

import os
import json
import time
import hashlib
import redis
from collections import OrderedDict
from typing import Optional, Any
from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("redis_cache")

# 从配置读取重连间隔（秒），默认30s
try:
    from src.core.config.service_config import get_service_config as _get_svc_config
    _REDIS_RECONNECT_INTERVAL = _get_svc_config().REDIS_RECONNECT_INTERVAL
except Exception:
    _REDIS_RECONNECT_INTERVAL = 30

# 本地兜底缓存参数
_LOCAL_FALLBACK_MAX_SIZE = 1000
_LOCAL_FALLBACK_TTL = 300  # 5分钟


class _LocalFallbackCache:
    """进程内 LRU 缓存，作为 Redis 不可用时的兜底方案。

    使用 OrderedDict 实现 LRU 淘汰，支持 TTL 过期。
    仅缓存分类结果（JSON可序列化数据）。
    """

    def __init__(self, max_size: int = _LOCAL_FALLBACK_MAX_SIZE, ttl: int = _LOCAL_FALLBACK_TTL):
        self._store: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值，过期或不存在返回 None"""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expire_at = entry
        if time.time() > expire_at:
            # 过期，删除
            del self._store[key]
            return None
        # LRU: 移到末尾（最近使用）
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        effective_ttl = ttl if ttl is not None else self._ttl
        expire_at = time.time() + effective_ttl
        self._store[key] = (value, expire_at)
        self._store.move_to_end(key)
        # 淘汰最老的条目
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)
        return True

    def delete(self, key: str) -> bool:
        """删除缓存值"""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> int:
        """清空所有缓存，返回清除数量"""
        count = len(self._store)
        self._store.clear()
        return count

    def size(self) -> int:
        """返回当前缓存条目数"""
        return len(self._store)


class RedisCache:
    """
    Redis缓存实现，支持自动重连和本地兜底缓存
    """

    def __init__(self):
        """
        初始化Redis连接
        """
        self._local_fallback = _LocalFallbackCache()
        self._last_retry_time = 0.0
        self._connect()

    def _connect(self) -> None:
        """尝试连接 Redis，成功设置 available=True，失败设置 available=False"""
        try:
            self.redis_client = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                db=int(os.environ.get("REDIS_DB", 0)),
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            self.redis_client.ping()
            self.available = True
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            self.redis_client = None
            self.available = False
            self._last_retry_time = time.time()

    def try_reconnect(self) -> bool:
        """尝试重新连接 Redis（距上次尝试超过 REDIS_RECONNECT_INTERVAL 秒才重试）

        Returns:
            bool: 重连后是否可用
        """
        if self.available:
            return True

        now = time.time()
        if now - self._last_retry_time < _REDIS_RECONNECT_INTERVAL:
            return False

        logger.info(f"尝试重新连接 Redis (距上次尝试 {now - self._last_retry_time:.1f}s)...")
        self._last_retry_time = now
        self._connect()
        if self.available:
            logger.info("Redis 重连成功")
        else:
            logger.warning("[DEGRADE] Redis 不可用，降级到本地内存缓存")
        return self.available

    @staticmethod
    def compute_image_hash(image_data: bytes) -> str:
        """
        计算图像内容的MD5哈希

        Args:
            image_data: 图像字节数据

        Returns:
            MD5哈希字符串
        """
        return hashlib.md5(image_data).hexdigest()

    def get_image_result(self, image_hash: str) -> Optional[dict]:
        """
        获取图像分类结果缓存

        Args:
            image_hash: 图像哈希

        Returns:
            缓存的分类结果或None
        """
        key = f"classify:result:{image_hash}"
        return self.get(key)

    def set_image_result(self, image_hash: str, result: dict, ttl: int = 3600) -> bool:
        """
        缓存图像分类结果

        Args:
            image_hash: 图像哈希
            result: 分类结果
            ttl: 过期时间（秒），默认1小时

        Returns:
            是否成功
        """
        key = f"classify:result:{image_hash}"
        return self.set(key, result, ttl)

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存

        Redis 可用时从 Redis 读取；不可用时尝试重连，仍不可用则从本地兜底缓存读取。

        Args:
            key: 缓存键

        Returns:
            缓存值或None
        """
        if not self.available:
            # 尝试重连
            self.try_reconnect()

        if self.available:
            try:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            except Exception as e:
                logger.error(f"Redis获取失败: {e}")
                self.available = False
                self._last_retry_time = time.time()
                # 降级到本地缓存
                logger.warning("[DEGRADE] Redis 不可用，降级到本地内存缓存")

        # Redis 不可用，从本地兜底缓存读取
        return self._local_fallback.get(key)

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        设置缓存

        Redis 可用时写入 Redis；不可用时尝试重连，仍不可用则写入本地兜底缓存。

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）

        Returns:
            是否成功
        """
        if not self.available:
            self.try_reconnect()

        if self.available:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
                return True
            except Exception as e:
                logger.error(f"Redis设置失败: {e}")
                self.available = False
                self._last_retry_time = time.time()
                logger.warning("[DEGRADE] Redis 不可用，降级到本地内存缓存")

        # Redis 不可用，写入本地兜底缓存
        return self._local_fallback.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """
        删除缓存

        Args:
            key: 缓存键

        Returns:
            是否成功
        """
        # 始终删除本地缓存
        self._local_fallback.delete(key)

        if not self.available:
            self.try_reconnect()

        if self.available:
            try:
                self.redis_client.delete(key)
                return True
            except Exception as e:
                logger.error(f"Redis删除失败: {e}")
                self.available = False
                self._last_retry_time = time.time()
                return False
        return True

    def clear(self, pattern: str = "*") -> int:
        """
        清除缓存（使用 SCAN 替代 KEYS，避免阻塞 Redis）

        Args:
            pattern: 键模式

        Returns:
            清除的键数量
        """
        # 始终清除本地缓存
        local_count = self._local_fallback.clear()

        if not self.available:
            self.try_reconnect()

        if not self.available:
            return local_count

        try:
            deleted_count = 0
            cursor = 0
            while True:
                cursor, keys = self.redis_client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted_count += self.redis_client.delete(*keys)
                if cursor == 0:
                    break
            return deleted_count + local_count
        except Exception as e:
            logger.error(f"Redis清除失败: {e}")
            self.available = False
            self._last_retry_time = time.time()
            return local_count

    def exists(self, key: str) -> bool:
        """
        检查键是否存在

        Args:
            key: 缓存键

        Returns:
            是否存在
        """
        if not self.available:
            self.try_reconnect()

        if self.available:
            try:
                return bool(self.redis_client.exists(key))
            except Exception as e:
                logger.error(f"Redis检查失败: {e}")
                self.available = False
                self._last_retry_time = time.time()
                return False

        # 降级到本地缓存
        return self._local_fallback.get(key) is not None

    def get_stats(self) -> dict:
        """
        获取Redis状态

        Returns:
            状态信息
        """
        if not self.available:
            self.try_reconnect()

        if not self.available:
            return {
                "available": False,
                "local_fallback_size": self._local_fallback.size(),
                "error": "Redis连接失败，使用本地兜底缓存",
            }

        try:
            info = self.redis_client.info()
            return {
                "available": True,
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
                "local_fallback_size": self._local_fallback.size(),
            }
        except Exception as e:
            logger.error(f"Redis获取状态失败: {e}")
            self.available = False
            self._last_retry_time = time.time()
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

    如果实例已存在但不可用，且距上次重连尝试超过阈值，会自动尝试重连。

    Returns:
        RedisCache实例
    """
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    else:
        # 单例已存在，如果不可用则尝试重连
        if not _redis_cache.available:
            _redis_cache.try_reconnect()
    return _redis_cache
