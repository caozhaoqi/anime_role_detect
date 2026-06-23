#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存模块
提供统一的缓存接口，支持热点数据缓存、缓存失效策略
支持 Redis 和内存缓存两种模式
"""

import json
import functools
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime

import redis
from redis.exceptions import ConnectionError as RedisConnectionError

from ardc.config import settings
from ardc.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryCache:
    """简单的内存缓存实现，作为 Redis 的回退或默认缓存"""

    def __init__(self):
        self._cache: Dict[str, dict] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None

        if datetime.now().timestamp() > entry["expire_at"]:
            del self._cache[key]
            return None

        return entry["value"]

    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        self._cache[key] = {"value": value, "expire_at": datetime.now().timestamp() + ttl_seconds}

    def delete(self, key: str):
        self._cache.pop(key, None)

    def exists(self, key: str) -> bool:
        entry = self._cache.get(key)
        if entry is None:
            return False
        if datetime.now().timestamp() > entry["expire_at"]:
            del self._cache[key]
            return False
        return True

    def keys(self, pattern: str = "*") -> List[str]:
        return list(self._cache.keys())

    def flush_all(self):
        self._cache.clear()


class CacheClient:
    """缓存客户端，支持 Redis 和内存缓存"""

    def __init__(self):
        self._redis_client = None
        self._memory_cache = MemoryCache()
        self._connect()

    def _connect(self):
        """连接到 Redis（如果启用）"""
        if not settings.redis.enabled:
            logger.info("Redis 未启用，将使用内存缓存")
            return

        try:
            self._redis_client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password,
                ssl=settings.redis.ssl,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            self._redis_client.ping()
            logger.info(f"Redis 连接成功: {settings.redis.host}:{settings.redis.port}")
        except RedisConnectionError as e:
            logger.warning(f"Redis 连接失败，将使用内存缓存: {e}")
            self._redis_client = None

    @property
    def client(self) -> Optional[redis.Redis]:
        """获取 Redis 客户端，如果未启用或连接失败返回 None"""
        if not settings.redis.enabled:
            return None
        if self._redis_client is None:
            self._connect()
        return self._redis_client

    def _get_key(self, key: str) -> str:
        """添加前缀的完整键名"""
        return f"{settings.redis.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if self.client is not None:
            try:
                value = self.client.get(self._get_key(key))
                if value is None:
                    return None

                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            except Exception as e:
                logger.error(f"Redis get 失败: {key}, 错误: {e}")

        return self._memory_cache.get(key)

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """设置缓存值"""
        ttl = ttl_seconds or settings.redis.cache_ttl_seconds

        if self.client is not None:
            try:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)

                self.client.setex(self._get_key(key), ttl, value)
                return True
            except Exception as e:
                logger.error(f"Redis set 失败: {key}, 错误: {e}")

        self._memory_cache.set(key, value, ttl)
        return True

    def delete(self, key: str) -> bool:
        """删除缓存"""
        if self.client is not None:
            try:
                self.client.delete(self._get_key(key))
                return True
            except Exception as e:
                logger.error(f"Redis delete 失败: {key}, 错误: {e}")

        self._memory_cache.delete(key)
        return True

    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        if self.client is not None:
            try:
                return self.client.exists(self._get_key(key)) > 0
            except Exception as e:
                logger.error(f"Redis exists 失败: {key}, 错误: {e}")

        return self._memory_cache.exists(key)

    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配模式的键"""
        if self.client is not None:
            try:
                full_pattern = self._get_key(pattern)
                return [k.replace(settings.redis.prefix, "") for k in self.client.keys(full_pattern)]
            except Exception as e:
                logger.error(f"Redis keys 失败: {pattern}, 错误: {e}")

        return self._memory_cache.keys(pattern)

    def flush_all(self) -> bool:
        """清空所有缓存"""
        if self.client is not None:
            try:
                self.client.flushdb()
                logger.info("Redis 缓存已清空")
                self._memory_cache.flush_all()
                return True
            except Exception as e:
                logger.error(f"Redis flushdb 失败: {e}")

        self._memory_cache.flush_all()
        logger.info("内存缓存已清空")
        return True

    def incr(self, key: str) -> Optional[int]:
        """递增计数器"""
        if self.client is not None:
            try:
                return self.client.incr(self._get_key(key))
            except Exception as e:
                logger.error(f"Redis incr 失败: {key}, 错误: {e}")

        return None

    def decr(self, key: str) -> Optional[int]:
        """递减计数器"""
        if self.client is not None:
            try:
                return self.client.decr(self._get_key(key))
            except Exception as e:
                logger.error(f"Redis decr 失败: {key}, 错误: {e}")

        return None


cache = CacheClient()


def cached(key_prefix: str, ttl_seconds: Optional[int] = None):
    """
    缓存装饰器

    Args:
        key_prefix: 缓存键前缀
        ttl_seconds: 缓存过期时间（秒），默认为配置中的值
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key_parts = [key_prefix]

            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))

            for k, v in sorted(kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}={v}")

            cache_key = ":".join(key_parts)

            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_result

            result = func(*args, **kwargs)

            if result is not None:
                cache.set(cache_key, result, ttl_seconds)
                logger.debug(f"缓存设置: {cache_key}")

            return result

        return wrapper

    return decorator


def invalidate_cache(pattern: str):
    """
    失效匹配模式的缓存

    Args:
        pattern: 缓存键模式，支持通配符 *
    """
    keys = cache.keys(pattern)
    for key in keys:
        cache.delete(key)
    logger.info(f"已失效 {len(keys)} 个缓存")


class CacheKeys:
    """缓存键常量"""

    SKILL_LIST = "skills:list"
    SKILL_DETAIL = "skills:detail:{skill_id}"
    SKILL_VERSIONS = "skills:versions:{skill_id}"
    SKILL_SEARCH = "skills:search:{keyword}:{category}"

    CATEGORIES = "categories"
    CATEGORY_SKILLS = "category:{category_id}:skills"

    STATS = "stats"

    USER_FAVORITES = "user:{user_id}:favorites"

    VERSION_CHECK = "version:check:{skill_id}:{current_version}"
