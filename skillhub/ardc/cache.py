#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 缓存模块
提供统一的缓存接口，支持热点数据缓存、缓存失效策略
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


class CacheClient:
    """Redis 缓存客户端"""

    def __init__(self):
        self._client = None
        self._connect()

    def _connect(self):
        """连接到 Redis"""
        try:
            self._client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password,
                ssl=settings.redis.ssl,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            # 测试连接
            self._client.ping()
            logger.info(f"Redis 连接成功: {settings.redis.host}:{settings.redis.port}")
        except RedisConnectionError as e:
            logger.warning(f"Redis 连接失败，将使用内存缓存: {e}")
            self._client = None

    @property
    def client(self) -> Optional[redis.Redis]:
        """获取 Redis 客户端，如果连接失败返回 None"""
        if self._client is None:
            self._connect()
        return self._client

    def _get_key(self, key: str) -> str:
        """添加前缀的完整键名"""
        return f"{settings.redis.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if self.client is None:
            return None

        try:
            value = self.client.get(self._get_key(key))
            if value is None:
                return None

            # 尝试解析 JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Redis get 失败: {key}, 错误: {e}")
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """设置缓存值"""
        if self.client is None:
            return False

        try:
            # 如果是复杂类型，序列化为 JSON
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)

            ttl = ttl_seconds or settings.redis.cache_ttl_seconds
            self.client.setex(self._get_key(key), ttl, value)
            return True
        except Exception as e:
            logger.error(f"Redis set 失败: {key}, 错误: {e}")
            return False

    def delete(self, key: str) -> bool:
        """删除缓存"""
        if self.client is None:
            return False

        try:
            self.client.delete(self._get_key(key))
            return True
        except Exception as e:
            logger.error(f"Redis delete 失败: {key}, 错误: {e}")
            return False

    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        if self.client is None:
            return False

        try:
            return self.client.exists(self._get_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis exists 失败: {key}, 错误: {e}")
            return False

    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配模式的键"""
        if self.client is None:
            return []

        try:
            full_pattern = self._get_key(pattern)
            return [k.replace(settings.redis.prefix, "") for k in self.client.keys(full_pattern)]
        except Exception as e:
            logger.error(f"Redis keys 失败: {pattern}, 错误: {e}")
            return []

    def flush_all(self) -> bool:
        """清空所有缓存"""
        if self.client is None:
            return False

        try:
            self.client.flushdb()
            logger.info("缓存已清空")
            return True
        except Exception as e:
            logger.error(f"Redis flushdb 失败: {e}")
            return False

    def incr(self, key: str) -> Optional[int]:
        """递增计数器"""
        if self.client is None:
            return None

        try:
            return self.client.incr(self._get_key(key))
        except Exception as e:
            logger.error(f"Redis incr 失败: {key}, 错误: {e}")
            return None

    def decr(self, key: str) -> Optional[int]:
        """递减计数器"""
        if self.client is None:
            return None

        try:
            return self.client.decr(self._get_key(key))
        except Exception as e:
            logger.error(f"Redis decr 失败: {key}, 错误: {e}")
            return None


# 创建全局缓存客户端实例
cache = CacheClient()


def cached(key_prefix: str, ttl_seconds: Optional[int] = None):
    """
    缓存装饰器

    Args:
        key_prefix: 缓存键前缀
        ttl_seconds: 缓存过期时间（秒），默认为配置中的值

    Example:
        @cached("skill_list", ttl_seconds=3600)
        def get_skills():
            # ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 构建缓存键
            key_parts = [key_prefix]

            # 添加位置参数作为键的一部分
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))

            # 添加关键字参数作为键的一部分
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}={v}")

            cache_key = ":".join(key_parts)

            # 尝试获取缓存
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_result

            # 执行函数并缓存结果
            result = func(*args, **kwargs)

            # 只缓存非 None 结果
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


# 缓存键常量定义
class CacheKeys:
    """缓存键常量"""

    # 技能相关
    SKILL_LIST = "skills:list"
    SKILL_DETAIL = "skills:detail:{skill_id}"
    SKILL_VERSIONS = "skills:versions:{skill_id}"
    SKILL_SEARCH = "skills:search:{keyword}:{category}"

    # 分类相关
    CATEGORIES = "categories"
    CATEGORY_SKILLS = "category:{category_id}:skills"

    # 统计相关
    STATS = "stats"

    # 用户相关
    USER_FAVORITES = "user:{user_id}:favorites"

    # 版本检查
    VERSION_CHECK = "version:check:{skill_id}:{current_version}"


# 内存缓存回退（当 Redis 不可用时使用）
class MemoryCache:
    """简单的内存缓存实现，作为 Redis 的回退"""

    def __init__(self):
        self._cache: Dict[str, dict] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None

        # 检查是否过期
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
