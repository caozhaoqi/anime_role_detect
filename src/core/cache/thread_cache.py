"""
请求级 ThreadLocal 缓存
用于单次请求内的临时数据缓存，避免重复计算和数据库查询

借鉴 HCM Core 的 HCMThreadCache 设计
"""

import threading
from typing import Any, Optional


class ThreadLocalCache:
    """
    线程本地缓存
    每个线程拥有独立的缓存空间，线程结束时自动清理
    """

    _local = threading.local()

    @classmethod
    def _get_cache_dict(cls) -> dict:
        """获取当前线程的缓存字典"""
        if not hasattr(cls._local, 'cache'):
            cls._local.cache = {}
        return cls._local.cache

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        获取缓存值

        Args:
            key: 缓存键
            default: 默认值

        Returns:
            缓存值或默认值
        """
        cache = cls._get_cache_dict()
        return cache.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any) -> Any:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值

        Returns:
            设置的值
        """
        cache = cls._get_cache_dict()
        cache[key] = value
        return value

    @classmethod
    def set_many(cls, items: dict) -> None:
        """
        批量设置缓存值

        Args:
            items: 键值对字典
        """
        cache = cls._get_cache_dict()
        cache.update(items)

    @classmethod
    def delete(cls, key: str) -> None:
        """
        删除缓存值

        Args:
            key: 缓存键
        """
        cache = cls._get_cache_dict()
        cache.pop(key, None)

    @classmethod
    def delete_many(cls, keys: list) -> None:
        """
        批量删除缓存值

        Args:
            keys: 缓存键列表
        """
        cache = cls._get_cache_dict()
        for key in keys:
            cache.pop(key, None)

    @classmethod
    def clear(cls) -> None:
        """清空当前线程的所有缓存"""
        if hasattr(cls._local, 'cache'):
            cls._local.cache = {}

    @classmethod
    def exists(cls, key: str) -> bool:
        """
        检查缓存键是否存在

        Args:
            key: 缓存键

        Returns:
            是否存在
        """
        cache = cls._get_cache_dict()
        return key in cache

    @classmethod
    def keys(cls) -> list:
        """获取所有缓存键"""
        cache = cls._get_cache_dict()
        return list(cache.keys())

    @classmethod
    def size(cls) -> int:
        """获取缓存条目数量"""
        cache = cls._get_cache_dict()
        return len(cache)

    @classmethod
    def get_or_set(cls, key: str, default_factory) -> Any:
        """
        获取缓存值，如果不存在则调用工厂函数设置

        Args:
            key: 缓存键
            default_factory: 工厂函数，返回默认值

        Returns:
            缓存值
        """
        cache = cls._get_cache_dict()
        if key in cache:
            return cache[key]
        value = default_factory()
        cache[key] = value
        return value


class RequestCache:
    """
    请求级缓存上下文管理器
    使用方式：
        with RequestCache():
            # 在请求期间使用 ThreadLocalCache
            ThreadLocalCache.set('key', 'value')
        # 请求结束后自动清空缓存
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ThreadLocalCache.clear()


def cache_on_thread(key: str):
    """
    装饰器：缓存函数结果到 ThreadLocalCache

    Args:
        key: 缓存键

    Usage:
        @cache_on_thread('model_result')
        def compute_heavy():
            # 耗时计算
            return result
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = f"{key}:{hash((args, frozenset(kwargs.items())))}"
            cached = ThreadLocalCache.get(cache_key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            ThreadLocalCache.set(cache_key, result)
            return result

        return wrapper

    return decorator


# 全局别名，兼容现有代码风格
thread_cache = ThreadLocalCache