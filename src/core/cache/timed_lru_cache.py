"""
进程级定时 LRU 缓存
在 functools.lru_cache 基础上增加 TTL（过期时间）支持

借鉴 HCM Core 的 timed_lru_cache 设计
"""

import time
from functools import wraps, lru_cache
from collections import OrderedDict
from typing import Callable, Any, Optional


class TimedLRUCache:
    """
    定时 LRU 缓存
    支持最大缓存数量和过期时间

    Args:
        maxsize: 最大缓存数量，None 表示无限制
        ttl: 过期时间（秒），None 表示永不过期
    """

    def __init__(self, maxsize: Optional[int] = 128, ttl: Optional[int] = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = OrderedDict()

    def __getitem__(self, key) -> Any:
        if key not in self.cache:
            raise KeyError(key)
        value, timestamp = self.cache[key]
        if self.ttl is not None and time.time() - timestamp > self.ttl:
            del self.cache[key]
            raise KeyError(key)
        self.cache.move_to_end(key)
        return value

    def __setitem__(self, key, value) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif self.maxsize is not None and len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = (value, time.time())

    def __contains__(self, key) -> bool:
        if key not in self.cache:
            return False
        if self.ttl is not None and time.time() - self.cache[key][1] > self.ttl:
            del self.cache[key]
            return False
        return True

    def get(self, key, default=None) -> Any:
        """获取缓存值，不存在返回默认值"""
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, key, value) -> None:
        """设置缓存值"""
        self[key] = value

    def delete(self, key) -> None:
        """删除缓存值"""
        self.cache.pop(key, None)

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()

    def size(self) -> int:
        """获取缓存数量"""
        return len(self.cache)

    def is_expired(self, key) -> bool:
        """检查缓存是否过期"""
        if key not in self.cache:
            return True
        if self.ttl is not None and time.time() - self.cache[key][1] > self.ttl:
            return True
        return False


def timed_lru_cache(maxsize: int = 128, ttl: Optional[int] = None):
    """
    装饰器：定时 LRU 缓存

    Args:
        maxsize: 最大缓存数量
        ttl: 过期时间（秒），None 表示永不过期

    Usage:
        @timed_lru_cache(maxsize=256, ttl=3600)
        def expensive_function(x):
            # 耗时计算
            return result
    """

    def decorator(func: Callable) -> Callable:
        cache = TimedLRUCache(maxsize=maxsize, ttl=ttl)

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            return result

        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        wrapper.delete_cache = cache.delete
        return wrapper

    return decorator


def ttl_cache(ttl: int = 3600, maxsize: int = 128):
    """
    装饰器：TTL 缓存（别名）

    Args:
        ttl: 过期时间（秒）
        maxsize: 最大缓存数量
    """
    return timed_lru_cache(maxsize=maxsize, ttl=ttl)


class LimitedDict(OrderedDict):
    """
    容量限制字典
    支持按数量和按字节大小限制

    Args:
        max_len: 最大条目数量
        maxsize_bytes: 最大字节大小
    """

    def __init__(self, max_len: Optional[int] = None, maxsize_bytes: Optional[int] = None, *args, **kwargs):
        self.max_len = max_len
        self.maxsize_bytes = maxsize_bytes
        self.current_size = 0
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            self.current_size += self._get_size(value)

    def _get_size(self, value: Any) -> int:
        """估算值的字节大小"""
        if isinstance(value, (int, float, bool)):
            return 28
        elif isinstance(value, str):
            return len(value.encode('utf-8'))
        elif isinstance(value, (list, tuple)):
            return sum(self._get_size(item) for item in value)
        elif isinstance(value, dict):
            return sum(self._get_size(k) + self._get_size(v) for k, v in value.items())
        else:
            return 100

    def __setitem__(self, key, value) -> None:
        if key in self:
            old_value = self[key]
            self.current_size -= self._get_size(old_value)

        new_size = self._get_size(value)

        while self.maxsize_bytes is not None and self.current_size + new_size > self.maxsize_bytes:
            if len(self) == 0:
                break
            oldest_key, oldest_value = self.popitem(last=False)
            self.current_size -= self._get_size(oldest_value)

        while self.max_len is not None and len(self) >= self.max_len:
            oldest_key, oldest_value = self.popitem(last=False)
            self.current_size -= self._get_size(oldest_value)

        super().__setitem__(key, value)
        self.current_size += new_size

    def __delitem__(self, key) -> None:
        if key in self:
            self.current_size -= self._get_size(self[key])
        super().__delitem__(key)

    def is_full(self) -> bool:
        """检查是否已满"""
        if self.max_len is not None and len(self) >= self.max_len:
            return True
        if self.maxsize_bytes is not None and self.current_size >= self.maxsize_bytes:
            return True
        return False

    def is_empty(self) -> bool:
        """检查是否为空"""
        return len(self) == 0