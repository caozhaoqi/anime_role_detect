"""
缓存模块
提供多层缓存支持：
- ThreadLocalCache: 请求级缓存（线程隔离）
- TimedLRUCache: 进程级定时 LRU 缓存
- timed_lru_cache: 装饰器形式的定时缓存

借鉴 HCM Core 的多层缓存设计
"""

from .thread_cache import ThreadLocalCache, RequestCache, cache_on_thread, thread_cache
from .timed_lru_cache import (
    TimedLRUCache,
    timed_lru_cache,
    ttl_cache,
    LimitedDict,
)

__all__ = [
    'ThreadLocalCache',
    'RequestCache',
    'cache_on_thread',
    'thread_cache',
    'TimedLRUCache',
    'timed_lru_cache',
    'ttl_cache',
    'LimitedDict',
]