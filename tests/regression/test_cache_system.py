"""Regression tests for cache system (2026-07-24).

Tests cover:
1. ThreadLocalCache - request-level caching
2. TimedLRUCache - process-level timed caching
3. timed_lru_cache decorator
4. RequestCache context manager
5. LimitedDict - size-limited dictionary
"""
import os
import sys
import time
import threading
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestThreadLocalCache:
    """Test ThreadLocalCache functionality."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        from core.cache import ThreadLocalCache
        ThreadLocalCache.clear()
        
        ThreadLocalCache.set('key', 'value')
        assert ThreadLocalCache.get('key') == 'value'

    def test_get_default(self):
        """Test get with default value."""
        from core.cache import ThreadLocalCache
        ThreadLocalCache.clear()
        
        result = ThreadLocalCache.get('nonexistent', 'default')
        assert result == 'default'

    def test_set_many(self):
        """Test batch set operation."""
        from core.cache import ThreadLocalCache
        ThreadLocalCache.clear()
        
        ThreadLocalCache.set_many({'a': 1, 'b': 2, 'c': 3})
        assert ThreadLocalCache.get('a') == 1
        assert ThreadLocalCache.get('b') == 2
        assert ThreadLocalCache.get('c') == 3

    def test_delete(self):
        """Test delete operation."""
        from core.cache import ThreadLocalCache
        ThreadLocalCache.clear()
        
        ThreadLocalCache.set('key', 'value')
        ThreadLocalCache.delete('key')
        assert ThreadLocalCache.get('key') is None

    def test_exists(self):
        """Test exists operation."""
        from core.cache import ThreadLocalCache
        ThreadLocalCache.clear()
        
        ThreadLocalCache.set('key', 'value')
        assert ThreadLocalCache.exists('key') is True
        assert ThreadLocalCache.exists('nonexistent') is False

    def test_size(self):
        """Test size operation."""
        from core.cache import ThreadLocalCache
        ThreadLocalCache.clear()
        
        ThreadLocalCache.set('a', 1)
        ThreadLocalCache.set('b', 2)
        assert ThreadLocalCache.size() == 2

    def test_clear(self):
        """Test clear operation."""
        from core.cache import ThreadLocalCache
        ThreadLocalCache.set('a', 1)
        ThreadLocalCache.set('b', 2)
        
        ThreadLocalCache.clear()
        assert ThreadLocalCache.size() == 0

    def test_get_or_set(self):
        """Test get_or_set operation."""
        from core.cache import ThreadLocalCache
        ThreadLocalCache.clear()
        
        result = ThreadLocalCache.get_or_set('key', lambda: 'computed')
        assert result == 'computed'
        
        result = ThreadLocalCache.get_or_set('key', lambda: 'should_not_call')
        assert result == 'computed'

    def test_thread_isolation(self):
        """Test thread isolation."""
        from core.cache import ThreadLocalCache
        ThreadLocalCache.clear()
        
        results = []
        
        def worker1():
            ThreadLocalCache.set('shared_key', 'thread1')
            time.sleep(0.1)
            results.append(ThreadLocalCache.get('shared_key'))
        
        def worker2():
            ThreadLocalCache.set('shared_key', 'thread2')
            time.sleep(0.1)
            results.append(ThreadLocalCache.get('shared_key'))
        
        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        assert 'thread1' in results
        assert 'thread2' in results


class TestTimedLRUCache:
    """Test TimedLRUCache functionality."""

    def test_basic_operations(self):
        """Test basic cache operations."""
        from core.cache import TimedLRUCache
        
        cache = TimedLRUCache(maxsize=3)
        cache['a'] = 1
        cache['b'] = 2
        cache['c'] = 3
        
        assert cache['a'] == 1
        assert cache.get('b') == 2

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        from core.cache import TimedLRUCache
        
        cache = TimedLRUCache(maxsize=2)
        cache['a'] = 1
        cache['b'] = 2
        cache['c'] = 3
        
        assert 'a' not in cache
        assert 'b' in cache
        assert 'c' in cache

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        from core.cache import TimedLRUCache
        
        cache = TimedLRUCache(ttl=0.1)
        cache['a'] = 1
        
        assert cache['a'] == 1
        time.sleep(0.2)
        assert 'a' not in cache

    def test_is_expired(self):
        """Test is_expired method."""
        from core.cache import TimedLRUCache
        
        cache = TimedLRUCache(ttl=0.1)
        cache['a'] = 1
        
        assert cache.is_expired('a') is False
        time.sleep(0.2)
        assert cache.is_expired('a') is True

    def test_clear(self):
        """Test clear method."""
        from core.cache import TimedLRUCache
        
        cache = TimedLRUCache(maxsize=3)
        cache['a'] = 1
        cache['b'] = 2
        
        cache.clear()
        assert cache.size() == 0


class TestTimedLRUCacheDecorator:
    """Test timed_lru_cache decorator."""

    def test_cache_hits(self):
        """Test that cached results are returned."""
        from core.cache import timed_lru_cache
        
        call_count = [0]
        
        @timed_lru_cache(maxsize=2)
        def func(x):
            call_count[0] += 1
            return x * 2
        
        assert func(1) == 2
        assert call_count[0] == 1
        assert func(1) == 2
        assert call_count[0] == 1

    def test_cache_eviction(self):
        """Test LRU eviction in decorator."""
        from core.cache import timed_lru_cache
        
        call_count = [0]
        
        @timed_lru_cache(maxsize=2)
        def func(x):
            call_count[0] += 1
            return x * 2
        
        func(1)
        func(2)
        func(3)
        func(1)
        
        assert call_count[0] == 4

    def test_ttl_expiration(self):
        """Test TTL expiration in decorator."""
        from core.cache import timed_lru_cache
        
        call_count = [0]
        
        @timed_lru_cache(ttl=0.1)
        def func(x):
            call_count[0] += 1
            return x * 2
        
        func(1)
        assert call_count[0] == 1
        func(1)
        assert call_count[0] == 1
        
        time.sleep(0.2)
        func(1)
        assert call_count[0] == 2

    def test_clear_cache(self):
        """Test clear_cache method."""
        from core.cache import timed_lru_cache
        
        @timed_lru_cache(maxsize=2)
        def func(x):
            return x * 2
        
        func(1)
        func.clear_cache()
        assert func.cache.size() == 0


class TestRequestCache:
    """Test RequestCache context manager."""

    def test_context_manager(self):
        """Test RequestCache context manager."""
        from core.cache import RequestCache, ThreadLocalCache
        
        with RequestCache():
            ThreadLocalCache.set('key', 'value')
            assert ThreadLocalCache.get('key') == 'value'
        
        assert ThreadLocalCache.get('key') is None


class TestLimitedDict:
    """Test LimitedDict functionality."""

    def test_max_len_limit(self):
        """Test max_len limit."""
        from core.cache import LimitedDict
        
        d = LimitedDict(max_len=2)
        d['a'] = 1
        d['b'] = 2
        d['c'] = 3
        
        assert len(d) == 2
        assert 'a' not in d

    def test_is_full(self):
        """Test is_full method."""
        from core.cache import LimitedDict
        
        d = LimitedDict(max_len=2)
        assert d.is_full() is False
        
        d['a'] = 1
        d['b'] = 2
        assert d.is_full() is True

    def test_is_empty(self):
        """Test is_empty method."""
        from core.cache import LimitedDict
        
        d = LimitedDict(max_len=2)
        assert d.is_empty() is True
        
        d['a'] = 1
        assert d.is_empty() is False


class TestCacheIntegration:
    """Integration tests for cache system."""

    def test_cache_imports(self):
        """Test that all cache components can be imported."""
        from core.cache import (
            ThreadLocalCache,
            RequestCache,
            cache_on_thread,
            thread_cache,
            TimedLRUCache,
            timed_lru_cache,
            ttl_cache,
            LimitedDict,
        )
        
        assert ThreadLocalCache is not None
        assert RequestCache is not None
        assert cache_on_thread is not None
        assert thread_cache is not None
        assert TimedLRUCache is not None
        assert timed_lru_cache is not None
        assert ttl_cache is not None
        assert LimitedDict is not None

    def test_cache_module_init(self):
        """Test cache module __init__ imports."""
        import core.cache
        
        assert hasattr(core.cache, 'ThreadLocalCache')
        assert hasattr(core.cache, 'timed_lru_cache')