"""
 - LRU
"""

import time
from collections import OrderedDict
from typing import Any, Optional


class ModelCache:
    """
    LRU

    Args:
        max_size: 
        ttl_seconds: 1
    """

    def __init__(self, max_size: int = 10, ttl_seconds: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.timestamps = {}

    def get(self, key: str) -> Optional[Any]:
        """
        

        Args:
            key: 

        Returns:
            None
        """
        if key not in self.cache:
            return None

        # 
        now = time.time()
        if now - self.timestamps[key] > self.ttl_seconds:
            self._remove(key)
            return None

        # LRU
        self.cache.move_to_end(key)
        self.timestamps[key] = now

        return self.cache[key]

    def set(self, key: str, value: Any) -> None:
        """
        

        Args:
            key: 
            value: 
        """
        if key in self.cache:
            # 
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            # 
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)

        self.cache[key] = value
        self.timestamps[key] = time.time()

    def _remove(self, key: str) -> None:
        """
        

        Args:
            key: 
        """
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]

    def delete(self, key: str) -> None:
        """
        

        Args:
            key: 
        """
        self._remove(key)

    def clear(self) -> None:
        """"""
        self.cache.clear()
        self.timestamps.clear()

    def size(self) -> int:
        """"""
        return len(self.cache)

    def keys(self):
        """"""
        return list(self.cache.keys())

    def contains(self, key: str) -> bool:
        """"""
        return key in self.cache

    def cleanup_expired(self) -> int:
        """
        

        Returns:
            
        """
        now = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items() if now - timestamp > self.ttl_seconds
        ]

        for key in expired_keys:
            self._remove(key)

        return len(expired_keys)


# 
model_cache = ModelCache(max_size=5, ttl_seconds=3600)
