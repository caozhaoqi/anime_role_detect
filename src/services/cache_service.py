import asyncio
import hashlib
import json
import os
from typing import Any, Dict, Optional
import redis
import aioredis

from src.core.logging.global_logger import get_logger

logger = get_logger("cache_service")

class CacheManager:
    """缓存管理器 - 支持多级缓存策略"""
    
    _instance: Optional['CacheManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, "initialized") and self.initialized:
            return
        
        self.initialized = True
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.redis_client = None
        self.redis_available = False
        
        # 配置
        self.REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
        self.REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
        self.REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', '')
        self.REDIS_DB = int(os.environ.get('REDIS_DB', '0'))
        
        self.CACHE_TTL = int(os.environ.get('CACHE_TTL', '3600'))
        self.LOCAL_CACHE_SIZE = int(os.environ.get('LOCAL_CACHE_SIZE', '1000'))
        
        # 初始化Redis连接
        self._init_redis()
    
    def _init_redis(self):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.Redis(
                host=self.REDIS_HOST,
                port=self.REDIS_PORT,
                password=self.REDIS_PASSWORD,
                db=self.REDIS_DB,
                decode_responses=True
            )
            # 测试连接
            self.redis_client.ping()
            self.redis_available = True
            logger.info(f"Redis连接成功: {self.REDIS_HOST}:{self.REDIS_PORT}")
        except Exception as e:
            logger.warning(f"Redis连接失败，将使用本地缓存: {e}")
            self.redis_available = False
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """生成缓存键"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        hash_obj = hashlib.md5(content.encode('utf-8'))
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        # 1. 先从本地缓存获取
        if key in self.local_cache:
            cached_data = self.local_cache[key]
            logger.debug(f"从本地缓存获取: {key}")
            return cached_data['value']
        
        # 2. 从Redis获取
        if self.redis_available:
            try:
                value = self.redis_client.get(key)
                if value:
                    value = json.loads(value)
                    # 更新本地缓存
                    self._update_local_cache(key, value)
                    logger.debug(f"从Redis获取: {key}")
                    return value
            except Exception as e:
                logger.error(f"Redis获取失败: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存"""
        ttl = ttl or self.CACHE_TTL
        
        try:
            # 1. 更新本地缓存
            self._update_local_cache(key, value)
            
            # 2. 更新Redis缓存
            if self.redis_available:
                try:
                    serialized_value = json.dumps(value, ensure_ascii=False)
                    self.redis_client.setex(key, ttl, serialized_value)
                    logger.debug(f"设置Redis缓存: {key}, TTL: {ttl}")
                except Exception as e:
                    logger.error(f"Redis设置失败: {e}")
            
            return True
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            # 1. 从本地缓存删除
            if key in self.local_cache:
                del self.local_cache[key]
                logger.debug(f"从本地缓存删除: {key}")
            
            # 2. 从Redis删除
            if self.redis_available:
                try:
                    self.redis_client.delete(key)
                    logger.debug(f"从Redis删除: {key}")
                except Exception as e:
                    logger.error(f"Redis删除失败: {e}")
            
            return True
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False
    
    def clear(self) -> bool:
        """清空缓存"""
        try:
            # 1. 清空本地缓存
            self.local_cache.clear()
            logger.debug("清空本地缓存")
            
            # 2. 清空Redis缓存
            if self.redis_available:
                try:
                    self.redis_client.flushdb()
                    logger.debug("清空Redis缓存")
                except Exception as e:
                    logger.error(f"Redis清空失败: {e}")
            
            return True
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False
    
    def _update_local_cache(self, key: str, value: Any):
        """更新本地缓存，保持大小限制"""
        # 检查本地缓存大小
        if len(self.local_cache) >= self.LOCAL_CACHE_SIZE:
            # 删除最旧的缓存项
            oldest_key = next(iter(self.local_cache))
            del self.local_cache[oldest_key]
        
        self.local_cache[key] = {
            'value': value,
            'timestamp': asyncio.get_event_loop().time()
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            'local_cache_size': len(self.local_cache),
            'redis_available': self.redis_available,
            'local_cache_max_size': self.LOCAL_CACHE_SIZE
        }
        
        if self.redis_available:
            try:
                redis_info = self.redis_client.info('memory')
                stats['redis_memory_used'] = redis_info.get('used_memory_human', 'N/A')
            except Exception as e:
                logger.error(f"获取Redis信息失败: {e}")
        
        return stats

# 全局缓存管理器实例
_cache_manager: Optional[CacheManager] = None

def get_cache_manager() -> CacheManager:
    """获取缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def init_cache_manager():
    """初始化缓存管理器"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        logger.info("缓存管理器初始化完成")
        logger.info(f"缓存统计: {_cache_manager.get_cache_stats()}")
    return _cache_manager

def get_cache(key: str) -> Optional[Any]:
    """获取缓存"""
    return get_cache_manager().get(key)

def set_cache(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """设置缓存"""
    return get_cache_manager().set(key, value, ttl)

def delete_cache(key: str) -> bool:
    """删除缓存"""
    return get_cache_manager().delete(key)

def clear_cache() -> bool:
    """清空缓存"""
    return get_cache_manager().clear()

def get_cache_stats() -> Dict[str, Any]:
    """获取缓存统计信息"""
    return get_cache_manager().get_cache_stats()
