#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存装饰器模块
提供请求级缓存封装，支持getOrSet模式
"""

import hashlib
import json
import functools
from typing import Callable, Any, Optional, Dict
from src.services.cache_service.redis_cache import get_redis_cache
from src.core.logging.global_logger import get_logger

logger = get_logger("cache_decorator")


def cache_result(prefix: str = "cache", ttl: int = 3600):
    """
    缓存装饰器
    将函数返回值缓存到Redis中
    
    Args:
        prefix: 缓存键前缀
        ttl: 过期时间（秒），默认1小时
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 计算缓存键
            cache_key = _generate_cache_key(prefix, func.__name__, args, kwargs)
            
            # 获取Redis缓存
            redis_cache = get_redis_cache()
            
            # 尝试从缓存获取
            cached_result = redis_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_result
            
            # 缓存未命中，执行函数
            result = func(*args, **kwargs)
            
            # 将结果写入缓存
            if result is not None:
                success = redis_cache.set(cache_key, result, ttl)
                if success:
                    logger.debug(f"缓存写入成功: {cache_key}")
                else:
                    logger.debug(f"缓存写入失败: {cache_key}")
            
            return result
        return wrapper
    return decorator


def cache_with_key(key_func: Callable):
    """
    自定义键生成的缓存装饰器
    
    Args:
        key_func: 自定义键生成函数，接收与原函数相同的参数
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 使用自定义函数生成缓存键
            cache_key = key_func(*args, **kwargs)
            
            # 获取Redis缓存
            redis_cache = get_redis_cache()
            
            # 尝试从缓存获取
            cached_result = redis_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_result
            
            # 缓存未命中，执行函数
            result = func(*args, **kwargs)
            
            # 将结果写入缓存（默认TTL为1小时）
            if result is not None:
                success = redis_cache.set(cache_key, result)
                if success:
                    logger.debug(f"缓存写入成功: {cache_key}")
            
            return result
        return wrapper
    return decorator


def get_or_set(cache_key: str, loader: Callable, ttl: int = 3600) -> Any:
    """
    getOrSet模式：先查缓存，未命中则执行loader并写入缓存
    
    Args:
        cache_key: 缓存键
        loader: 数据加载函数，无参数，返回要缓存的数据
        ttl: 过期时间（秒）
    
    Returns:
        缓存数据或loader返回的数据
    """
    redis_cache = get_redis_cache()
    
    # 尝试从缓存获取
    cached_result = redis_cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"getOrSet 缓存命中: {cache_key}")
        return cached_result
    
    # 缓存未命中，执行loader
    logger.debug(f"getOrSet 缓存未命中，执行loader: {cache_key}")
    result = loader()
    
    # 将结果写入缓存
    if result is not None:
        success = redis_cache.set(cache_key, result, ttl)
        if success:
            logger.debug(f"getOrSet 缓存写入成功: {cache_key}")
    
    return result


def invalidate_cache(pattern: str) -> int:
    """
    失效指定模式的缓存
    
    Args:
        pattern: 键模式，支持通配符
    
    Returns:
        清除的键数量
    """
    redis_cache = get_redis_cache()
    count = redis_cache.clear(pattern)
    logger.info(f"清除缓存模式 '{pattern}'，共 {count} 个键")
    return count


def _generate_cache_key(prefix: str, func_name: str, args: tuple, kwargs: dict) -> str:
    """
    生成缓存键
    
    Args:
        prefix: 前缀
        func_name: 函数名
        args: 位置参数
        kwargs: 关键字参数
    
    Returns:
        缓存键字符串
    """
    # 将参数转换为可哈希的字符串
    args_str = str(args)
    kwargs_str = str(sorted(kwargs.items()))
    
    # 计算参数哈希
    params_hash = hashlib.md5(f"{args_str}{kwargs_str}".encode()).hexdigest()[:16]
    
    return f"{prefix}:{func_name}:{params_hash}"


class RequestCache:
    """
    请求级缓存
    在单个请求生命周期内缓存数据，避免重复计算
    """
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        """设置缓存值"""
        self._cache[key] = value
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return key in self._cache
    
    def clear(self):
        """清除所有缓存"""
        self._cache.clear()
    
    def get_or_set(self, key: str, loader: Callable) -> Any:
        """getOrSet模式"""
        if key in self._cache:
            return self._cache[key]
        value = loader()
        self._cache[key] = value
        return value


# 全局请求级缓存实例
_request_cache = RequestCache()


def get_request_cache() -> RequestCache:
    """获取请求级缓存实例"""
    return _request_cache


def clear_request_cache():
    """清除请求级缓存"""
    _request_cache.clear()
