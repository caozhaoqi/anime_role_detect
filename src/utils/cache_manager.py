#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存管理系统

实现内存缓存和文件缓存，减少重复请求和提高响应速度
"""

import os
import json
import logging
import pickle
import time
import threading
from functools import lru_cache
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CacheManager:
    """
    缓存管理器
    """
    
    def __init__(self, cache_dir='./cache', max_memory_size=1000, max_file_size=10000, default_ttl=3600):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存文件目录
            max_memory_size: 内存缓存最大条目数
            max_file_size: 文件缓存最大条目数
            default_ttl: 默认缓存过期时间（秒）
        """
        self.cache_dir = cache_dir
        self.max_memory_size = max_memory_size
        self.max_file_size = max_file_size
        self.default_ttl = default_ttl
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 内存缓存（使用OrderedDict实现LRU）
        self.memory_cache = OrderedDict()
        
        # 文件缓存索引
        self.file_cache_index = {}
        self.file_cache_stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'evictions': 0
        }
        
        # 锁
        self.lock = threading.RLock()
        
        # 加载文件缓存索引
        self._load_file_cache_index()
        
        logger.info(f"初始化缓存管理器，内存缓存大小: {max_memory_size}, 文件缓存大小: {max_file_size}")
    
    def _load_file_cache_index(self):
        """
        加载文件缓存索引
        """
        index_file = os.path.join(self.cache_dir, 'cache_index.json')
        try:
            if os.path.exists(index_file):
                with open(index_file, 'r', encoding='utf-8') as f:
                    self.file_cache_index = json.load(f)
                logger.info(f"加载文件缓存索引，包含 {len(self.file_cache_index)} 个条目")
        except Exception as e:
            logger.error(f"加载文件缓存索引失败: {e}")
            self.file_cache_index = {}
    
    def _save_file_cache_index(self):
        """
        保存文件缓存索引
        """
        index_file = os.path.join(self.cache_dir, 'cache_index.json')
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(self.file_cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存文件缓存索引失败: {e}")
    
    def _generate_cache_key(self, *args, **kwargs):
        """
        生成缓存键
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            str: 缓存键
        """
        # 构建键的组成部分
        key_parts = []
        
        # 添加位置参数
        for arg in args:
            if isinstance(arg, (str, int, float, bool, type(None))):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                key_parts.append('_'.join(str(item) for item in arg))
            elif isinstance(arg, dict):
                sorted_items = sorted(arg.items())
                key_parts.append('_'.join(f"{k}={v}" for k, v in sorted_items))
            else:
                key_parts.append(str(hash(arg)))
        
        # 添加关键字参数
        sorted_kwargs = sorted(kwargs.items())
        for k, v in sorted_kwargs:
            if isinstance(v, (str, int, float, bool, type(None))):
                key_parts.append(f"{k}={v}")
            elif isinstance(v, (list, tuple)):
                key_parts.append(f"{k}={'_'.join(str(item) for item in v)}")
            elif isinstance(v, dict):
                sorted_items = sorted(v.items())
                key_parts.append(f"{k}={'_'.join(f'{kk}={vv}' for kk, vv in sorted_items)}")
            else:
                key_parts.append(f"{k}={hash(v)}")
        
        # 生成最终的键
        key = '_'.join(key_parts)
        return key
    
    def _get_cache_file_path(self, key):
        """
        获取缓存文件路径
        
        Args:
            key: 缓存键
            
        Returns:
            str: 缓存文件路径
        """
        # 使用哈希值作为文件名，避免文件名过长
        import hashlib
        file_name = hashlib.md5(key.encode('utf-8')).hexdigest() + '.pkl'
        return os.path.join(self.cache_dir, file_name)
    
    def get(self, key=None, *args, **kwargs):
        """
        获取缓存
        
        Args:
            key: 缓存键，如果为None则根据args和kwargs生成
            *args: 位置参数（用于生成缓存键）
            **kwargs: 关键字参数（用于生成缓存键）
            
        Returns:
            缓存值或None
        """
        with self.lock:
            # 生成缓存键
            if key is None:
                key = self._generate_cache_key(*args, **kwargs)
            
            # 先检查内存缓存
            if key in self.memory_cache:
                item = self.memory_cache[key]
                
                # 检查是否过期
                if not self._is_expired(item):
                    # 更新访问时间（LRU）
                    self.memory_cache.move_to_end(key)
                    self.file_cache_stats['hits'] += 1
                    logger.debug(f"内存缓存命中: {key}")
                    return item['value']
                else:
                    # 过期，删除
                    del self.memory_cache[key]
                    logger.debug(f"内存缓存过期: {key}")
            
            # 检查文件缓存
            if key in self.file_cache_index:
                cache_info = self.file_cache_index[key]
                
                # 检查是否过期
                if not self._is_expired(cache_info):
                    # 读取文件缓存
                    cache_file = self._get_cache_file_path(key)
                    try:
                        with open(cache_file, 'rb') as f:
                            value = pickle.load(f)
                        
                        # 更新访问时间
                        cache_info['last_accessed'] = datetime.now().isoformat()
                        self._save_file_cache_index()
                        
                        # 添加到内存缓存
                        self._add_to_memory_cache(key, value, cache_info['ttl'])
                        
                        self.file_cache_stats['hits'] += 1
                        logger.debug(f"文件缓存命中: {key}")
                        return value
                    except Exception as e:
                        logger.error(f"读取文件缓存失败 {cache_file}: {e}")
                        # 删除损坏的缓存
                        del self.file_cache_index[key]
                        self._save_file_cache_index()
                        if os.path.exists(cache_file):
                            os.remove(cache_file)
            
            # 缓存未命中
            self.file_cache_stats['misses'] += 1
            logger.debug(f"缓存未命中: {key}")
            return None
    
    def set(self, value, key=None, ttl=None, *args, **kwargs):
        """
        设置缓存
        
        Args:
            value: 缓存值
            key: 缓存键，如果为None则根据args和kwargs生成
            ttl: 缓存过期时间（秒），如果为None则使用默认值
            *args: 位置参数（用于生成缓存键）
            **kwargs: 关键字参数（用于生成缓存键）
            
        Returns:
            str: 缓存键
        """
        with self.lock:
            # 生成缓存键
            if key is None:
                key = self._generate_cache_key(*args, **kwargs)
            
            # 设置TTL
            if ttl is None:
                ttl = self.default_ttl
            
            # 添加到内存缓存
            self._add_to_memory_cache(key, value, ttl)
            
            # 添加到文件缓存
            self._add_to_file_cache(key, value, ttl)
            
            logger.debug(f"设置缓存: {key}, TTL: {ttl}秒")
            return key
    
    def _add_to_memory_cache(self, key, value, ttl):
        """
        添加到内存缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 缓存过期时间（秒）
        """
        # 检查内存缓存大小
        if len(self.memory_cache) >= self.max_memory_size:
            # 移除最不常用的条目（LRU）
            oldest_key, _ = self.memory_cache.popitem(last=False)
            self.file_cache_stats['evictions'] += 1
            logger.debug(f"内存缓存溢出，移除: {oldest_key}")
        
        # 添加到内存缓存
        self.memory_cache[key] = {
            'value': value,
            'created': datetime.now().isoformat(),
            'ttl': ttl
        }
    
    def _add_to_file_cache(self, key, value, ttl):
        """
        添加到文件缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 缓存过期时间（秒）
        """
        # 检查文件缓存大小
        if len(self.file_cache_index) >= self.max_file_size:
            # 移除最不常用的条目
            oldest_key = min(self.file_cache_index.items(), 
                            key=lambda x: x[1].get('last_accessed', x[1]['created']))[0]
            self._remove_from_file_cache(oldest_key)
            self.file_cache_stats['evictions'] += 1
            logger.debug(f"文件缓存溢出，移除: {oldest_key}")
        
        # 写入文件缓存
        cache_file = self._get_cache_file_path(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # 更新缓存索引
            now = datetime.now().isoformat()
            self.file_cache_index[key] = {
                'created': now,
                'last_accessed': now,
                'ttl': ttl,
                'file': os.path.basename(cache_file)
            }
            
            # 保存索引
            self._save_file_cache_index()
            
            self.file_cache_stats['writes'] += 1
        except Exception as e:
            logger.error(f"写入文件缓存失败 {cache_file}: {e}")
            if os.path.exists(cache_file):
                os.remove(cache_file)
    
    def _remove_from_file_cache(self, key):
        """
        从文件缓存中移除
        
        Args:
            key: 缓存键
        """
        if key in self.file_cache_index:
            cache_info = self.file_cache_index[key]
            cache_file = os.path.join(self.cache_dir, cache_info['file'])
            
            # 删除缓存文件
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except Exception as e:
                    logger.error(f"删除缓存文件失败 {cache_file}: {e}")
            
            # 从索引中删除
            del self.file_cache_index[key]
            self._save_file_cache_index()
    
    def delete(self, key=None, *args, **kwargs):
        """
        删除缓存
        
        Args:
            key: 缓存键，如果为None则根据args和kwargs生成
            *args: 位置参数（用于生成缓存键）
            **kwargs: 关键字参数（用于生成缓存键）
        """
        with self.lock:
            # 生成缓存键
            if key is None:
                key = self._generate_cache_key(*args, **kwargs)
            
            # 从内存缓存中删除
            if key in self.memory_cache:
                del self.memory_cache[key]
                logger.debug(f"从内存缓存中删除: {key}")
            
            # 从文件缓存中删除
            if key in self.file_cache_index:
                self._remove_from_file_cache(key)
                logger.debug(f"从文件缓存中删除: {key}")
    
    def clear(self):
        """
        清除所有缓存
        """
        with self.lock:
            # 清除内存缓存
            self.memory_cache.clear()
            
            # 清除文件缓存
            for key in list(self.file_cache_index.keys()):
                self._remove_from_file_cache(key)
            
            # 清除统计信息
            self.file_cache_stats = {
                'hits': 0,
                'misses': 0,
                'writes': 0,
                'evictions': 0
            }
            
            logger.info("清除所有缓存")
    
    def _is_expired(self, item):
        """
        检查缓存是否过期
        
        Args:
            item: 缓存项
            
        Returns:
            bool: 是否已过期
        """
        created = datetime.fromisoformat(item['created'])
        ttl = item['ttl']
        return datetime.now() - created > timedelta(seconds=ttl)
    
    def get_stats(self):
        """
        获取缓存统计信息
        
        Returns:
            dict: 统计信息
        """
        with self.lock:
            return {
                'memory_cache_size': len(self.memory_cache),
                'file_cache_size': len(self.file_cache_index),
                'max_memory_size': self.max_memory_size,
                'max_file_size': self.max_file_size,
                'cache_dir': self.cache_dir,
                'stats': self.file_cache_stats.copy(),
                'hit_rate': self._calculate_hit_rate()
            }
    
    def _calculate_hit_rate(self):
        """
        计算缓存命中率
        
        Returns:
            float: 命中率
        """
        total = self.file_cache_stats['hits'] + self.file_cache_stats['misses']
        if total == 0:
            return 0.0
        return self.file_cache_stats['hits'] / total * 100
    
    def decorator(self, ttl=None):
        """
        缓存装饰器
        
        Args:
            ttl: 缓存过期时间（秒）
            
        Returns:
            装饰器函数
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 生成缓存键
                key = self._generate_cache_key(func.__name__, *args, **kwargs)
                
                # 尝试从缓存获取
                cached_value = self.get(key)
                if cached_value is not None:
                    logger.debug(f"装饰器缓存命中: {func.__name__}")
                    return cached_value
                
                # 执行函数
                value = func(*args, **kwargs)
                
                # 存入缓存
                self.set(value, key, ttl)
                logger.debug(f"装饰器缓存设置: {func.__name__}")
                
                return value
            return wrapper
        return decorator
    
    def cleanup(self):
        """
        清理过期缓存
        """
        with self.lock:
            # 清理内存缓存
            expired_keys = []
            for key, item in self.memory_cache.items():
                if self._is_expired(item):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
                logger.debug(f"清理过期内存缓存: {key}")
            
            # 清理文件缓存
            expired_keys = []
            for key, cache_info in self.file_cache_index.items():
                if self._is_expired(cache_info):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_from_file_cache(key)
                logger.debug(f"清理过期文件缓存: {key}")
            
            if expired_keys:
                logger.info(f"清理了 {len(expired_keys)} 个过期缓存")


# 全局缓存管理器实例
cache_manager = CacheManager()


# 便捷函数
def get_cache(key=None, *args, **kwargs):
    """
    获取缓存
    """
    return cache_manager.get(key, *args, **kwargs)


def set_cache(value, key=None, ttl=None, *args, **kwargs):
    """
    设置缓存
    """
    return cache_manager.set(value, key, ttl, *args, **kwargs)


def delete_cache(key=None, *args, **kwargs):
    """
    删除缓存
    """
    return cache_manager.delete(key, *args, **kwargs)


def clear_cache():
    """
    清除所有缓存
    """
    return cache_manager.clear()


def get_cache_stats():
    """
    获取缓存统计信息
    """
    return cache_manager.get_stats()


def cache(ttl=None):
    """
    缓存装饰器
    """
    return cache_manager.decorator(ttl)


def cleanup_cache():
    """
    清理过期缓存
    """
    return cache_manager.cleanup()
