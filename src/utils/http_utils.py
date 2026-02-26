#!/usr/bin/env python3
"""
HTTP工具类，封装网络请求逻辑
"""
import requests
import time
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache

from src.utils.cache_manager import cache_manager

logger = logging.getLogger(__name__)


class HTTPUtils:
    """HTTP工具类"""
    
    def __init__(self, max_retries=3, backoff_factor=0.5, timeout=15, monitoring_system=None):
        """
        初始化HTTP工具
        
        Args:
            max_retries: 最大重试次数
            backoff_factor: 退避因子
            timeout: 超时时间
            monitoring_system: 监控系统实例
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.monitoring_system = monitoring_system
        self.session = self._create_session()
    
    def _create_session(self):
        """创建带有重试机制的会话"""
        session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"],
            backoff_factor=self.backoff_factor
        )
        
        # 配置适配器
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 默认请求头
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        return session
    
    def get(self, url, params=None, headers=None, cache_enabled=True, cache_ttl=3600, **kwargs):
        """发送GET请求"""
        start_time = time.time()
        
        # 生成缓存键
        cache_key = None
        if cache_enabled:
            cache_key = cache_manager._generate_cache_key('http_get', url, params, headers)
            
            # 尝试从缓存获取
            cached_response = cache_manager.get(cache_key)
            if cached_response:
                logger.info(f"从缓存获取GET请求结果: {url}")
                
                # 记录成功请求
                if self.monitoring_system:
                    response_time = time.time() - start_time
                    self.monitoring_system.update_network_stats(True, response_time)
                
                return cached_response
        
        try:
            logger.info(f"发送GET请求: {url}")
            response = self.session.get(
                url, 
                params=params, 
                headers=headers, 
                timeout=self.timeout, 
                **kwargs
            )
            response.raise_for_status()
            logger.info(f"GET请求成功: {url}")
            
            # 存入缓存
            if cache_enabled:
                cache_manager.set(response, cache_key, cache_ttl)
                logger.debug(f"缓存GET请求结果: {url}, TTL: {cache_ttl}秒")
            
            # 记录成功请求
            if self.monitoring_system:
                response_time = time.time() - start_time
                self.monitoring_system.update_network_stats(True, response_time)
                
            return response
        except requests.RequestException as e:
            logger.error(f"GET请求失败: {url}, 错误: {e}")
            
            # 记录失败请求
            if self.monitoring_system:
                response_time = time.time() - start_time
                self.monitoring_system.update_network_stats(False, response_time)
                
            raise
    
    def post(self, url, data=None, json=None, headers=None, **kwargs):
        """发送POST请求"""
        start_time = time.time()
        try:
            logger.info(f"发送POST请求: {url}")
            response = self.session.post(
                url, 
                data=data, 
                json=json, 
                headers=headers, 
                timeout=self.timeout, 
                **kwargs
            )
            response.raise_for_status()
            logger.info(f"POST请求成功: {url}")
            
            # 记录成功请求
            if self.monitoring_system:
                response_time = time.time() - start_time
                self.monitoring_system.update_network_stats(True, response_time)
                
            return response
        except requests.RequestException as e:
            logger.error(f"POST请求失败: {url}, 错误: {e}")
            
            # 记录失败请求
            if self.monitoring_system:
                response_time = time.time() - start_time
                self.monitoring_system.update_network_stats(False, response_time)
                
            raise
    
    @lru_cache(maxsize=100)
    def get_cached(self, url, params=None, headers=None, **kwargs):
        """发送带缓存的GET请求"""
        return self.get(url, params, headers, **kwargs)
    
    def download_file(self, url, timeout=30, cache_enabled=True, cache_ttl=86400):
        """下载文件"""
        start_time = time.time()
        
        # 生成缓存键
        cache_key = None
        if cache_enabled:
            cache_key = cache_manager._generate_cache_key('download_file', url)
            
            # 尝试从缓存获取
            cached_content = cache_manager.get(cache_key)
            if cached_content:
                logger.info(f"从缓存获取文件: {url}")
                
                # 记录成功请求
                if self.monitoring_system:
                    response_time = time.time() - start_time
                    self.monitoring_system.update_network_stats(True, response_time)
                
                return cached_content
        
        try:
            logger.info(f"下载文件: {url}")
            response = self.session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('Content-Type', '')
            logger.info(f"文件内容类型: {content_type}")
            
            content = response.content
            
            # 存入缓存
            if cache_enabled:
                cache_manager.set(content, cache_key, cache_ttl)
                logger.debug(f"缓存文件: {url}, TTL: {cache_ttl}秒")
            
            # 记录成功请求
            if self.monitoring_system:
                response_time = time.time() - start_time
                self.monitoring_system.update_network_stats(True, response_time)
                
            return content
        except requests.RequestException as e:
            logger.error(f"下载文件失败: {url}, 错误: {e}")
            
            # 记录失败请求
            if self.monitoring_system:
                response_time = time.time() - start_time
                self.monitoring_system.update_network_stats(False, response_time)
                
            raise
    
    def close(self):
        """关闭会话"""
        if hasattr(self, 'session'):
            self.session.close()
            logger.info("HTTP会话已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 全局HTTP工具实例
http_utils = HTTPUtils()
