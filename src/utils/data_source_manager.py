#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据源管理模块

管理多个数据源，实现智能调度和质量评估
"""

import time
import random
import logging
from abc import ABC, abstractmethod
from collections import defaultdict

from src.utils.http_utils import HTTPUtils
from src.utils.config_manager import config_manager

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """
    数据源抽象基类
    """
    
    def __init__(self, name, priority=1, rate_limit=1.0):
        """
        初始化数据源
        
        Args:
            name: 数据源名称
            priority: 优先级 (1-10, 数字越小优先级越高)
            rate_limit: 速率限制 (请求/秒)
        """
        self.name = name
        self.priority = priority
        self.rate_limit = rate_limit
        self.http_utils = HTTPUtils()
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0,
            'last_request_time': 0,
            'consecutive_failures': 0,
            'quality_score': 0.0  # 0-100
        }
    
    @abstractmethod
    def fetch_images(self, query, limit=50):
        """
        获取图像
        
        Args:
            query: 搜索查询
            limit: 获取数量
            
        Returns:
            图像列表 [{url, width, height, source}]
        """
        pass
    
    def calculate_quality_score(self):
        """
        计算数据源质量分数
        
        Returns:
            质量分数 (0-100)
        """
        if self.stats['total_requests'] == 0:
            return 50.0  # 默认分数
        
        success_rate = self.stats['successful_requests'] / self.stats['total_requests']
        response_time_factor = max(0, 1 - (self.stats['avg_response_time'] / 5))  # 5秒以上响应时间视为差
        failure_factor = max(0, 1 - (self.stats['consecutive_failures'] / 10))  # 连续失败10次以上视为差
        
        score = (success_rate * 60) + (response_time_factor * 20) + (failure_factor * 20)
        self.stats['quality_score'] = min(100, max(0, score))
        return self.stats['quality_score']
    
    def can_request(self):
        """
        检查是否可以发送请求（基于速率限制）
        
        Returns:
            是否可以发送请求
        """
        current_time = time.time()
        time_since_last = current_time - self.stats['last_request_time']
        return time_since_last >= (1 / self.rate_limit)
    
    def wait_for_rate_limit(self):
        """
        等待速率限制
        """
        current_time = time.time()
        time_since_last = current_time - self.stats['last_request_time']
        required_wait = max(0, (1 / self.rate_limit) - time_since_last)
        if required_wait > 0:
            time.sleep(required_wait)
    
    def update_stats(self, success, response_time=0):
        """
        更新统计信息
        
        Args:
            success: 是否成功
            response_time: 响应时间
        """
        self.stats['total_requests'] += 1
        self.stats['last_request_time'] = time.time()
        
        if success:
            self.stats['successful_requests'] += 1
            self.stats['consecutive_failures'] = 0
            # 更新平均响应时间
            if self.stats['avg_response_time'] == 0:
                self.stats['avg_response_time'] = response_time
            else:
                self.stats['avg_response_time'] = (
                    self.stats['avg_response_time'] * 0.9 + response_time * 0.1
                )
        else:
            self.stats['failed_requests'] += 1
            self.stats['consecutive_failures'] += 1
        
        # 重新计算质量分数
        self.calculate_quality_score()


class SafebooruDataSource(DataSource):
    """
    Safebooru数据源
    """
    
    def __init__(self):
        super().__init__('safebooru', priority=2, rate_limit=0.5)  # 2秒一次请求
    
    def fetch_images(self, query, limit=50):
        """
        从Safebooru获取图像
        """
        base_url = 'https://safebooru.org/index.php'
        images = []
        start_time = time.time()
        
        try:
            self.wait_for_rate_limit()
            
            params = {
                'page': 'dapi',
                's': 'post',
                'q': 'index',
                'tags': query,
                'limit': limit,
                'json': '1'
            }
            
            response = self.http_utils.get(base_url, params=params)
            response_time = time.time() - start_time
            
            if not response.text:
                self.update_stats(False, response_time)
                return images
            
            data = response.json()
            
            if isinstance(data, dict) and 'posts' in data:
                data = data['posts']
            
            for post in data:
                if post.get('file_url'):
                    images.append({
                        'url': post['file_url'],
                        'width': post.get('width', 0),
                        'height': post.get('height', 0),
                        'source': self.name
                    })
            
            self.update_stats(True, response_time)
            logger.info(f"从Safebooru获取了 {len(images)} 张图像，查询: {query}")
        except Exception as e:
            response_time = time.time() - start_time
            self.update_stats(False, response_time)
            logger.error(f"从Safebooru获取图像失败，查询: {query}, 错误: {e}")
        
        return images


class WaifuPicsDataSource(DataSource):
    """
    Waifu.pics数据源
    """
    
    def __init__(self):
        super().__init__('waifu_pics', priority=3, rate_limit=1.0)  # 1秒一次请求
    
    def fetch_images(self, query, limit=50):
        """
        从Waifu.pics获取图像
        """
        base_url = 'https://api.waifu.pics/sfw/waifu'
        images = []
        start_time = time.time()
        
        try:
            for _ in range(min(limit, 10)):  # 每次最多获取10张
                self.wait_for_rate_limit()
                
                response = self.http_utils.get(base_url)
                response_time = time.time() - start_time
                
                data = response.json()
                if data.get('url'):
                    images.append({
                        'url': data['url'],
                        'width': 0,
                        'height': 0,
                        'source': self.name
                    })
            
            self.update_stats(True, response_time)
            logger.info(f"从Waifu.pics获取了 {len(images)} 张图像")
        except Exception as e:
            response_time = time.time() - start_time
            self.update_stats(False, response_time)
            logger.error(f"从Waifu.pics获取图像失败: {e}")
        
        return images


class DanbooruDataSource(DataSource):
    """
    Danbooru数据源
    """
    
    def __init__(self):
        super().__init__('danbooru', priority=1, rate_limit=0.3)  # 3秒一次请求
    
    def fetch_images(self, query, limit=50):
        """
        从Danbooru获取图像
        """
        base_url = 'https://danbooru.donmai.us/posts.json'
        images = []
        start_time = time.time()
        
        try:
            self.wait_for_rate_limit()
            
            params = {
                'tags': query,
                'limit': limit,
                'random': 'true'
            }
            
            response = self.http_utils.get(base_url, params=params)
            response_time = time.time() - start_time
            
            if not response.text:
                self.update_stats(False, response_time)
                return images
            
            data = response.json()
            
            for post in data:
                if post.get('file_url'):
                    images.append({
                        'url': post['file_url'],
                        'width': post.get('width', 0),
                        'height': post.get('height', 0),
                        'source': self.name
                    })
            
            self.update_stats(True, response_time)
            logger.info(f"从Danbooru获取了 {len(images)} 张图像，查询: {query}")
        except Exception as e:
            response_time = time.time() - start_time
            self.update_stats(False, response_time)
            logger.error(f"从Danbooru获取图像失败，查询: {query}, 错误: {e}")
        
        return images


class DataSourceManager:
    """
    数据源管理器
    """
    
    def __init__(self):
        """
        初始化数据源管理器
        """
        self.data_sources = {
            'safebooru': SafebooruDataSource(),
            'waifu_pics': WaifuPicsDataSource(),
            'danbooru': DanbooruDataSource()
        }
        self.historical_performance = defaultdict(list)
    
    def get_sorted_data_sources(self):
        """
        获取排序后的数据源列表（基于质量分数和优先级）
        
        Returns:
            排序后的数据源列表
        """
        # 计算所有数据源的质量分数
        for source in self.data_sources.values():
            source.calculate_quality_score()
        
        # 排序：质量分数 * 0.7 + 优先级 * 0.3
        sorted_sources = sorted(
            self.data_sources.values(),
            key=lambda x: (x.stats['quality_score'] * 0.7) + ((11 - x.priority) * 3),
            reverse=True
        )
        
        return sorted_sources
    
    def fetch_images(self, query, limit=50, max_sources=3):
        """
        从多个数据源获取图像
        
        Args:
            query: 搜索查询
            limit: 获取数量
            max_sources: 最大尝试数据源数量
            
        Returns:
            图像列表
        """
        all_images = []
        tried_sources = 0
        
        # 获取排序后的数据源
        sorted_sources = self.get_sorted_data_sources()
        
        for source in sorted_sources:
            if len(all_images) >= limit or tried_sources >= max_sources:
                break
            
            try:
                # 计算还需要的图像数量
                needed = limit - len(all_images)
                
                # 从数据源获取图像
                source_images = source.fetch_images(query, limit=needed)
                
                # 添加到结果列表
                all_images.extend(source_images)
                tried_sources += 1
                
                # 短暂延迟，避免请求过快
                time.sleep(random.uniform(0.5, 1.0))
                
            except Exception as e:
                logger.error(f"使用数据源 {source.name} 获取图像失败: {e}")
                tried_sources += 1
        
        # 去重
        seen_urls = set()
        unique_images = []
        for img in all_images:
            if img['url'] not in seen_urls:
                seen_urls.add(img['url'])
                unique_images.append(img)
        
        return unique_images[:limit]
    
    def get_data_source_stats(self):
        """
        获取数据源统计信息
        
        Returns:
            统计信息字典
        """
        stats = {}
        for name, source in self.data_sources.items():
            source.calculate_quality_score()
            stats[name] = {
                'priority': source.priority,
                'quality_score': source.stats['quality_score'],
                'success_rate': source.stats['successful_requests'] / source.stats['total_requests'] if source.stats['total_requests'] > 0 else 0,
                'avg_response_time': source.stats['avg_response_time'],
                'consecutive_failures': source.stats['consecutive_failures'],
                'total_requests': source.stats['total_requests']
            }
        return stats
    
    def add_data_source(self, name, data_source):
        """
        添加自定义数据源
        
        Args:
            name: 数据源名称
            data_source: 数据源实例
        """
        self.data_sources[name] = data_source
        logger.info(f"添加了新的数据源: {name}")
    
    def remove_data_source(self, name):
        """
        移除数据源
        
        Args:
            name: 数据源名称
        """
        if name in self.data_sources:
            del self.data_sources[name]
            logger.info(f"移除了数据源: {name}")
    
    def update_source_priority(self, name, priority):
        """
        更新数据源优先级
        
        Args:
            name: 数据源名称
            priority: 新优先级 (1-10)
        """
        if name in self.data_sources:
            self.data_sources[name].priority = max(1, min(10, priority))
            logger.info(f"更新了数据源 {name} 的优先级为: {priority}")
