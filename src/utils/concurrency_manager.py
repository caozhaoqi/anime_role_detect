#!/usr/bin/env python3
"""
并发管理器类，实现动态并发数调整
"""
import os
import psutil
import logging
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ConcurrencyManager:
    """并发管理器类"""
    
    def __init__(self, min_workers=2, max_workers=10, check_interval=5):
        """
        初始化并发管理器
        
        Args:
            min_workers: 最小并发数
            max_workers: 最大并发数
            check_interval: 检查间隔（秒）
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.check_interval = check_interval
        self.current_workers = min_workers
        self.last_check_time = time.time()
        self.executor = None
        self.resource_stats = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'network_sent': 0,
            'network_recv': 0
        }
    
    def _get_system_resources(self):
        """
        获取系统资源使用情况
        
        Returns:
            dict: 系统资源使用情况
        """
        # 获取CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 获取内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # 获取网络使用情况
        net_io = psutil.net_io_counters()
        network_sent = net_io.bytes_sent
        network_recv = net_io.bytes_recv
        
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'network_sent': network_sent,
            'network_recv': network_recv
        }
    
    def _calculate_optimal_workers(self):
        """
        计算最优并发数
        
        Returns:
            int: 最优并发数
        """
        resources = self._get_system_resources()
        self.resource_stats.update(resources)
        
        # 基于系统资源计算最优并发数
        cpu_usage = resources['cpu_usage']
        memory_usage = resources['memory_usage']
        
        # 初始化最优并发数为当前值
        optimal_workers = self.current_workers
        
        # 根据CPU使用率调整
        if cpu_usage < 50 and self.current_workers < self.max_workers:
            # CPU使用率低，可以增加并发数
            optimal_workers = min(self.current_workers + 1, self.max_workers)
            logger.info(f"CPU使用率低 ({cpu_usage}%), 增加并发数到: {optimal_workers}")
        elif cpu_usage > 80 and self.current_workers > self.min_workers:
            # CPU使用率高，需要减少并发数
            optimal_workers = max(self.current_workers - 1, self.min_workers)
            logger.info(f"CPU使用率高 ({cpu_usage}%), 减少并发数到: {optimal_workers}")
        
        # 根据内存使用率调整
        if memory_usage > 80 and self.current_workers > self.min_workers:
            # 内存使用率高，需要减少并发数
            optimal_workers = max(optimal_workers - 1, self.min_workers)
            logger.info(f"内存使用率高 ({memory_usage}%), 减少并发数到: {optimal_workers}")
        
        return optimal_workers
    
    def get_executor(self):
        """
        获取线程池执行器
        
        Returns:
            ThreadPoolExecutor: 线程池执行器
        """
        # 检查是否需要调整并发数
        current_time = time.time()
        if current_time - self.last_check_time > self.check_interval:
            optimal_workers = self._calculate_optimal_workers()
            
            # 如果最优并发数发生变化，重新创建执行器
            if optimal_workers != self.current_workers:
                logger.info(f"调整并发数: {self.current_workers} -> {optimal_workers}")
                
                # 关闭旧的执行器
                if self.executor:
                    self.executor.shutdown(wait=False)
                
                # 创建新的执行器
                self.executor = ThreadPoolExecutor(max_workers=optimal_workers)
                self.current_workers = optimal_workers
            
            self.last_check_time = current_time
        
        # 如果执行器不存在，创建一个
        if not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=self.current_workers)
            logger.info(f"创建线程池执行器，初始并发数: {self.current_workers}")
        
        return self.executor
    
    def submit(self, func, *args, **kwargs):
        """
        提交任务到线程池
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            Future: 任务的Future对象
        """
        executor = self.get_executor()
        return executor.submit(func, *args, **kwargs)
    
    def shutdown(self, wait=True):
        """
        关闭线程池执行器
        
        Args:
            wait: 是否等待所有任务完成
        """
        if self.executor:
            logger.info(f"关闭线程池执行器，当前并发数: {self.current_workers}")
            self.executor.shutdown(wait=wait)
            self.executor = None
    
    def get_current_workers(self):
        """
        获取当前并发数
        
        Returns:
            int: 当前并发数
        """
        return self.current_workers
    
    def get_resource_stats(self):
        """
        获取资源统计信息
        
        Returns:
            dict: 资源统计信息
        """
        return self.resource_stats
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器出口
        """
        self.shutdown()


# 全局并发管理器实例
concurrency_manager = ConcurrencyManager()
