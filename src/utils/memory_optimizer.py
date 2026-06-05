#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化工具
提供批量推理、流式处理和内存监控功能
"""

import os
import gc
import sys
import time
import psutil
import threading
from typing import List, Optional, Union, Iterator, Callable
from pathlib import Path
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger("memory_optimizer")


class MemoryMonitor:
    """
    内存监控器
    实时监控内存使用情况
    """
    
    def __init__(self, interval: float = 1.0, threshold: float = 0.85):
        """
        初始化
        
        Args:
            interval: 监控间隔（秒）
            threshold: 内存使用阈值（百分比）
        """
        self.interval = interval
        self.threshold = threshold
        self.monitoring = False
        self.thread = None
        self.peak_memory = 0
        
    def start(self):
        """开始监控"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("内存监控已启动")
    
    def stop(self):
        """停止监控"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info(f"内存监控已停止，峰值: {self.peak_memory:.2f}MB")
    
    def _monitor_loop(self):
        """监控循环"""
        process = psutil.Process(os.getpid())
        
        while self.monitoring:
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > self.peak_memory:
                self.peak_memory = memory_mb
            
            # 检查是否超过阈值
            system_memory = psutil.virtual_memory()
            if system_memory.percent > self.threshold * 100:
                logger.warning(f"内存使用过高: {system_memory.percent:.1f}%, 触发GC")
                gc.collect()
            
            time.sleep(self.interval)
    
    def get_current_memory(self) -> float:
        """获取当前内存使用（MB）"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def get_peak_memory(self) -> float:
        """获取峰值内存使用（MB）"""
        return self.peak_memory


class StreamingProcessor:
    """
    流式处理器
    支持大文件的流式处理，避免一次性加载到内存
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        buffer_size: int = 100,
    ):
        """
        初始化
        
        Args:
            batch_size: 批处理大小
            buffer_size: 缓冲区大小
        """
        self.batch_size = batch_size
        self.buffer_size = buffer_size
    
    def process_image_stream(
        self,
        image_paths: Iterator[str],
        process_func: Callable[[List[str]], List],
    ) -> Iterator:
        """
        流式处理图片
        
        Args:
            image_paths: 图片路径迭代器
            process_func: 处理函数
            
        Yields:
            处理结果
        """
        batch = []
        
        for path in image_paths:
            batch.append(path)
            
            if len(batch) >= self.batch_size:
                results = process_func(batch)
                for result in results:
                    yield result
                
                # 清空批次
                batch = []
                
                # 触发GC
                if len(batch) % (self.batch_size * 10) == 0:
                    gc.collect()
        
        # 处理剩余
        if batch:
            results = process_func(batch)
            for result in results:
                yield result
    
    def process_with_memory_limit(
        self,
        items: List,
        process_func: Callable[[List], List],
        memory_limit_mb: float = 4096,
    ) -> List:
        """
        在内存限制下处理
        
        Args:
            items: 待处理项列表
            process_func: 处理函数
            memory_limit_mb: 内存限制（MB）
            
        Returns:
            处理结果列表
        """
        results = []
        batch = []
        
        monitor = MemoryMonitor()
        current_memory = monitor.get_current_memory()
        
        for item in items:
            batch.append(item)
            
            # 检查内存
            if len(batch) >= self.batch_size:
                current_memory = monitor.get_current_memory()
                
                if current_memory > memory_limit_mb:
                    logger.warning(f"内存接近限制: {current_memory:.0f}MB, 减小批次")
                    # 处理当前批次
                    results.extend(process_func(batch))
                    batch = []
                    gc.collect()
                else:
                    results.extend(process_func(batch))
                    batch = []
        
        # 处理剩余
        if batch:
            results.extend(process_func(batch))
        
        return results


class BatchInferenceOptimizer:
    """
    批量推理优化器
    优化批量推理的内存使用
    """
    
    def __init__(
        self,
        embedder,
        optimal_batch_size: int = 16,
        max_batch_size: int = 64,
    ):
        """
        初始化
        
        Args:
            embedder: 特征提取器
            optimal_batch_size: 最优批大小
            max_batch_size: 最大批大小
        """
        self.embedder = embedder
        self.optimal_batch_size = optimal_batch_size
        self.max_batch_size = max_batch_size
        
    def infer_with_auto_batch(
        self,
        image_paths: List[str],
        target_memory_mb: float = 2048,
    ) -> List[Optional[np.ndarray]]:
        """
        自动调整批大小进行推理
        
        Args:
            image_paths: 图片路径列表
            target_memory_mb: 目标内存使用（MB）
            
        Returns:
            特征列表
        """
        monitor = MemoryMonitor()
        results = []
        
        batch_size = self.optimal_batch_size
        i = 0
        
        while i < len(image_paths):
            batch = image_paths[i:i+batch_size]
            
            try:
                # 推理
                batch_results = self.embedder.embed_images(batch, batch_size=len(batch))
                results.extend(batch_results)
                
                # 检查内存
                current_memory = monitor.get_current_memory()
                
                if current_memory > target_memory_mb and batch_size > 1:
                    # 内存过高，减小批大小
                    batch_size = max(1, batch_size // 2)
                    logger.info(f"内存优化: 批大小调整为 {batch_size}")
                elif current_memory < target_memory_mb * 0.5 and batch_size < self.max_batch_size:
                    # 内存充裕，增大批大小
                    batch_size = min(self.max_batch_size, batch_size * 2)
                
                i += len(batch)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM错误，减小批大小")
                    batch_size = max(1, batch_size // 2)
                    gc.collect()
                    
                    if batch_size == 1:
                        # 单张也OOM，跳过
                        logger.error(f"单张图片OOM，跳过: {batch[0]}")
                        results.append(None)
                        i += 1
                else:
                    raise
        
        return results
    
    def infer_with_prefetch(
        self,
        image_paths: List[str],
        prefetch_factor: int = 2,
    ) -> List[Optional[np.ndarray]]:
        """
        预取推理
        
        Args:
            image_paths: 图片路径列表
            prefetch_factor: 预取因子
            
        Returns:
            特征列表
        """
        from concurrent.futures import ThreadPoolExecutor
        
        results = [None] * len(image_paths)
        
        def load_and_infer(idx_batch):
            indices, paths = idx_batch
            features = self.embedder.embed_images(paths, batch_size=len(paths))
            return indices, features
        
        # 分批处理
        batches = []
        for i in range(0, len(image_paths), self.optimal_batch_size):
            indices = list(range(i, min(i + self.optimal_batch_size, len(image_paths))))
            paths = [image_paths[j] for j in indices]
            batches.append((indices, paths))
        
        # 并行预取和推理
        with ThreadPoolExecutor(max_workers=prefetch_factor) as executor:
            futures = [executor.submit(load_and_infer, batch) for batch in batches]
            
            for future in futures:
                indices, features = future.result()
                for idx, feature in zip(indices, features):
                    results[idx] = feature
        
        return results


def optimize_for_inference():
    """
    推理优化设置
    设置环境变量和优化选项
    """
    # 设置内存分配器
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # 禁用梯度计算
    try:
        import torch
        torch.set_grad_enabled(False)
        
        # 启用cudnn优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    except:
        pass
    
    # 设置垃圾回收阈值
    gc.set_threshold(700, 10, 5)
    
    logger.info("推理优化设置完成")


def cleanup_memory():
    """清理内存"""
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    logger.debug("内存已清理")


if __name__ == "__main__":
    # 测试
    monitor = MemoryMonitor()
    monitor.start()
    
    time.sleep(2)
    
    print(f"当前内存: {monitor.get_current_memory():.2f}MB")
    
    monitor.stop()
