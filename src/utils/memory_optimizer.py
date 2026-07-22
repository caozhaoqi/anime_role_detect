#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一资源监控与内存优化工具 (P2: 合并自 memory_optimizer.py + memory_monitor.py)

提供：
- 进程级 RSS 监控 + GC 触发
- 系统级内存监控 + 泄漏检测
- 批量推理内存优化
- 推理后内存清理
- health check 数据接口
"""

import os
import gc
import sys
import time
import json
import psutil
import threading
from typing import List, Optional, Union, Iterator, Callable, Dict
from collections import deque
from datetime import datetime
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger("resource_monitor")


class ResourceMonitor:
    """
    统一资源监控器 (P2: 合并 MemoryMonitor #1 + MemoryMonitor #2)

    同时监控系统级内存和进程级 RSS，支持泄漏检测和自动 GC。
    提供 get_snapshot() 供 health check 调用。
    """

    def __init__(
        self,
        interval: float = 10.0,
        memory_threshold: float = 80,
        critical_threshold: float = 90,
        max_history: int = 100,
    ):
        self.interval = interval
        self.memory_threshold = memory_threshold
        self.critical_threshold = critical_threshold
        self.max_history = max_history
        self.monitoring = False
        self.thread = None
        self._lock = threading.Lock()

        # 进程级数据
        self._process = psutil.Process(os.getpid())
        self._peak_rss_mb = 0.0

        # 系统级数据（历史记录）
        self._system_history: deque = deque(maxlen=max_history)
        self._memory_trend: deque = deque(maxlen=10)
        self._last_alert_time = 0.0
        self._alert_cooldown = 300  # 5 分钟

    def start(self):
        """启动后台监控线程"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("ResourceMonitor 已启动")

    def stop(self):
        """停止监控"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info(f"ResourceMonitor 已停止，峰值 RSS: {self._peak_rss_mb:.1f}MB")

    def _monitor_loop(self):
        """监控主循环"""
        while self.monitoring:
            try:
                # 进程级 RSS
                rss_mb = self._process.memory_info().rss / 1024 / 1024
                if rss_mb > self._peak_rss_mb:
                    self._peak_rss_mb = rss_mb

                # 系统级内存
                vm = psutil.virtual_memory()
                sys_percent = vm.percent
                sys_used_gb = vm.used / 1024 / 1024 / 1024
                sys_total_gb = vm.total / 1024 / 1024 / 1024

                now = datetime.now()
                with self._lock:
                    self._system_history.append({
                        "timestamp": now.isoformat(),
                        "sys_memory_percent": sys_percent,
                        "sys_memory_used_gb": round(sys_used_gb, 2),
                        "sys_memory_total_gb": round(sys_total_gb, 2),
                        "proc_rss_mb": round(rss_mb, 1),
                    })
                    self._memory_trend.append(sys_percent)

                # 阈值告警
                self._check_thresholds(sys_percent, rss_mb)

                # 泄漏检测
                self._check_leak()

                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"ResourceMonitor 监控出错: {e}")
                time.sleep(self.interval)

    def _check_thresholds(self, sys_percent: float, rss_mb: float):
        """阈值检查 + 自动 GC"""
        now_ts = time.time()
        if sys_percent > self.critical_threshold:
            logger.critical(
                f"临界内存: 系统 {sys_percent:.1f}%, 进程 RSS {rss_mb:.1f}MB, 触发 GC"
            )
            gc.collect()
        elif sys_percent > self.memory_threshold:
            if now_ts - self._last_alert_time > self._alert_cooldown:
                logger.warning(
                    f"内存预警: 系统 {sys_percent:.1f}%, 进程 RSS {rss_mb:.1f}MB"
                )
                self._last_alert_time = now_ts

    def _check_leak(self):
        """检测内存泄漏趋势"""
        if len(self._memory_trend) < 10:
            return
        recent = list(self._memory_trend)[-5:]
        older = list(self._memory_trend)[:-5]
        if older and recent:
            avg_recent = sum(recent) / len(recent)
            avg_older = sum(older) / len(older)
            if avg_recent > avg_older + 3:
                logger.warning(
                    f"检测到内存泄漏迹象: {avg_older:.1f}% → {avg_recent:.1f}%"
                )
                gc.collect()

    def get_snapshot(self) -> Dict:
        """获取当前资源快照（供 health check 调用）

        Returns:
            dict: 包含进程级和系统级内存数据
        """
        try:
            rss = self._process.memory_info().rss
            vm = psutil.virtual_memory()
            return {
                "proc_rss_mb": round(rss / 1024 / 1024, 1),
                "proc_memory_percent": round(self._process.memory_percent(), 2),
                "proc_cpu_percent": self._process.cpu_percent(interval=0.1),
                "sys_memory_percent": vm.percent,
                "sys_memory_used_gb": round(vm.used / 1024 / 1024 / 1024, 2),
                "sys_memory_total_gb": round(vm.total / 1024 / 1024 / 1024, 2),
                "peak_rss_mb": round(self._peak_rss_mb, 1),
            }
        except Exception as e:
            logger.error(f"获取资源快照失败: {e}")
            return {}

    def get_current_memory(self) -> float:
        """获取当前进程 RSS（MB）"""
        return self._process.memory_info().rss / 1024 / 1024

    def get_peak_memory(self) -> float:
        """获取峰值 RSS（MB）"""
        return self._peak_rss_mb


# 兼容别名（旧代码可能引用 MemoryMonitor 名称）
MemoryMonitor = ResourceMonitor


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
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def process_image_stream(
        self,
        image_paths: Iterator[str],
        process_func: Callable[[List[str]], List],
    ) -> Iterator:
        """流式处理图片"""
        batch = []
        for path in image_paths:
            batch.append(path)
            if len(batch) >= self.batch_size:
                results = process_func(batch)
                for result in results:
                    yield result
                batch = []
                if len(batch) % (self.batch_size * 10) == 0:
                    gc.collect()
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
        """在内存限制下处理"""
        results = []
        batch = []
        monitor = ResourceMonitor()
        for item in items:
            batch.append(item)
            if len(batch) >= self.batch_size:
                current_memory = monitor.get_current_memory()
                if current_memory > memory_limit_mb:
                    logger.warning(f"内存接近限制: {current_memory:.0f}MB, 减小批次")
                    results.extend(process_func(batch))
                    batch = []
                    gc.collect()
                else:
                    results.extend(process_func(batch))
                    batch = []
        if batch:
            results.extend(process_func(batch))
        return results


class BatchInferenceOptimizer:
    """
    批量推理优化器
    """

    def __init__(
        self,
        embedder,
        optimal_batch_size: int = 16,
        max_batch_size: int = 64,
    ):
        self.embedder = embedder
        self.optimal_batch_size = optimal_batch_size
        self.max_batch_size = max_batch_size

    def infer_with_auto_batch(
        self,
        image_paths: List[str],
        target_memory_mb: float = 2048,
    ) -> List[Optional[np.ndarray]]:
        """自动调整批大小进行推理"""
        monitor = ResourceMonitor()
        results = []
        batch_size = self.optimal_batch_size
        i = 0
        while i < len(image_paths):
            batch = image_paths[i:i+batch_size]
            try:
                batch_results = self.embedder.embed_images(batch, batch_size=len(batch))
                results.extend(batch_results)
                current_memory = monitor.get_current_memory()
                if current_memory > target_memory_mb and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    logger.info(f"内存优化: 批大小调整为 {batch_size}")
                elif current_memory < target_memory_mb * 0.5 and batch_size < self.max_batch_size:
                    batch_size = min(self.max_batch_size, batch_size * 2)
                i += len(batch)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM错误，减小批大小")
                    batch_size = max(1, batch_size // 2)
                    gc.collect()
                    if batch_size == 1:
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
        """预取推理"""
        from concurrent.futures import ThreadPoolExecutor
        results = [None] * len(image_paths)

        def load_and_infer(idx_batch):
            indices, paths = idx_batch
            features = self.embedder.embed_images(paths, batch_size=len(paths))
            return indices, features

        batches = []
        for i in range(0, len(image_paths), self.optimal_batch_size):
            indices = list(range(i, min(i + self.optimal_batch_size, len(image_paths))))
            paths = [image_paths[j] for j in indices]
            batches.append((indices, paths))

        with ThreadPoolExecutor(max_workers=prefetch_factor) as executor:
            futures = [executor.submit(load_and_infer, batch) for batch in batches]
            for future in futures:
                indices, features = future.result()
                for idx, feature in zip(indices, features):
                    results[idx] = feature
        return results


def optimize_for_inference():
    """推理优化设置"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    try:
        import torch
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    except Exception:
        pass
    gc.set_threshold(700, 10, 5)
    logger.info("推理优化设置完成")


def cleanup_memory():
    """清理内存（推理后调用）"""
    gc.collect()
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.debug("内存已清理")


# 全局 ResourceMonitor 实例
_resource_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """获取全局 ResourceMonitor 实例"""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor


def init_resource_monitoring():
    """初始化资源监控（供应用启动时调用）"""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
        _resource_monitor.start()
        logger.info("资源监控系统初始化完成")


def shutdown_resource_monitoring():
    """关闭资源监控"""
    global _resource_monitor
    if _resource_monitor:
        _resource_monitor.stop()
        _resource_monitor = None
        logger.info("资源监控系统已关闭")


if __name__ == "__main__":
    init_resource_monitoring()
    try:
        print("ResourceMonitor 已启动，按 Ctrl+C 退出...")
        while True:
            time.sleep(5)
            snap = get_resource_monitor().get_snapshot()
            print(f"RSS: {snap.get('proc_rss_mb', 0):.1f}MB, "
                  f"系统: {snap.get('sys_memory_percent', 0):.1f}%")
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_resource_monitoring()
