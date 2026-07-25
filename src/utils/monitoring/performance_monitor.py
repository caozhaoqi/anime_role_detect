#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控

负责监控系统性能
"""

import time
from functools import wraps
from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("monitoring_system")


class PerformanceMonitor:
    """
    性能监控类
    """

    def __init__(self):
        """
        初始化性能监控
        """
        self.metrics = {}
        self.start_times = {}

    def measure_time(self, func):
        """
        测量函数执行时间的装饰器

        Args:
            func: 要测量的函数

        Returns:
            function: 装饰后的函数
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time

            function_name = func.__name__
            if function_name not in self.metrics:
                self.metrics[function_name] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_time": 0,
                    "max_time": 0,
                    "min_time": float("inf"),
                }

            # 更新指标
            self.metrics[function_name]["count"] += 1
            self.metrics[function_name]["total_time"] += execution_time
            self.metrics[function_name]["avg_time"] = (
                self.metrics[function_name]["total_time"] / self.metrics[function_name]["count"]
            )
            self.metrics[function_name]["max_time"] = max(
                self.metrics[function_name]["max_time"], execution_time
            )
            self.metrics[function_name]["min_time"] = min(
                self.metrics[function_name]["min_time"], execution_time
            )

            logger.debug(f"函数 {function_name} 执行时间: {execution_time:.4f} 秒")
            return result

        return wrapper

    def start_measurement(self, name):
        """
        开始测量

        Args:
            name: 测量名称
        """
        self.start_times[name] = time.time()

    def stop_measurement(self, name):
        """
        停止测量

        Args:
            name: 测量名称

        Returns:
            float: 测量时间
        """
        if name in self.start_times:
            end_time = time.time()
            execution_time = end_time - self.start_times[name]
            del self.start_times[name]

            if name not in self.metrics:
                self.metrics[name] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_time": 0,
                    "max_time": 0,
                    "min_time": float("inf"),
                }

            # 更新指标
            self.metrics[name]["count"] += 1
            self.metrics[name]["total_time"] += execution_time
            self.metrics[name]["avg_time"] = (
                self.metrics[name]["total_time"] / self.metrics[name]["count"]
            )
            self.metrics[name]["max_time"] = max(self.metrics[name]["max_time"], execution_time)
            self.metrics[name]["min_time"] = min(self.metrics[name]["min_time"], execution_time)

            return execution_time
        return 0

    def get_metrics(self):
        """
        获取性能指标

        Returns:
            dict: 性能指标
        """
        return self.metrics

    def reset_metrics(self):
        """
        重置性能指标
        """
        self.metrics = {}

    def log_metrics(self):
        """
        记录性能指标
        """
        for name, metrics in self.metrics.items():
            logger.info(f"性能指标 - {name}: ")
            logger.info(f"  调用次数: {metrics['count']}")
            logger.info(f"  总执行时间: {metrics['total_time']:.4f} 秒")
            logger.info(f"  平均执行时间: {metrics['avg_time']:.4f} 秒")
            logger.info(f"  最大执行时间: {metrics['max_time']:.4f} 秒")
            logger.info(f"  最小执行时间: {metrics['min_time']:.4f} 秒")
