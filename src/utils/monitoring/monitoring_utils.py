#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控工具

提供监控相关的辅助函数
"""

import threading
import time
from src.core.logging.global_logger import get_logger
from .system_monitor import SystemMonitor
from .performance_monitor import PerformanceMonitor
from .resource_monitor import ResourceMonitor

logger = get_logger("monitoring_system")

# 全局监控实例
_system_monitor = None
_performance_monitor = None
_resource_monitor = None
_monitoring_thread = None
_is_monitoring = False


def start_monitoring(interval=5):
    """
    开始监控

    Args:
        interval: 监控间隔（秒）
    """
    global _system_monitor, _performance_monitor, _resource_monitor, _monitoring_thread, _is_monitoring

    if _is_monitoring:
        logger.warning("监控已经在运行中")
        return

    # 初始化监控实例
    _system_monitor = SystemMonitor()
    _performance_monitor = PerformanceMonitor()
    _resource_monitor = ResourceMonitor()

    # 启动监控线程
    _is_monitoring = True
    _monitoring_thread = threading.Thread(target=_monitoring_loop, args=(interval,), daemon=True)
    _monitoring_thread.start()

    logger.info("监控已启动")


def stop_monitoring():
    """
    停止监控
    """
    global _is_monitoring, _monitoring_thread

    if not _is_monitoring:
        logger.warning("监控未运行")
        return

    _is_monitoring = False
    if _monitoring_thread:
        _monitoring_thread.join(timeout=5)

    logger.info("监控已停止")


def _monitoring_loop(interval):
    """
    监控循环

    Args:
        interval: 监控间隔（秒）
    """
    global _is_monitoring, _resource_monitor

    while _is_monitoring:
        if _resource_monitor:
            _resource_monitor.monitor_resources()
        time.sleep(interval)


def get_monitoring_data():
    """
    获取监控数据

    Returns:
        dict: 监控数据
    """
    global _system_monitor, _performance_monitor, _resource_monitor

    monitoring_data = {}

    if _system_monitor:
        monitoring_data["system"] = _system_monitor.get_usage_stats()

    if _performance_monitor:
        monitoring_data["performance"] = _performance_monitor.get_metrics()

    if _resource_monitor:
        monitoring_data["resource"] = _resource_monitor.get_resource_summary()

    return monitoring_data


def get_performance_monitor():
    """
    获取性能监控实例

    Returns:
        PerformanceMonitor: 性能监控实例
    """
    global _performance_monitor

    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()

    return _performance_monitor


def get_resource_monitor():
    """
    获取资源监控实例

    Returns:
        ResourceMonitor: 资源监控实例
    """
    global _resource_monitor

    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()

    return _resource_monitor


def get_system_monitor():
    """
    获取系统监控实例

    Returns:
        SystemMonitor: 系统监控实例
    """
    global _system_monitor

    if _system_monitor is None:
        _system_monitor = SystemMonitor()

    return _system_monitor
