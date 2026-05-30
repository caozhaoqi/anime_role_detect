#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统监控

负责监控系统状态
"""

import os
import time
import psutil
from src.core.logging.global_logger import get_logger

logger = get_logger("monitoring_system")


class SystemMonitor:
    """
    系统监控类
    """

    def __init__(self):
        """
        初始化系统监控
        """
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())

    def get_system_info(self):
        """
        获取系统信息

        Returns:
            dict: 系统信息
        """
        try:
            # 系统信息
            system_info = {
                "uptime": time.time() - self.start_time,
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "used": psutil.virtual_memory().used,
                    "percent": psutil.virtual_memory().percent,
                },
                "disk": {
                    "total": psutil.disk_usage("/").total,
                    "used": psutil.disk_usage("/").used,
                    "free": psutil.disk_usage("/").free,
                    "percent": psutil.disk_usage("/").percent,
                },
                "network": {
                    "sent": psutil.net_io_counters().bytes_sent,
                    "received": psutil.net_io_counters().bytes_recv,
                },
            }
            return system_info
        except Exception as e:
            logger.error(f"获取系统信息失败: {e}")
            return {}

    def get_process_info(self):
        """
        获取进程信息

        Returns:
            dict: 进程信息
        """
        try:
            # 进程信息
            process_info = {
                "pid": self.process.pid,
                "name": self.process.name(),
                "cpu_percent": self.process.cpu_percent(interval=0.1),
                "memory_info": {
                    "rss": self.process.memory_info().rss,
                    "vms": self.process.memory_info().vms,
                    "percent": self.process.memory_percent(),
                },
                "threads": self.process.num_threads(),
                "open_files": len(self.process.open_files()),
            }
            return process_info
        except Exception as e:
            logger.error(f"获取进程信息失败: {e}")
            return {}

    def get_usage_stats(self):
        """
        获取使用统计

        Returns:
            dict: 使用统计
        """
        try:
            # 组合系统和进程信息
            stats = {
                "system": self.get_system_info(),
                "process": self.get_process_info(),
                "timestamp": time.time(),
            }
            return stats
        except Exception as e:
            logger.error(f"获取使用统计失败: {e}")
            return {}

    def monitor(self, interval=5):
        """
        监控系统状态

        Args:
            interval: 监控间隔（秒）
        """
        while True:
            stats = self.get_usage_stats()
            logger.info(f"系统状态: {stats}")
            time.sleep(interval)
