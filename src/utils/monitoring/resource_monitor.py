#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源监控

负责监控系统资源使用情况
"""

import time
import psutil
from src.core.logging.global_logger import get_logger

logger = get_logger("monitoring_system")


class ResourceMonitor:
    """
    资源监控类
    """
    
    def __init__(self):
        """
        初始化资源监控
        """
        self.history = []
        self.max_history = 100
    
    def monitor_resources(self):
        """
        监控资源使用情况
        
        Returns:
            dict: 资源使用情况
        """
        try:
            # 监控CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_times = psutil.cpu_times()
            
            # 监控内存
            memory = psutil.virtual_memory()
            
            # 监控磁盘
            disk = psutil.disk_usage('/')
            
            # 监控网络
            net_io = psutil.net_io_counters()
            
            # 构建资源使用情况
            resource_usage = {
                "timestamp": time.time(),
                "cpu": {
                    "percent": cpu_percent,
                    "user": cpu_times.user,
                    "system": cpu_times.system,
                    "idle": cpu_times.idle
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                },
                "network": {
                    "sent": net_io.bytes_sent,
                    "received": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            }
            
            # 添加到历史记录
            self.history.append(resource_usage)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            return resource_usage
        except Exception as e:
            logger.error(f"监控资源使用情况失败: {e}")
            return {}
    
    def get_resource_trend(self, metric, duration=60):
        """
        获取资源使用趋势
        
        Args:
            metric: 指标名称
            duration: 持续时间（秒）
        
        Returns:
            list: 趋势数据
        """
        try:
            # 过滤指定时间范围内的数据
            cutoff_time = time.time() - duration
            recent_data = [data for data in self.history if data['timestamp'] >= cutoff_time]
            
            # 提取指定指标的数据
            trend_data = []
            for data in recent_data:
                if metric == 'cpu':
                    value = data['cpu']['percent']
                elif metric == 'memory':
                    value = data['memory']['percent']
                elif metric == 'disk':
                    value = data['disk']['percent']
                elif metric == 'network':
                    value = data['network']['sent'] + data['network']['received']
                else:
                    continue
                
                trend_data.append({
                    "timestamp": data['timestamp'],
                    "value": value
                })
            
            return trend_data
        except Exception as e:
            logger.error(f"获取资源使用趋势失败: {e}")
            return []
    
    def get_resource_summary(self):
        """
        获取资源使用摘要
        
        Returns:
            dict: 资源使用摘要
        """
        try:
            if not self.history:
                return {}
            
            # 获取最新的资源使用情况
            latest = self.history[-1]
            
            # 计算平均值
            avg_cpu = sum(data['cpu']['percent'] for data in self.history) / len(self.history)
            avg_memory = sum(data['memory']['percent'] for data in self.history) / len(self.history)
            avg_disk = sum(data['disk']['percent'] for data in self.history) / len(self.history)
            
            summary = {
                "latest": latest,
                "average": {
                    "cpu_percent": avg_cpu,
                    "memory_percent": avg_memory,
                    "disk_percent": avg_disk
                },
                "history_count": len(self.history)
            }
            
            return summary
        except Exception as e:
            logger.error(f"获取资源使用摘要失败: {e}")
            return {}
    
    def clear_history(self):
        """
        清除历史记录
        """
        self.history = []
