#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源监控

负责监控系统资源使用情况
支持 K8s Pod/Node 感知（POD_NAME/NODE_NAME 环境变量）
支持容器内存 limit 比对（cgroup v1/v2），防 OOM（参考 K8s 文档 worker-2 162% 事故）
"""

import os
import time
from typing import Optional
import psutil
from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("monitoring_system")


def _read_cgroup_memory_limit() -> Optional[int]:
    """
    读取 cgroup 内存 limit（K8s 容器 limit）。
    cgroup v2: /sys/fs/cgroup/memory.max
    cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes
    返回字节数，None 表示未设限或读取失败。
    """
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                val = int(f.read().strip())
            # cgroup v2 中 "max" 会被 int() 拒绝，但上面 try 已覆盖
            if val > 0 and val < 1 << 62:  # 排除无限制的巨大值
                return val
        except (FileNotFoundError, ValueError, OSError):
            continue
    return None


def _read_cgroup_memory_usage() -> Optional[int]:
    """读取 cgroup 当前内存使用量（字节）。"""
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError, OSError):
            continue
    return None


class ResourceMonitor:
    """
    资源监控类
    支持 K8s Pod/Node 感知和容器内存 limit 比对告警
    """

    # 内存使用占 limit 的告警阈值（百分比）
    MEM_WARNING_THRESHOLD = 80.0
    MEM_CRITICAL_THRESHOLD = 90.0

    def __init__(self):
        """
        初始化资源监控
        """
        self.history = []
        self.max_history = 100
        # K8s 环境感知（非 K8s 时为 None）
        self.pod_name = os.environ.get("POD_NAME")
        self.node_name = os.environ.get("NODE_NAME")
        self.container_mem_limit = _read_cgroup_memory_limit()
        self._last_warn_time = 0.0

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
            disk = psutil.disk_usage("/")

            # 监控网络
            net_io = psutil.net_io_counters()

            # 容器内存使用 vs limit（K8s 下精确到 Pod 级别）
            container_mem_usage = _read_cgroup_memory_usage()
            container_mem_percent = None
            if self.container_mem_limit and container_mem_usage:
                container_mem_percent = round(
                    container_mem_usage / self.container_mem_limit * 100, 2
                )
                # 内存告警（参考 K8s 文档 worker-2 OOM 162% 事故，防止重蹈覆辙）
                now = time.time()
                if container_mem_percent >= self.MEM_CRITICAL_THRESHOLD:
                    if now - self._last_warn_time > 30:  # 30s 去重
                        logger.critical(
                            f"容器内存使用率 {container_mem_percent}% 超过临界阈值 "
                            f"{self.MEM_CRITICAL_THRESHOLD}%！"
                            f"(used={container_mem_usage // 1024 // 1024}MB, "
                            f"limit={self.container_mem_limit // 1024 // 1024}MB"
                            f"{f', pod={self.pod_name}' if self.pod_name else ''}"
                            f"{f', node={self.node_name}' if self.node_name else ''})"
                        )
                        self._last_warn_time = now
                elif container_mem_percent >= self.MEM_WARNING_THRESHOLD:
                    if now - self._last_warn_time > 60:  # 60s 去重
                        logger.warning(
                            f"容器内存使用率 {container_mem_percent}% 超过告警阈值 "
                            f"{self.MEM_WARNING_THRESHOLD}%"
                            f"{f', pod={self.pod_name}' if self.pod_name else ''}"
                        )
                        self._last_warn_time = now

            # 构建资源使用情况
            resource_usage = {
                "timestamp": time.time(),
                "cpu": {
                    "percent": cpu_percent,
                    "user": cpu_times.user,
                    "system": cpu_times.system,
                    "idle": cpu_times.idle,
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent,
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent,
                },
                "network": {
                    "sent": net_io.bytes_sent,
                    "received": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                },
            }

            # K8s Pod/Node 感知信息
            if self.pod_name:
                resource_usage["pod"] = self.pod_name
            if self.node_name:
                resource_usage["node"] = self.node_name

            # 容器级内存（K8s 下精确于系统级）
            if container_mem_usage is not None:
                resource_usage["container_memory"] = {
                    "used": container_mem_usage,
                    "limit": self.container_mem_limit,
                    "percent": container_mem_percent,
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
            recent_data = [data for data in self.history if data["timestamp"] >= cutoff_time]

            # 提取指定指标的数据
            trend_data = []
            for data in recent_data:
                if metric == "cpu":
                    value = data["cpu"]["percent"]
                elif metric == "memory":
                    value = data["memory"]["percent"]
                elif metric == "disk":
                    value = data["disk"]["percent"]
                elif metric == "network":
                    value = data["network"]["sent"] + data["network"]["received"]
                else:
                    continue

                trend_data.append({"timestamp": data["timestamp"], "value": value})

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
            avg_cpu = sum(data["cpu"]["percent"] for data in self.history) / len(self.history)
            avg_memory = sum(data["memory"]["percent"] for data in self.history) / len(self.history)
            avg_disk = sum(data["disk"]["percent"] for data in self.history) / len(self.history)

            summary = {
                "latest": latest,
                "average": {
                    "cpu_percent": avg_cpu,
                    "memory_percent": avg_memory,
                    "disk_percent": avg_disk,
                },
                "history_count": len(self.history),
            }

            # K8s 环境信息
            if self.pod_name:
                summary["pod"] = self.pod_name
            if self.node_name:
                summary["node"] = self.node_name

            # 容器内存 limit 摘要
            if self.container_mem_limit:
                summary["container_memory_limit_mb"] = self.container_mem_limit // 1024 // 1024
                cm = latest.get("container_memory")
                if cm:
                    summary["container_memory_percent"] = cm.get("percent")

            return summary
        except Exception as e:
            logger.error(f"获取资源使用摘要失败: {e}")
            return {}

    def clear_history(self):
        """
        清除历史记录
        """
        self.history = []
