#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源监控守护进程入口

使用 src/utils/monitoring/resource_monitor.py 中的 ResourceMonitor 类，
定期采集系统资源使用情况并写入日志。

Usage:
    python scripts/resource_monitor.py --daemon --interval 30
    python scripts/resource_monitor.py --once
"""

import os
import sys
import time
import signal
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.monitoring.resource_monitor import ResourceMonitor
from src.core.logging.global_logger import get_logger

logger = get_logger("resource_monitor")

_running = True


def _signal_handler(signum, frame):
    global _running
    _running = False
    logger.info("资源监控进程收到终止信号")


def log_resource_summary(data: dict):
    """将资源使用情况格式化为一行日志输出"""
    if not data:
        return

    cpu = data.get("cpu", {})
    memory = data.get("memory", {})
    disk = data.get("disk", {})

    cpu_pct = cpu.get("percent", 0)
    mem_total = memory.get("total", 0) / (1024 ** 3)
    mem_used = memory.get("used", 0) / (1024 ** 3)
    mem_pct = memory.get("percent", 0)
    disk_total = disk.get("total", 0) / (1024 ** 3)
    disk_used = disk.get("used", 0) / (1024 ** 3)
    disk_pct = disk.get("percent", 0)

    # 周期采样降级为 DEBUG：jsonl/控制台 sink 均为 INFO 级别，
    # 因此空闲时的资源采样不再刷屏；仅超阈值告警保留 WARNING（见日志噪音诊断）。
    logger.debug(
        f"资源使用 | CPU: {cpu_pct:.1f}% | "
        f"内存: {mem_used:.1f}/{mem_total:.1f}GB ({mem_pct:.1f}%) | "
        f"磁盘: {disk_used:.1f}/{disk_total:.1f}GB ({disk_pct:.1f}%)"
    )

    # 阈值告警
    warnings = []
    if mem_pct > 80:
        warnings.append(f"内存使用率过高: {mem_pct:.1f}%")
    if cpu_pct > 90:
        warnings.append(f"CPU使用率过高: {cpu_pct:.1f}%")
    if disk_pct > 85:
        warnings.append(f"磁盘使用率过高: {disk_pct:.1f}%")
    for w in warnings:
        logger.warning(f"资源告警: {w}")


def run_once():
    """运行一次资源采集"""
    monitor = ResourceMonitor()
    data = monitor.monitor_resources()
    log_resource_summary(data)
    return data


def run_daemon(interval: int = 30):
    """以守护模式运行，定期采集资源"""
    global _running

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    monitor = ResourceMonitor()
    logger.info(f"资源监控守护进程启动，采集间隔: {interval}s")

    while _running:
        try:
            data = monitor.monitor_resources()
            log_resource_summary(data)
        except Exception as e:
            logger.error(f"资源监控采集异常: {e}")

        # 分段等待，便于快速响应终止信号
        for _ in range(interval):
            if not _running:
                break
            time.sleep(1)

    logger.info("资源监控守护进程已停止")


def main():
    parser = argparse.ArgumentParser(description="资源监控守护进程")
    parser.add_argument("--daemon", action="store_true", help="以守护模式运行")
    parser.add_argument("--interval", type=int, default=30, help="采集间隔（秒）")
    parser.add_argument("--once", action="store_true", help="只运行一次")
    args = parser.parse_args()

    if args.daemon:
        run_daemon(interval=args.interval)
    elif args.once:
        run_once()
    else:
        run_once()


if __name__ == "__main__":
    main()