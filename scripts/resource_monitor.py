#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源监控脚本 - 监控内存、CPU使用情况，防止资源耗尽
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from collections import deque
import psutil

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.logging.global_logger import get_logger

logger = get_logger("resource_monitor")


class ResourceMonitor:
    """资源监控器"""

    def __init__(
        self,
        check_interval: int = 30,
        history_size: int = 100,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        初始化资源监控器

        Args:
            check_interval: 检查间隔（秒）
            history_size: 历史记录大小
            alert_thresholds: 告警阈值
        """
        self.check_interval = check_interval
        self.history_size = history_size

        # 默认告警阈值
        self.alert_thresholds = alert_thresholds or {
            "memory_percent": 80,      # 内存使用率阈值
            "cpu_percent": 90,         # CPU使用率阈值
            "disk_percent": 85,        # 磁盘使用率阈值
            "memory_mb": 4000,         # 内存使用量阈值（MB）
            "process_memory_mb": 2000, # 单进程内存阈值（MB）
            "process_cpu_percent": 80, # 单进程CPU阈值
        }

        # 历史记录
        self.history = deque(maxlen=history_size)

        # 告警历史
        self.alert_history = {}

        # 服务端口映射
        self.service_ports = {
            "api-service": 8001,
            "model-service": 8000,
            "multimedia-service": 8002,
            "search-service": 8003,
            "api-gateway": 8080,
            "monitor-dashboard": 9000,
        }

        logger.info(f"资源监控器初始化完成，检查间隔: {check_interval}秒")

    def get_system_resources(self) -> Dict[str, any]:
        """
        获取系统资源使用情况

        Returns:
            系统资源信息
        """
        try:
            # CPU信息
            cpu_info = {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            }

            # 内存信息
            memory = psutil.virtual_memory()
            memory_info = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "free": memory.free,
                "percent": memory.percent,
                "total_mb": memory.total / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
            }

            # 磁盘信息
            disk_info = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": usage.percent,
                        "total_gb": usage.total / (1024 ** 3),
                        "used_gb": usage.used / (1024 ** 3),
                        "free_gb": usage.free / (1024 ** 3),
                    })
                except Exception as e:
                    logger.warning(f"获取磁盘 {partition.mountpoint} 信息失败: {e}")

            # 网络信息
            network_info = {
                "io_counters": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else None,
                "connections": len(psutil.net_connections()),
            }

            return {
                "timestamp": datetime.now().isoformat(),
                "cpu": cpu_info,
                "memory": memory_info,
                "disk": disk_info,
                "network": network_info,
            }

        except Exception as e:
            logger.error(f"获取系统资源信息失败: {e}")
            return {}

    def get_process_resources(self, service_name: str) -> Optional[Dict[str, any]]:
        """
        获取指定服务的进程资源使用情况

        Args:
            service_name: 服务名称

        Returns:
            进程资源信息
        """
        try:
            port = self.service_ports.get(service_name)
            if not port:
                return None

            # 通过端口查找进程
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == "LISTEN":
                    try:
                        process = psutil.Process(conn.pid)

                        # CPU信息
                        cpu_percent = process.cpu_percent(interval=1)

                        # 内存信息
                        memory_info = process.memory_info()
                        memory_percent = process.memory_percent()

                        return {
                            "service": service_name,
                            "pid": conn.pid,
                            "name": process.name(),
                            "status": process.status(),
                            "cpu": {
                                "percent": cpu_percent,
                                "num_threads": process.num_threads(),
                            },
                            "memory": {
                                "rss": memory_info.rss,
                                "vms": memory_info.vms,
                                "percent": memory_percent,
                                "rss_mb": memory_info.rss / (1024 * 1024),
                                "vms_mb": memory_info.vms / (1024 * 1024),
                            },
                            "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
                            "timestamp": datetime.now().isoformat(),
                        }

                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        logger.warning(f"获取进程 {conn.pid} 信息失败: {e}")
                        return None

        except Exception as e:
            logger.error(f"获取服务 {service_name} 进程资源失败: {e}")

        return None

    def get_all_services_resources(self) -> List[Dict[str, any]]:
        """
        获取所有服务的进程资源使用情况

        Returns:
            所有服务的资源信息列表
        """
        resources = []

        for service_name in self.service_ports:
            resource = self.get_process_resources(service_name)
            if resource:
                resources.append(resource)

        return resources

    def check_thresholds(self, system_resources: Dict[str, any], service_resources: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        检查资源使用是否超过阈值

        Args:
            system_resources: 系统资源信息
            service_resources: 服务资源信息列表

        Returns:
            告警列表
        """
        alerts = []

        # 检查系统内存
        if system_resources.get("memory", {}).get("percent", 0) > self.alert_thresholds["memory_percent"]:
            alerts.append({
                "type": "system",
                "severity": "high",
                "resource": "memory",
                "message": f"系统内存使用率过高: {system_resources['memory']['percent']:.1f}%",
                "current_value": system_resources['memory']['percent'],
                "threshold": self.alert_thresholds["memory_percent"],
                "timestamp": datetime.now().isoformat(),
            })

        # 检查系统CPU
        if system_resources.get("cpu", {}).get("percent", 0) > self.alert_thresholds["cpu_percent"]:
            alerts.append({
                "type": "system",
                "severity": "high",
                "resource": "cpu",
                "message": f"系统CPU使用率过高: {system_resources['cpu']['percent']:.1f}%",
                "current_value": system_resources['cpu']['percent'],
                "threshold": self.alert_thresholds["cpu_percent"],
                "timestamp": datetime.now().isoformat(),
            })

        # 检查磁盘使用
        for disk in system_resources.get("disk", []):
            if disk.get("percent", 0) > self.alert_thresholds["disk_percent"]:
                alerts.append({
                    "type": "system",
                    "severity": "medium",
                    "resource": "disk",
                    "message": f"磁盘 {disk['mountpoint']} 使用率过高: {disk['percent']:.1f}%",
                    "current_value": disk['percent'],
                    "threshold": self.alert_thresholds["disk_percent"],
                    "mountpoint": disk['mountpoint'],
                    "timestamp": datetime.now().isoformat(),
                })

        # 检查服务进程资源
        for service in service_resources:
            # 检查进程内存
            memory_mb = service.get("memory", {}).get("rss_mb", 0)
            if memory_mb > self.alert_thresholds["process_memory_mb"]:
                alerts.append({
                    "type": "service",
                    "severity": "medium",
                    "resource": "process_memory",
                    "service": service["service"],
                    "message": f"服务 {service['service']} 内存使用过高: {memory_mb:.1f}MB",
                    "current_value": memory_mb,
                    "threshold": self.alert_thresholds["process_memory_mb"],
                    "timestamp": datetime.now().isoformat(),
                })

            # 检查进程CPU
            cpu_percent = service.get("cpu", {}).get("percent", 0)
            if cpu_percent > self.alert_thresholds["process_cpu_percent"]:
                alerts.append({
                    "type": "service",
                    "severity": "medium",
                    "resource": "process_cpu",
                    "service": service["service"],
                    "message": f"服务 {service['service']} CPU使用过高: {cpu_percent:.1f}%",
                    "current_value": cpu_percent,
                    "threshold": self.alert_thresholds["process_cpu_percent"],
                    "timestamp": datetime.now().isoformat(),
                })

        return alerts

    def should_alert(self, alert_key: str, cooldown_minutes: int = 10) -> bool:
        """
        判断是否需要发送告警（避免重复告警）

        Args:
            alert_key: 告警键
            cooldown_minutes: 冷却时间（分钟）

        Returns:
            是否可以发送告警
        """
        now = datetime.now()

        if alert_key in self.alert_history:
            last_alert = self.alert_history[alert_key]
            if now - last_alert < timedelta(minutes=cooldown_minutes):
                return False

        self.alert_history[alert_key] = now
        return True

    def monitor_once(self) -> Dict[str, any]:
        """
        执行一次资源监控

        Returns:
            监控结果
        """
        # 获取资源信息
        system_resources = self.get_system_resources()
        service_resources = self.get_all_services_resources()

        # 检查阈值
        all_alerts = self.check_thresholds(system_resources, service_resources)

        # 过滤告警（避免重复）
        filtered_alerts = []
        for alert in all_alerts:
            alert_key = f"{alert['type']}:{alert['resource']}:{alert.get('service', 'system')}"
            if self.should_alert(alert_key):
                filtered_alerts.append(alert)

                # 记录告警
                if alert["severity"] == "high":
                    logger.error(alert["message"])
                elif alert["severity"] == "medium":
                    logger.warning(alert["message"])
                else:
                    logger.info(alert["message"])

        # 构建结果
        result = {
            "timestamp": datetime.now().isoformat(),
            "system": system_resources,
            "services": service_resources,
            "alerts": filtered_alerts,
            "summary": {
                "total_services": len(service_resources),
                "total_alerts": len(filtered_alerts),
                "high_severity_alerts": sum(1 for a in filtered_alerts if a["severity"] == "high"),
                "medium_severity_alerts": sum(1 for a in filtered_alerts if a["severity"] == "medium"),
            }
        }

        # 添加到历史记录
        self.history.append(result)

        return result

    def monitor_continuously(self):
        """持续监控资源"""
        logger.info(f"开始持续监控资源，检查间隔: {self.check_interval}秒")

        try:
            while True:
                try:
                    result = self.monitor_once()

                    if result["summary"]["total_alerts"] > 0:
                        logger.info(
                            f"资源监控周期完成: 服务={result['summary']['total_services']}, "
                            f"告警={result['summary']['total_alerts']}"
                        )

                    time.sleep(self.check_interval)

                except KeyboardInterrupt:
                    logger.info("资源监控已停止")
                    break
                except Exception as e:
                    logger.error(f"监控过程中出错: {e}")
                    time.sleep(self.check_interval)

        except Exception as e:
            logger.error(f"持续监控失败: {e}")

    def get_statistics(self) -> Dict[str, any]:
        """
        获取统计信息

        Returns:
            统计信息
        """
        if not self.history:
            return {"message": "暂无历史数据"}

        # 计算平均值
        avg_memory = sum(r["system"].get("memory", {}).get("percent", 0) for r in self.history) / len(self.history)
        avg_cpu = sum(r["system"].get("cpu", {}).get("percent", 0) for r in self.history) / len(self.history)

        # 统计告警
        total_alerts = sum(len(r["alerts"]) for r in self.history)

        return {
            "timestamp": datetime.now().isoformat(),
            "history_size": len(self.history),
            "averages": {
                "memory_percent": avg_memory,
                "cpu_percent": avg_cpu,
            },
            "total_alerts": total_alerts,
            "alert_history": {
                key: value.isoformat()
                for key, value in self.alert_history.items()
            }
        }

    def save_report(self, output_file: Optional[Path] = None):
        """
        保存监控报告

        Args:
            output_file: 输出文件路径
        """
        if output_file is None:
            output_dir = project_root / "logs" / "resource_monitor"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"resource_report_{timestamp}.json"

        try:
            result = self.monitor_once()

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(f"资源监控报告已保存: {output_file}")

        except Exception as e:
            logger.error(f"保存监控报告失败: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="资源监控脚本")
    parser.add_argument("--once", action="store_true", help="执行一次监控检查")
    parser.add_argument("--daemon", action="store_true", help="以守护进程模式运行")
    parser.add_argument("--interval", type=int, default=30, help="检查间隔（秒），默认30秒")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--report", action="store_true", help="生成监控报告")
    parser.add_argument("--memory-threshold", type=float, default=80, help="内存使用率阈值（%）")
    parser.add_argument("--cpu-threshold", type=float, default=90, help="CPU使用率阈值（%）")

    args = parser.parse_args()

    # 自定义阈值
    custom_thresholds = {
        "memory_percent": args.memory_threshold,
        "cpu_percent": args.cpu_threshold,
    }

    monitor = ResourceMonitor(
        check_interval=args.interval,
        alert_thresholds=custom_thresholds
    )

    if args.once:
        result = monitor.monitor_once()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.stats:
        stats = monitor.get_statistics()
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    elif args.report:
        monitor.save_report()
        print("资源监控报告已生成")

    elif args.daemon:
        monitor.monitor_continuously()

    else:
        # 默认行为：执行一次监控检查
        result = monitor.monitor_once()

        print("\n=== 资源监控报告 ===")
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 系统资源
        system = result.get("system", {})
        if system:
            print(f"\n系统资源:")
            print(f"  CPU使用率: {system.get('cpu', {}).get('percent', 0):.1f}%")
            print(f"  内存使用率: {system.get('memory', {}).get('percent', 0):.1f}%")
            print(f"  内存使用量: {system.get('memory', {}).get('used_mb', 0):.1f}MB / {system.get('memory', {}).get('total_mb', 0):.1f}MB")

            disk = system.get('disk', [])
            if disk:
                print(f"  磁盘使用率:")
                for d in disk:
                    print(f"    {d['mountpoint']}: {d['percent']:.1f}% ({d['used_gb']:.1f}GB / {d['total_gb']:.1f}GB)")

        # 服务资源
        services = result.get("services", [])
        if services:
            print(f"\n服务资源:")
            for service in services:
                print(f"  {service['service']:20s} - "
                      f"CPU: {service.get('cpu', {}).get('percent', 0):5.1f}%, "
                      f"内存: {service.get('memory', {}).get('rss_mb', 0):6.1f}MB, "
                      f"PID: {service['pid']}")

        # 告警信息
        alerts = result.get("alerts", [])
        if alerts:
            print(f"\n告警信息:")
            for alert in alerts:
                print(f"  [{alert['severity'].upper()}] {alert['message']}")
        else:
            print(f"\n✓ 所有资源使用正常")


if __name__ == "__main__":
    main()