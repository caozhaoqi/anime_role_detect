#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健康检查脚本 - 定期检查服务状态，发现异常自动重启
"""

import os
import sys
import time
import json
import psutil
import requests
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.logging.global_logger import get_logger

logger = get_logger("health_check")


class ServiceHealthChecker:
    """服务健康检查器"""

    def __init__(self):
        self.supervisor_config = project_root / "supervisord.conf"
        self.supervisorctl = project_root / ".venv" / "bin" / "supervisorctl"
        self.services = {
            "api-service": {"port": 8001, "endpoint": "/api/health", "timeout": 5},
            "model-service": {"port": 8000, "endpoint": "/api/health", "timeout": 10},
            "multimedia-service": {"port": 8002, "endpoint": "/api/health", "timeout": 5},
            "search-service": {"port": 8003, "endpoint": "/api/health", "timeout": 5},
            "api-gateway": {"port": 8080, "endpoint": "/api/health", "timeout": 5},
            "monitor-dashboard": {"port": 9000, "endpoint": "/", "timeout": 5},
        }
        self.alert_history = {}
        self.resource_thresholds = {
            "memory_percent": 80,  # 内存使用率阈值
            "cpu_percent": 90,     # CPU使用率阈值
            "response_time": 10,   # 响应时间阈值（秒）
        }
        self.is_kubernetes = self._detect_kubernetes()

    def _detect_kubernetes(self) -> bool:
        """检测是否在 Kubernetes 环境中运行"""
        return os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token")

    def _get_service_host(self, service_name: str) -> str:
        """获取服务主机名（K8s环境使用服务名，本地使用localhost）"""
        if self.is_kubernetes:
            return f"{service_name}.anime-role-detect.svc.cluster.local"
        return "localhost"

    def check_service_status(self, service_name: str) -> Dict[str, any]:
        """检查单个服务的状态"""
        result = {
            "service": service_name,
            "status": "unknown",
            "port": None,
            "response_time": None,
            "memory_percent": None,
            "memory_rss_mb": None,
            "cpu_percent": None,
            "error": None,
            "timestamp": datetime.now().isoformat()
        }

        try:
            if not self.is_kubernetes:
                if sys.platform != "win32":
                    supervisor_status = self._get_supervisor_status(service_name)
                    if supervisor_status != "RUNNING":
                        result["status"] = "stopped"
                        result["error"] = f"Supervisor status: {supervisor_status}"
                        return result

                port_info = self._check_port(service_name)
                if not port_info:
                    result["status"] = "port_not_listening"
                    result["error"] = f"Port {self.services[service_name]['port']} not listening"
                    return result

                result["port"] = self.services[service_name]["port"]

                process_info = self._get_process_info(service_name)
                if process_info:
                    result["memory_percent"] = process_info["memory_percent"]
                    result["memory_rss_mb"] = process_info.get("memory_rss_mb")
                    result["cpu_percent"] = process_info["cpu_percent"]

            health_check = self._check_http_endpoint(service_name)
            if not health_check["success"]:
                result["status"] = "health_check_failed"
                result["error"] = health_check["error"]
                return result

            result["status"] = "healthy"
            result["response_time"] = health_check["response_time"]
            if not self.is_kubernetes:
                result["port"] = self.services[service_name]["port"]

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"检查服务 {service_name} 时出错: {e}")

        return result

    def _get_supervisor_status(self, service_name: str) -> Optional[str]:
        """获取Supervisor中的服务状态"""
        # Windows平台不支持supervisorctl，直接返回None跳过此检查
        if sys.platform == "win32":
            return None

        try:
            cmd = [
                str(self.supervisorctl),
                "-c", str(self.supervisor_config),
                "-u", "CHANGE_ME_supervisor_admin",
                "-p", "CHANGE_ME_supervisor_pwd",
                "status", service_name
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # 输出格式: service_name  RUNNING   pid 12345, uptime 1:23:45
                parts = result.stdout.strip().split()
                if len(parts) >= 2:
                    return parts[1]

        except Exception as e:
            logger.error(f"获取Supervisor状态失败: {e}")

        return None

    def _get_supervisor_pid(self, service_name: str) -> Optional[int]:
        """通过 supervisor XML-RPC API 获取进程 PID

        macOS 上 psutil.net_connections() 需要 root 权限，
        改用 supervisorctl 获取 PID，再用 psutil.Process(pid) 读取资源。
        """
        if sys.platform == "win32":
            return None
        try:
            cmd = [
                str(self.supervisorctl),
                "-c", str(self.supervisor_config),
                "-u", "CHANGE_ME_supervisor_admin",
                "-p", "CHANGE_ME_supervisor_pwd",
                "pid", service_name
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                pid_str = result.stdout.strip()
                if pid_str.isdigit():
                    return int(pid_str)
        except Exception as e:
            logger.debug(f"获取 {service_name} PID 失败: {e}")
        return None

    def _check_port(self, service_name: str) -> bool:
        """检查端口是否在监听

        优先使用 socket 直连检测（不依赖 root 权限），
        不再使用 psutil.net_connections()。
        """
        import socket
        port = self.services[service_name]["port"]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex(("127.0.0.1", port))
            if result == 0:
                return True
        except Exception as e:
            logger.debug(f"检查端口 {port} 失败: {e}")
        finally:
            sock.close()
        return False

    def _get_process_info(self, service_name: str) -> Optional[Dict[str, float]]:
        """获取进程资源使用情况

        通过 supervisor API 获取 PID，再用 psutil.Process(pid) 读取 RSS/CPU。
        macOS 不需要 root 权限即可读取自己用户的进程信息。
        """
        pid = self._get_supervisor_pid(service_name)
        if pid is None:
            return None
        try:
            process = psutil.Process(pid)
            mem_info = process.memory_info()
            return {
                "memory_percent": process.memory_percent(),
                "memory_rss_mb": round(mem_info.rss / 1024 / 1024, 1),
                "cpu_percent": process.cpu_percent(interval=1)
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.debug(f"获取进程 {pid} 信息失败: {e}")
        return None

    def _check_http_endpoint(self, service_name: str) -> Dict[str, any]:
        """检查HTTP健康端点"""
        result = {"success": False, "response_time": None, "error": None}

        try:
            config = self.services[service_name]
            host = self._get_service_host(service_name)
            url = f"http://{host}:{config['port']}{config['endpoint']}"

            start_time = time.time()
            response = requests.get(url, timeout=config["timeout"])
            response_time = time.time() - start_time

            if response.status_code == 200:
                result["success"] = True
                result["response_time"] = response_time
            else:
                result["error"] = f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            result["error"] = "请求超时"
        except requests.exceptions.ConnectionError:
            result["error"] = "连接失败"
        except Exception as e:
            result["error"] = str(e)

        return result

    def check_all_services(self) -> List[Dict[str, any]]:
        """检查所有服务的状态"""
        results = []
        for service_name in self.services:
            result = self.check_service_status(service_name)
            results.append(result)
        return results

    def restart_service(self, service_name: str) -> bool:
        """重启服务"""
        if self.is_kubernetes:
            logger.info(f"Kubernetes环境下，服务 {service_name} 将由K8s自动重启，跳过手动重启")
            return False

        if sys.platform == "win32":
            logger.warning(f"Windows平台不支持通过supervisor重启服务: {service_name}")
            return False

        try:
            cmd = [
                str(self.supervisorctl),
                "-c", str(self.supervisor_config),
                "-u", "CHANGE_ME_supervisor_admin",
                "-p", "CHANGE_ME_supervisor_pwd",
                "restart", service_name
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=90
            )

            if result.returncode == 0:
                logger.info(f"服务 {service_name} 重启成功")
                return True
            else:
                logger.error(f"服务 {service_name} 重启失败: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"重启服务 {service_name} 时出错: {e}")
            return False

    def should_alert(self, service_name: str, issue_type: str) -> bool:
        """判断是否需要发送告警（避免重复告警）"""
        key = f"{service_name}:{issue_type}"
        now = datetime.now()

        if key in self.alert_history:
            last_alert = self.alert_history[key]
            # 同一问题10分钟内只告警一次
            if now - last_alert < timedelta(minutes=10):
                return False

        self.alert_history[key] = now
        return True

    def check_and_alert(self) -> List[Dict[str, any]]:
        """检查所有服务并发送告警"""
        results = self.check_all_services()
        alerts = []

        for result in results:
            service_name = result["service"]

            # 检查服务状态
            if result["status"] != "healthy":
                if self.should_alert(service_name, f"status_{result['status']}"):
                    alert = {
                        "service": service_name,
                        "type": "service_unhealthy",
                        "severity": "high",
                        "message": f"服务 {service_name} 状态异常: {result['status']}",
                        "details": result,
                        "timestamp": datetime.now().isoformat()
                    }
                    alerts.append(alert)
                    logger.error(alert["message"])

            # 检查资源使用情况
            if result["memory_percent"] and result["memory_percent"] > self.resource_thresholds["memory_percent"]:
                if self.should_alert(service_name, "memory_high"):
                    alert = {
                        "service": service_name,
                        "type": "resource_high",
                        "severity": "medium",
                        "message": f"服务 {service_name} 内存使用率过高: {result['memory_percent']:.1f}%",
                        "details": result,
                        "timestamp": datetime.now().isoformat()
                    }
                    alerts.append(alert)
                    logger.warning(alert["message"])

            if result["cpu_percent"] and result["cpu_percent"] > self.resource_thresholds["cpu_percent"]:
                if self.should_alert(service_name, "cpu_high"):
                    alert = {
                        "service": service_name,
                        "type": "resource_high",
                        "severity": "medium",
                        "message": f"服务 {service_name} CPU使用率过高: {result['cpu_percent']:.1f}%",
                        "details": result,
                        "timestamp": datetime.now().isoformat()
                    }
                    alerts.append(alert)
                    logger.warning(alert["message"])

            # 检查响应时间
            if result["response_time"] and result["response_time"] > self.resource_thresholds["response_time"]:
                if self.should_alert(service_name, "response_slow"):
                    alert = {
                        "service": service_name,
                        "type": "performance_slow",
                        "severity": "low",
                        "message": f"服务 {service_name} 响应时间过长: {result['response_time']:.2f}s",
                        "details": result,
                        "timestamp": datetime.now().isoformat()
                    }
                    alerts.append(alert)
                    logger.warning(alert["message"])

        return alerts

    def auto_restart_unhealthy_services(self) -> List[str]:
        """自动重启不健康的服务"""
        restarted_services = []
        results = self.check_all_services()

        for result in results:
            service_name = result["service"]

            # 只重启状态为 stopped 或 port_not_listening 的服务
            if result["status"] in ["stopped", "port_not_listening"]:
                logger.warning(f"服务 {service_name} 状态为 {result['status']}，尝试自动重启")

                if self.restart_service(service_name):
                    restarted_services.append(service_name)

                    # 等待服务启动
                    time.sleep(3)

                    # 再次检查状态
                    new_result = self.check_service_status(service_name)
                    if new_result["status"] == "healthy":
                        logger.info(f"服务 {service_name} 重启后恢复正常")
                    else:
                        logger.error(f"服务 {service_name} 重启后仍不健康: {new_result['status']}")

        return restarted_services

    def _get_system_memory(self) -> Dict[str, any]:
        """获取系统级内存信息（不依赖服务进程）"""
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                "total_gb": round(mem.total / 1024**3, 1),
                "available_gb": round(mem.available / 1024**3, 1),
                "used_gb": round(mem.used / 1024**3, 1),
                "percent": mem.percent,
                "swap_used_gb": round(swap.used / 1024**3, 1) if swap.total > 0 else 0,
                "swap_percent": swap.percent if swap.total > 0 else 0,
            }
        except Exception as e:
            logger.debug(f"获取系统内存失败: {e}")
            return {"error": str(e)}

    def save_health_report(self, results: List[Dict[str, any]], alerts: List[Dict[str, any]]):
        """保存健康检查报告"""
        try:
            report_dir = project_root / "logs" / "health_check"
            report_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"health_report_{timestamp}.json"

            report = {
                "timestamp": datetime.now().isoformat(),
                "system_memory": self._get_system_memory(),
                "services": results,
                "alerts": alerts,
                "summary": {
                    "total_services": len(results),
                    "healthy_services": sum(1 for r in results if r["status"] == "healthy"),
                    "unhealthy_services": sum(1 for r in results if r["status"] != "healthy"),
                    "total_alerts": len(alerts)
                }
            }

            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"健康检查报告已保存: {report_file}")

        except Exception as e:
            logger.error(f"保存健康检查报告失败: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="服务健康检查脚本")
    parser.add_argument("--check", action="store_true", help="检查所有服务状态")
    parser.add_argument("--restart", action="store_true", help="自动重启不健康的服务")
    parser.add_argument("--alert", action="store_true", help="检查并发送告警")
    parser.add_argument("--report", action="store_true", help="生成健康检查报告")
    parser.add_argument("--daemon", action="store_true", help="以守护进程模式运行，定期检查")
    parser.add_argument("--interval", type=int, default=60, help="检查间隔（秒），默认60秒")

    args = parser.parse_args()

    checker = ServiceHealthChecker()

    if args.check:
        results = checker.check_all_services()
        print(json.dumps(results, ensure_ascii=False, indent=2))

    elif args.restart:
        restarted = checker.auto_restart_unhealthy_services()
        print(f"已重启服务: {restarted}")

    elif args.alert:
        alerts = checker.check_and_alert()
        print(f"发现 {len(alerts)} 个告警")
        if alerts:
            print(json.dumps(alerts, ensure_ascii=False, indent=2))

    elif args.report:
        results = checker.check_all_services()
        alerts = checker.check_and_alert()
        checker.save_health_report(results, alerts)
        print("健康检查报告已生成")

    elif args.daemon:
        logger.info(f"健康检查守护进程启动，检查间隔: {args.interval}秒")

        # P2: 状态变化检测 + 自适应频率
        last_status_hash = None
        all_stopped_interval = 300  # 全部服务 stopped 时降频到 5 分钟
        normal_interval = args.interval

        while True:
            try:
                # 检查并告警
                alerts = checker.check_and_alert()

                # 自动重启不健康的服务
                if alerts:
                    restarted = checker.auto_restart_unhealthy_services()
                    if restarted:
                        logger.info(f"自动重启服务: {restarted}")

                # 生成报告（仅在状态变化时写入磁盘）
                results = checker.check_all_services()

                # P2: 计算状态指纹，状态无变化时跳过写盘
                current_status_hash = json.dumps([
                    {"service": r["service"], "status": r["status"]}
                    for r in results
                ], sort_keys=True)

                if current_status_hash != last_status_hash:
                    checker.save_health_report(results, alerts)
                    last_status_hash = current_status_hash
                else:
                    logger.debug("状态无变化，跳过报告写入")

                # P2: 全部服务 stopped 时降低检查频率
                all_stopped = all(r["status"] != "healthy" for r in results)
                sleep_interval = all_stopped_interval if all_stopped else normal_interval

                # 等待下一次检查
                time.sleep(sleep_interval)

            except KeyboardInterrupt:
                logger.info("健康检查守护进程已停止")
                break
            except Exception as e:
                logger.error(f"健康检查过程中出错: {e}")
                time.sleep(args.interval)

    else:
        # 默认行为：检查并显示结果
        results = checker.check_all_services()
        alerts = checker.check_and_alert()

        print("\n=== 服务健康检查报告 ===")
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总服务数: {len(results)}")
        print(f"健康服务: {sum(1 for r in results if r['status'] == 'healthy')}")
        print(f"异常服务: {sum(1 for r in results if r['status'] != 'healthy')}")
        print(f"告警数量: {len(alerts)}")

        # 显示系统内存信息
        sys_mem = checker._get_system_memory()
        if "error" not in sys_mem:
            print(f"\n系统内存: 总量={sys_mem['total_gb']}GB  已用={sys_mem['used_gb']}GB ({sys_mem['percent']}%)  可用={sys_mem['available_gb']}GB")
            if sys_mem.get("swap_used_gb", 0) > 0:
                print(f"交换分区: 已用={sys_mem['swap_used_gb']}GB ({sys_mem['swap_percent']}%)")

        print("\n=== 服务状态详情 ===")
        for result in results:
            status_icon = "✓" if result["status"] == "healthy" else "✗"
            print(f"{status_icon} {result['service']:20s} - {result['status']:20s}", end="")

            if result["memory_percent"]:
                print(f" 内存: {result['memory_percent']:5.1f}%", end="")
            if result["cpu_percent"]:
                print(f" CPU: {result['cpu_percent']:5.1f}%", end="")
            if result["response_time"]:
                print(f" 响应: {result['response_time']:.2f}s", end="")

            print()

        if alerts:
            print("\n=== 告警信息 ===")
            for alert in alerts:
                print(f"[{alert['severity'].upper()}] {alert['message']}")


if __name__ == "__main__":
    main()