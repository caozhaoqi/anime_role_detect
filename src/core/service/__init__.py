#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务注册模块
"""

import os
import subprocess
import time
import socket
import sys
from typing import Dict, List, Optional, Callable

from src.core.config import get_config
from src.core.logging import get_logger

logger = get_logger("service_registry")


class ServiceRegistry:
    """服务注册管理器"""

    def __init__(self):
        self._services: Dict[str, Dict] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._config = get_config()

    def register_service(self, name: str, config: Dict):
        """
        注册服务

        Args:
            name: 服务名称
            config: 服务配置
        """
        self._services[name] = config
        logger.info(f"服务已注册: {name}")

    def register_services(self, services: Dict[str, Dict]):
        """批量注册服务"""
        for name, config in services.items():
            self.register_service(name, config)

    def start_service(self, name: str) -> bool:
        """
        启动单个服务

        Args:
            name: 服务名称

        Returns:
            是否启动成功
        """
        if name not in self._services:
            logger.error(f"服务未注册: {name}")
            return False

        service = self._services[name]

        if not service.get("enabled", True):
            logger.info(f"服务已禁用: {name}")
            return False

        port = service.get("port")
        script = service.get("script")
        description = service.get("description", "")
        args = service.get("args", [])
        env_vars = service.get("env", {})
        directory = service.get("directory", "")

        # 检查端口是否被占用
        if port and self._is_port_in_use(port):
            logger.warning(f"端口 {port} 已被占用: {name}")
            return False

        logger.info(f"启动服务: {name}")
        logger.info(f"   描述: {description}")
        if port:
            logger.info(f"   端口: {port}")

        # 获取项目根目录
        project_root = self._get_project_root()

        # 确定工作目录
        work_dir = os.path.join(project_root, directory) if directory else project_root

        # 构建命令
        script_path = script

        # 处理相对路径
        if not os.path.isabs(script_path):
            script_path = os.path.join(work_dir, script_path)

        cmd = [sys.executable, script_path]
        cmd.extend(args)

        # 处理特殊命令（如npm、shell脚本等）
        if script.startswith("npm ") or script.startswith("node "):
            cmd = script.split()
            # Windows平台需要通过cmd执行npm
            if sys.platform == "win32":
                cmd = ["cmd", "/c"] + cmd
        elif script.startswith("-m "):
            cmd = [sys.executable] + script.split()
        elif script.endswith(".sh"):
            cmd = ["bash", script_path]

        try:
            env = self._build_env(env_vars)
            process = subprocess.Popen(cmd, cwd=work_dir, env=env)
            self._processes[name] = process

            # 前端服务启动较慢，需要更长的等待时间
            wait_time = 15 if name == "frontend" else 3
            time.sleep(wait_time)

            # 检查是否启动成功
            if process.poll() is None:
                # 等待端口就绪（如果有端口）
                if port:
                    # 前端服务端口等待时间也需要更长
                    timeout = 30 if name == "frontend" else 15
                    if self._wait_for_port(port, timeout=timeout):
                        logger.info(f"服务启动成功: {name}")
                        return True
                    else:
                        logger.error(f"服务端口未就绪: {name}")
                        process.terminate()
                        del self._processes[name]
                        return False
                else:
                    # 无端口服务，直接认为启动成功
                    logger.info(f"服务启动成功: {name}")
                    return True
            else:
                logger.error(f"服务启动失败，退出码: {process.returncode}")
                return False

        except Exception as e:
            logger.error(f"服务启动异常: {name}, 错误: {str(e)}")
            return False

    def start_all_services(self) -> int:
        """启动所有已注册的服务"""
        success_count = 0
        total_count = len([s for s in self._services.values() if s.get("enabled", True)])

        # 按依赖顺序启动
        for name in self._get_start_order():
            if self.start_service(name):
                success_count += 1
            # 服务间延迟
            time.sleep(2)

        logger.info(f"服务启动完成: {success_count}/{total_count}")
        return success_count

    def stop_service(self, name: str):
        """停止单个服务"""
        if name in self._processes:
            process = self._processes[name]
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"服务已停止: {name}")
            del self._processes[name]

    def stop_all_services(self):
        """停止所有服务"""
        for name in list(self._processes.keys()):
            self.stop_service(name)

    def get_service_status(self, name: str) -> str:
        """获取服务状态"""
        if name not in self._services:
            return "未注册"
        if name not in self._processes:
            return "未启动"

        process = self._processes[name]
        if process.poll() is None:
            return "运行中"
        else:
            return f"已退出 ({process.returncode})"

    def get_all_status(self) -> Dict[str, str]:
        """获取所有服务状态"""
        return {name: self.get_service_status(name) for name in self._services}

    def _is_port_in_use(self, port: int) -> bool:
        """检查端口是否被占用"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _wait_for_port(self, port: int, timeout: int = 30) -> bool:
        """等待端口就绪"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_port_in_use(port):
                return True
            time.sleep(1)
        return False

    def _build_env(self, extra_env: Dict[str, str] = None) -> Dict[str, str]:
        """构建环境变量"""
        env = dict(os.environ)
        project_root = self._get_project_root()
        env["PYTHONPATH"] = project_root
        env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

        # 添加自定义环境变量
        if extra_env:
            env.update(extra_env)

        return env

    def _get_project_root(self) -> str:
        """获取项目根目录"""
        import os

        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    def _get_start_order(self) -> List[str]:
        """获取服务启动顺序"""
        order = {
            "model-service": 1,
            "api-service": 2,
            "multimedia-service": 3,
            "search-service": 4,
            "api-gateway": 5,
            "monitor-dashboard": 6,
            "frontend": 7,
            "log-viewer": 8,
            "health-check": 9,
            "log-monitor": 10,
            "resource-monitor": 11,
        }
        return sorted(self._services.keys(), key=lambda x: order.get(x, 99))


# 全局服务注册实例
service_registry = ServiceRegistry()


def get_service_registry() -> ServiceRegistry:
    """获取服务注册实例"""
    return service_registry
