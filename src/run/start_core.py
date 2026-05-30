#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动漫角色识别系统 - 核心服务启动脚本
只启动必需的核心服务，减少内存占用
"""

import os
import sys
import subprocess
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 核心服务配置（最小化内存占用）
CORE_SERVICES = [
    {
        "name": "多媒体服务",
        "script": "services/multimedia_service/multimedia_service_app.py",
        "port": 8002,
        "description": "图像搜索和视频识别",
    }
]


def log_info(msg):
    print(f"[\033[94mINFO\033[0m] {msg}")


def log_success(msg):
    print(f"[\033[92mSUCCESS\033[0m] {msg}")


def log_error(msg):
    print(f"[\033[91mERROR\033[0m] {msg}")


def start_core_service(service):
    """启动单个核心服务"""
    port = service["port"]
    script = service["script"]
    name = service["name"]

    log_info(f"🚀 启动 {name}...")
    log_info(f"   端口: {port}")

    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    script_path = os.path.join(project_root, "src", script)
    cmd = [sys.executable, script_path, "--port", str(port)]

    try:
        process = subprocess.Popen(cmd, cwd=project_root, env=env)

        # 等待启动
        time.sleep(5)

        if process.poll() is None:
            # 检查端口
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(("localhost", port))
            sock.close()

            if result == 0:
                log_success(f"✅ {name} 启动成功: http://localhost:{port}")
                return process
            else:
                log_error(f"❌ {name} 端口未就绪")
                process.terminate()
                return None
        else:
            log_error(f"❌ {name} 启动失败")
            return None
    except Exception as e:
        log_error(f"❌ {name} 启动异常: {str(e)}")
        return None


def main():
    """主函数"""
    print("=" * 60)
    print("🎬 动漫角色识别系统 - 核心服务启动")
    print("=" * 60)
    print("仅启动核心服务，减少内存占用")
    print("=" * 60)

    processes = []

    for service in CORE_SERVICES:
        proc = start_core_service(service)
        if proc:
            processes.append(proc)

    if processes:
        print("\n" + "=" * 60)
        log_success(f"🎉 核心服务启动完成！")
        print("=" * 60)
        print("\n按 Ctrl+C 停止服务...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 停止所有服务...")
            for proc in processes:
                proc.terminate()
                proc.wait()
    else:
        log_error("❌ 所有服务启动失败")


if __name__ == "__main__":
    main()
