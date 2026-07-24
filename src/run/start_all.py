#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动漫角色识别系统 - 统一启动脚本
将所有服务集中管理，合理分配端口

端口分配计划:
- 8000: 模型服务 (model_service)
- 8001: 主API服务 (api_service)
- 8002: 多媒体服务 (multimedia_service) - 整合图像搜索和视频识别
- 8080: API网关 (api_gateway)
"""

import os
import sys
import time
import signal
import subprocess
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.run.services_config import SERVICES, SERVICE_GROUPS

# 全局进程列表
processes = {}


def is_port_available(port):
    """检查端口是否可用"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def start_service(service_key):
    """启动单个服务"""
    config = SERVICES[service_key]
    print(f"\n🚀 启动 {config['name']}...")
    print(f"   描述: {config['description']}")
    print(f"   端口: {config['port']}")

    # 检查端口是否可用
    if not is_port_available(config["port"]):
        print(f"   ⚠️ 端口 {config['port']} 已被占用，跳过启动")
        return True

    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root

    # 前端服务特殊处理 - 使用npm启动
    if service_key == "frontend":
        frontend_dir = os.path.join(project_root, "src/frontend")
        
        # 检查npm是否可用
        try:
            subprocess.run(["npm", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print(f"   ❌ npm 命令不可用，请确保已安装 Node.js")
            return False

        # 检查前端目录是否存在
        if not os.path.exists(frontend_dir):
            print(f"   ❌ 前端目录不存在: {frontend_dir}")
            return False

        # 启动前端服务
        cmd = ["npm", "run", "dev"]
        process = subprocess.Popen(cmd, cwd=frontend_dir, env=env)
        processes[service_key] = process

        # 等待服务启动（前端启动较慢）
        time.sleep(10)

        # 检查是否启动成功
        if process.poll() is None:
            print(f"   ✅ {config['name']} 启动成功")
            print(f"   🌐 访问地址: http://localhost:{config['port']}")
            return True
        else:
            print(f"   ❌ {config['name']} 启动失败")
            return False

    # 普通Python服务
    cmd = [sys.executable, config["script"]]
    process = subprocess.Popen(cmd, cwd=project_root, env=env)
    processes[service_key] = process

    # 等待服务启动
    time.sleep(3)

    # 检查是否启动成功
    if process.poll() is None:
        print(f"   ✅ {config['name']} 启动成功")
        return True
    else:
        print(f"   ❌ {config['name']} 启动失败")
        return False


def stop_all_services(signal_num=None, frame=None):
    """停止所有服务"""
    print("\n🛑 正在停止所有服务...")
    for key, process in processes.items():
        if process.poll() is None:
            print(f"   停止 {SERVICES[key]['name']}...")
            process.terminate()
            process.wait()
            print(f"   ✅ {SERVICES[key]['name']} 已停止")
    sys.exit(0)


def check_service_health():
    """检查服务健康状态"""
    try:
        import requests

        print("\n📊 服务健康检查")
        print("-" * 50)

        for key, config in SERVICES.items():
            if processes.get(key) and processes[key].poll() is None:
                try:
                    health_path = config.get("health_path", "/health")
                    response = requests.get(
                        f"http://localhost:{config['port']}{health_path}", timeout=2
                    )
                    if response.status_code == 200:
                        print(f"✅ {config['name']}: http://localhost:{config['port']}")
                    else:
                        print(f"⚠️ {config['name']}: 状态码 {response.status_code}")
                except Exception as e:
                    print(f"❌ {config['name']}: {e}")
    except ImportError:
        pass


def print_api_docs():
    """打印API文档分类"""
    print("\n📚 API文档分类")
    print("=" * 60)

    api_categories = {
        "搜索服务 API": {
            "base_url": f"http://localhost:{SERVICES['multimedia_service']['port']}",
            "description": "以图搜图和视频识别服务（核心）",
            "endpoints": [
                {
                    "method": "POST",
                    "path": "/search/image",
                    "description": "以图搜图，上传图像搜索相似角色",
                },
                {
                    "method": "POST",
                    "path": "/search/build-index",
                    "description": "构建FAISS搜索索引",
                },
                {"method": "GET", "path": "/search/stats", "description": "获取索引统计信息"},
                {
                    "method": "POST",
                    "path": "/search/video/recognize",
                    "description": "视频实时抽帧识别",
                },
                {"method": "GET", "path": "/health", "description": "健康检查"},
            ],
        },
        "主API服务 API": {
            "base_url": f"http://localhost:{SERVICES['api_service']['port']}",
            "description": "主API网关，聚合所有服务（核心）",
            "endpoints": [
                {"method": "POST", "path": "/api/search/image", "description": "以图搜图"},
                {
                    "method": "POST",
                    "path": "/api/search/video/recognize",
                    "description": "视频识别",
                },
                {"method": "POST", "path": "/api/classify", "description": "图像分类"},
                {"method": "GET", "path": "/api/health", "description": "健康检查"},
                {"method": "GET", "path": "/api/roles", "description": "获取角色列表"},
            ],
        },
        "模型服务 API": {
            "base_url": f"http://localhost:{SERVICES['model_service']['port']}",
            "description": "AI模型推理服务",
            "endpoints": [
                {"method": "POST", "path": "/api/model/predict", "description": "模型预测"},
                {"method": "POST", "path": "/api/model/classify", "description": "角色分类"},
                {"method": "GET", "path": "/api/health", "description": "健康检查"},
            ],
        },
        "视频识别 API": {
            "base_url": f"http://localhost:{SERVICES['multimedia_service']['port']}",
            "description": "多媒体服务 - 视频处理功能",
            "endpoints": [
                {"method": "POST", "path": "/video/recognize", "description": "视频抽帧识别"},
                {"method": "POST", "path": "/video/extract", "description": "视频抽帧"},
                {"method": "GET", "path": "/video/stats", "description": "视频统计"},
            ],
        },
        "API网关服务": {
            "base_url": f"http://localhost:{SERVICES['api_gateway']['port']}",
            "description": "统一API网关，聚合所有后端服务",
            "endpoints": [
                {"method": "POST", "path": "/api/v1/search", "description": "统一搜索接口"},
                {"method": "POST", "path": "/api/v1/classify", "description": "统一分类接口"},
                {"method": "GET", "path": "/health", "description": "健康检查"},
            ],
        },
    }

    for category, info in api_categories.items():
        print(f"\n📖 {category}")
        print(f"   基础URL: {info['base_url']}")
        print(f"   Swagger文档: {info['base_url']}/docs")
        print("   端点列表:")
        for endpoint in info["endpoints"]:
            print(f"     [{endpoint['method']}] {endpoint['path']}")
            print(f"         {endpoint['description']}")


def print_port_plan():
    """打印端口分配计划"""
    print("\n📋 端口分配计划")
    print("=" * 60)
    print("┌────────────────────┬───────┬─────────────────────────────────────┐")
    print("│ 服务名称           │ 端口  │ 描述                               │")
    print("├────────────────────┼───────┼─────────────────────────────────────┤")

    for key, config in SERVICES.items():
        print(f"│ {config['name']:16} │ {config['port']:5} │ {config['description']:39} │")

    print("└────────────────────┴───────┴─────────────────────────────────────┘")


def start_monitor():
    """启动监控仪表板"""
    import subprocess

    print("\n🚀 启动监控仪表板...")
    print("   描述: 统一监控所有服务状态")
    print("   端口: 9000")

    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root

    # 启动监控服务
    cmd = [sys.executable, os.path.join(project_root, "src/run/monitor/monitor_dashboard.py")]
    process = subprocess.Popen(cmd, cwd=project_root, env=env)
    processes["monitor"] = process

    # 等待服务启动
    time.sleep(3)

    # 检查是否启动成功
    if process.poll() is None:
        print("   ✅ 监控仪表板启动成功")
        print(f"   📊 访问地址: http://localhost:9000")
        return True
    else:
        print("   ❌ 监控仪表板启动失败")
        return False


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="启动动漫角色识别系统服务")
    parser.add_argument(
        "--group",
        "-g",
        default="all",
        choices=["all", "core", "ai", "video", "gateway"],
        help="选择要启动的服务分组",
    )
    parser.add_argument("--list", "-l", action="store_true", help="列出所有服务和端口配置")
    args = parser.parse_args()

    print("=" * 60)
    print("🎬 动漫角色识别系统 - 统一启动脚本")
    print("=" * 60)

    # 如果只是列出服务配置
    if args.list:
        print_port_plan()
        return

    # 获取要启动的服务列表
    service_keys = SERVICE_GROUPS.get(args.group, ["all"])

    # 打印启动信息
    print(f"\n📦 启动服务分组: {args.group}")
    print("将启动以下服务:")
    for key in service_keys:
        config = SERVICES.get(key)
        if config:
            print(f"   • {config['name']} (端口: {config['port']})")

    # 注册信号处理器
    signal.signal(signal.SIGINT, stop_all_services)
    signal.signal(signal.SIGTERM, stop_all_services)

    # 逐个启动服务
    success_count = 0
    for key in service_keys:
        config = SERVICES.get(key)
        if not config or not config.get("enabled", True):
            continue
        if start_service(key):
            success_count += 1

    # 启动监控仪表板
    if start_monitor():
        success_count += 1

    # 检查健康状态
    check_service_health()

    # 打印端口计划
    print_port_plan()

    # 打印API文档
    print_api_docs()

    print(f"\n🎉 服务启动完成！成功启动 {success_count}/{len(service_keys) + 1} 个服务")
    print(f"\n📊 监控仪表板: http://localhost:9000")
    print("按 Ctrl+C 停止所有服务...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_all_services()


if __name__ == "__main__":
    main()
