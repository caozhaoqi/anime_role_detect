#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一应用入口

提供：
- 服务启动/停止管理
- 配置初始化
- 日志系统初始化
- 优雅的信号处理
"""

import os
import sys
import signal
import argparse
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.config import get_config
from src.core.logging import get_logger
from src.core.service import get_service_registry

# 全局变量
logger = get_logger("application")
service_registry = get_service_registry()


def check_dependencies():
    """检查并安装依赖"""
    from src.core.dependency_manager import DependencyManager

    print("\n" + "=" * 60)
    print("🔍 检查依赖状态")
    print("=" * 60)

    manager = DependencyManager(mirror_priority=["tsinghua", "aliyun", "douban"])

    # 检查核心依赖
    if not manager.ensure_essential():
        print("\n📦 开始自动安装核心依赖...")
        success, fail = manager.install_core(verbose=True)
        print(f"\n安装完成: ✅ {success} 个, ❌ {fail} 个")

        if fail > 0:
            print("\n⚠️  部分依赖安装失败，应用可能无法正常运行")
            print("   请手动执行: python -m src.core.dependency_manager install-core")
    else:
        print("✅ 核心依赖检查通过")


def initialize():
    """初始化应用"""
    logger.info("=" * 60)
    logger.info("动漫角色识别系统 - 初始化")
    logger.info("=" * 60)

    # 自动检查并安装依赖
    check_dependencies()

    # 初始化配置
    config = get_config()
    logger.info(f"配置加载完成")

    # 注册服务
    services = config.get_all_services()
    service_registry.register_services(services)
    logger.info(f"已注册服务: {list(services.keys())}")

    logger.info("初始化完成")


def start_services(service_names=None):
    """启动服务"""
    logger.info("\n开始启动服务")

    if service_names:
        # 启动指定服务
        for name in service_names:
            service_registry.start_service(name)
            time.sleep(2)
    else:
        # 启动所有服务
        service_registry.start_all_services()

    logger.info("\n服务状态:")
    status = service_registry.get_all_status()
    for name, state in status.items():
        logger.info(f"   {name}: {state}")


def stop_services(signal_num, frame):
    """停止所有服务（信号处理）"""
    logger.info("\n收到停止信号，正在停止服务...")
    service_registry.stop_all_services()
    logger.info("所有服务已停止")
    sys.exit(0)


def show_status():
    """显示服务状态"""
    logger.info("\n当前服务状态:")
    status = service_registry.get_all_status()
    for name, state in status.items():
        logger.info(f"   {name}: {state}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="动漫角色识别系统 - 统一应用入口")
    parser.add_argument("action", choices=["start", "stop", "status"], help="操作类型")
    parser.add_argument("--services", nargs="+", help="指定服务名称")
    parser.add_argument("--core", action="store_true", help="仅启动核心服务")

    args = parser.parse_args()

    # 注册信号处理
    signal.signal(signal.SIGINT, stop_services)
    signal.signal(signal.SIGTERM, stop_services)

    # 初始化
    initialize()

    if args.action == "start":
        if args.core:
            # 仅启动核心服务
            logger.info("启动核心服务模式")
            service_registry.start_service("multimedia-service")
        elif args.services:
            # 启动指定服务
            start_services(args.services)
        else:
            # 启动所有服务
            start_services()

        # 保持运行
        logger.info("\n按 Ctrl+C 停止所有服务...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_services(None, None)

    elif args.action == "stop":
        service_registry.stop_all_services()

    elif args.action == "status":
        show_status()


if __name__ == "__main__":
    main()
