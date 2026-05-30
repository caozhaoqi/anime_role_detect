#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目主入口模块
提供统一的命令行接口
"""

import argparse
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.common import setup_logger, DownloadConfig


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="动漫角色检测项目 - 数据采集与管理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用子命令:
  download    - 下载图片
  collect     - 数据采集
  clean       - 数据清理
  stats       - 统计分析
  help        - 显示帮助信息
        """,
    )

    parser.add_argument(
        "command", choices=["download", "collect", "clean", "stats", "help"], help="要执行的命令"
    )

    parser.add_argument("--config", "-c", default=None, help="配置文件路径")

    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出模式")

    args = parser.parse_args()

    logger = setup_logger("main")

    if args.verbose:
        import logging

        logger.setLevel(logging.DEBUG)

    logger.info(f"执行命令: {args.command}")

    try:
        if args.command == "download":
            run_download(args)
        elif args.command == "collect":
            run_collect(args)
        elif args.command == "clean":
            run_clean(args)
        elif args.command == "stats":
            run_stats(args)
        elif args.command == "help":
            parser.print_help()
    except Exception as e:
        logger.error(f"命令执行失败: {e}")
        sys.exit(1)


def run_download(args):
    """执行下载命令"""
    from scripts.data_collection.downloaders.smart_downloader import SmartDownloader

    logger = setup_logger("download")
    logger.info("开始图片下载...")

    downloader = SmartDownloader()
    downloader.download_all()

    logger.info("图片下载完成")


def run_collect(args):
    """执行采集命令"""
    from scripts.data_collection.batch_collectors.batch_collect_data import main as collect_main

    logger = setup_logger("collect")
    logger.info("开始数据采集...")

    collect_main()

    logger.info("数据采集完成")


def run_clean(args):
    """执行清理命令"""
    from scripts.data_cleaning.clean_collection import main as clean_main

    logger = setup_logger("clean")
    logger.info("开始数据清理...")

    clean_main()

    logger.info("数据清理完成")


def run_stats(args):
    """执行统计命令"""
    logger = setup_logger("stats")
    logger.info("生成统计报告...")

    # 统计数据目录中的图片数量
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "organized_images"
    )

    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        return

    total_images = 0
    total_roles = 0
    role_stats = []

    for role_dir in os.listdir(data_dir):
        role_path = os.path.join(data_dir, role_dir)
        if os.path.isdir(role_path):
            images = [
                f
                for f in os.listdir(role_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            count = len(images)
            if count > 0:
                total_images += count
                total_roles += 1
                role_stats.append((role_dir, count))

    # 按图片数量排序
    role_stats.sort(key=lambda x: x[1], reverse=True)

    logger.info("=" * 60)
    logger.info("📊 数据统计报告")
    logger.info("=" * 60)
    logger.info(f"总角色数: {total_roles}")
    logger.info(f"总图片数: {total_images}")
    logger.info(f"平均每角色图片数: {total_images / total_roles:.1f}")
    logger.info("-" * 60)
    logger.info("前10个角色图片数量:")

    for role, count in role_stats[:10]:
        logger.info(f"  {role}: {count} 张")

    if len(role_stats) > 10:
        logger.info(f"  ... 还有 {len(role_stats) - 10} 个角色")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
