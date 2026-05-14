#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接从文件系统下载最新URL（使用公共模块重构版）
直接从 spider_image_system/data/img_url/ 目录读取URL文件进行下载
"""

import os
import sys
import threading
import time
from pathlib import Path
from datetime import datetime

# 添加公共模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))

from download_utils import (
    setup_logger,
    download_image,
    load_urls_from_file,
    DownloadStats
)
from notification_utils import ProgressNotifier

# 配置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
URL_DIR = PROJECT_ROOT / "spider_image_system" / "data" / "img_url"
OUTPUT_DIR = PROJECT_ROOT / "data" / "organized_images"
CONFIG_PATH = PROJECT_ROOT / "scripts" / "notification_config.json"


class Downloader:
    def __init__(self):
        self.logger = setup_logger("download_from_files")
        self.notifier = ProgressNotifier(interval=300)
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = DownloadStats()
        self.lock = threading.Lock()
        self.running = True
        self.paused = False
        self.pause_event = threading.Event()

    def get_all_urls_from_files(self) -> list:
        """从所有URL文件加载URL"""
        all_urls = []
        for url_file in URL_DIR.glob("*_img.txt"):
            role_name = url_file.stem.replace('_img', '')
            urls = load_urls_from_file(str(url_file))
            all_urls.extend([(role_name, url) for url in urls])
        return all_urls

    def download_image_with_stats(self, role_name: str, url: str) -> bool:
        """下载单张图片并更新统计"""
        if not self.running:
            return False
        if self.paused:
            self.pause_event.wait()
        
        role_dir = self.output_dir / role_name
        role_dir.mkdir(exist_ok=True)
        
        success, message = download_image(url, str(role_dir))
        
        with self.lock:
            if success:
                self.stats.downloaded += 1
                return True
            elif message == "文件已存在":
                self.stats.skipped += 1
            else:
                self.stats.failed += 1
        
        return False

    def download_worker(self, role_name: str, urls: list):
        """下载工作线程"""
        for url in urls:
            if not self.running:
                break
            if self.download_image_with_stats(role_name, url):
                if self.stats.downloaded % 50 == 0:
                    self.logger.info(f"进度: 已下载 {self.stats.downloaded}, 跳过 {self.stats.skipped}, 失败 {self.stats.failed}")
            time.sleep(0.1)

    def run(self):
        """运行下载任务"""
        self.logger.info("=" * 60)
        self.logger.info("🚀 从文件系统下载最新URL")
        self.logger.info("=" * 60)

        all_urls = self.get_all_urls_from_files()

        if not all_urls:
            self.logger.warning("未找到任何URL文件!")
            return

        # 按角色分组
        role_urls = {}
        for role_name, url in all_urls:
            if role_name not in role_urls:
                role_urls[role_name] = []
            role_urls[role_name].append(url)

        total_urls = len(all_urls)
        total_roles = len(role_urls)
        self.logger.info(f"📊 总计: {total_urls} 个URL, {total_roles} 个角色")

        # 发送开始通知
        self.notifier.send_message(
            f"🚀 开始下载!\n📊 总计: {total_urls} 个URL, {total_roles} 个角色\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # 创建并启动线程
        threads = []
        for role_name, urls in role_urls.items():
            t = threading.Thread(target=self.download_worker, args=(role_name, urls))
            t.start()
            threads.append(t)
            time.sleep(0.05)

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 输出结果
        self.logger.info("=" * 60)
        self.logger.info("✅ 下载完成!")
        self.logger.info(f"已下载: {self.stats.downloaded}")
        self.logger.info(f"跳过(已存在): {self.stats.skipped}")
        self.logger.info(f"失败: {self.stats.failed}")
        self.logger.info("=" * 60)

        # 发送完成通知
        self.notifier.send_message(
            f"✅ 下载完成!\n"
            f"📥 已下载: {self.stats.downloaded}\n"
            f"⏭️ 跳过: {self.stats.skipped}\n"
            f"❌ 失败: {self.stats.failed}\n"
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


def main():
    downloader = Downloader()
    downloader.run()


if __name__ == '__main__':
    main()
