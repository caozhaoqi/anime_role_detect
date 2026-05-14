#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接从文件系统下载最新URL - 带进度显示（使用公共模块重构版）
"""

import os
import sys
import time
import threading

# 添加公共模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))

from download_utils import (
    setup_logger,
    download_image,
    load_urls_from_file,
    load_local_hashes,
    DownloadStats
)
from notification_utils import ProgressNotifier

# 配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
URL_DIR = os.path.join(PROJECT_ROOT, "spider_image_system", "data", "img_url")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "organized_images")


def main():
    logger = setup_logger("download_simple")
    notifier = ProgressNotifier(interval=300)
    
    logger.info("=" * 60)
    logger.info("🚀 从文件系统下载最新URL")
    logger.info("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载本地哈希（用于去重）
    local_hashes = load_local_hashes(OUTPUT_DIR)
    logger.info(f"已加载 {len(local_hashes)} 个本地图片哈希")
    
    # 获取所有URL文件
    all_urls = []
    for url_file in os.listdir(URL_DIR):
        if url_file.endswith('_img.txt'):
            role_name = url_file.replace('_img.txt', '')
            url_file_path = os.path.join(URL_DIR, url_file)
            urls = load_urls_from_file(url_file_path)
            all_urls.extend([(role_name, url) for url in urls])
    
    if not all_urls:
        logger.warning("未找到任何URL文件!")
        return
    
    # 按角色分组
    role_urls = {}
    for role_name, url in all_urls:
        if role_name not in role_urls:
            role_urls[role_name] = []
        role_urls[role_name].append(url)
    
    total_urls = len(all_urls)
    total_roles = len(role_urls)
    logger.info(f"📊 总计: {total_urls} 个URL, {total_roles} 个角色")
    logger.info(f"📁 输出目录: {OUTPUT_DIR}")
    
    # 发送开始通知
    notifier.send_message(
        f"🚀 开始下载!\n📊 总计: {total_urls} 个URL, {total_roles} 个角色\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    # 初始化统计
    stats = DownloadStats()
    lock = threading.Lock()
    running = True
    
    def download_worker(role_name, urls):
        nonlocal stats
        role_dir = os.path.join(OUTPUT_DIR, role_name)
        os.makedirs(role_dir, exist_ok=True)
        
        for url in urls:
            if not running:
                break
            
            success, message = download_image(url, role_dir)
            
            with lock:
                if success:
                    stats.downloaded += 1
                elif message == "文件已存在":
                    stats.skipped += 1
                else:
                    stats.failed += 1
    
    # 创建并启动线程
    threads = []
    for role_name, urls in role_urls.items():
        t = threading.Thread(target=download_worker, args=(role_name, urls))
        t.start()
        threads.append(t)
        time.sleep(0.02)
    
    # 等待完成并显示进度
    last_log_time = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(2)
        if time.time() - last_log_time > 30:
            logger.info(f"进度: 已下载 {stats.downloaded}, 跳过 {stats.skipped}, 失败 {stats.failed}")
            last_log_time = time.time()
    
    for t in threads:
        t.join()
    
    # 输出结果
    logger.info("=" * 60)
    logger.info("✅ 下载完成!")
    logger.info(f"已下载: {stats.downloaded}")
    logger.info(f"跳过(已存在): {stats.skipped}")
    logger.info(f"失败: {stats.failed}")
    logger.info("=" * 60)
    
    # 发送完成通知
    notifier.send_message(
        f"✅ 下载完成!\n"
        f"📥 已下载: {stats.downloaded}\n"
        f"⏭️ 跳过: {stats.skipped}\n"
        f"❌ 失败: {stats.failed}\n"
        f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == '__main__':
    main()
