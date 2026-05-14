#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从URL文件下载图片（使用公共模块重构版）
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加公共模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))

from download_utils import (
    setup_logger,
    is_valid_image_url,
    download_image,
    load_urls_from_file,
    DownloadStats
)

# 配置
URL_FILE = './data/img_url/arona_img.txt'
OUTPUT_DIR = './data/downloaded_images/arona'
MAX_WORKERS = 5


def main():
    logger = setup_logger("download_images")
    
    logger.info("=" * 60)
    logger.info("开始下载图片")
    logger.info("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载URL列表
    urls = load_urls_from_file(URL_FILE)
    logger.info(f"有效图片URL数量: {len(urls)}")
    
    if not urls:
        logger.warning("没有找到有效的图片URL")
        return
    
    logger.info(f"下载目录: {OUTPUT_DIR}")
    logger.info(f"并发数: {MAX_WORKERS}")
    
    # 初始化统计
    stats = DownloadStats()
    
    # 批量下载
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_image, url, OUTPUT_DIR): url for url in urls}
        
        for future in as_completed(futures):
            success, message = future.result()
            if success:
                stats.downloaded += 1
                logger.info(f"下载成功: {message}")
            elif message == "文件已存在":
                stats.skipped += 1
            else:
                stats.failed += 1
                logger.warning(f"下载失败: {message}")
    
    logger.info("=" * 60)
    logger.info(f"下载完成: 成功 {stats.downloaded}, 跳过 {stats.skipped}, 失败 {stats.failed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
