#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 spider_image_system/data/img_url 目录下载图片（使用公共模块重构版）
"""

import os
import sys
import time

# 添加公共模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))

from download_utils import (
    setup_logger,
    is_valid_image_url,
    download_image,
    load_urls_from_file,
    DownloadConfig
)

# 配置
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'
DOWNLOAD_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/role_images'

# 使用公共配置类
config = DownloadConfig(
    download_dir=DOWNLOAD_DIR,
    max_workers=5,
    timeout=30,
    max_retries=3,
    delay=1.0
)


def process_img_url_file(url_file_path):
    """处理单个角色的图片URL文件"""
    logger = setup_logger("download_from_img_url")
    
    # 从文件名提取角色拼音
    file_name = os.path.basename(url_file_path)
    if '_img.txt' not in file_name:
        return
    
    role_pinyin = file_name.replace('_img.txt', '')
    
    # 创建角色目录
    role_dir = os.path.join(config.download_dir, role_pinyin)
    os.makedirs(role_dir, exist_ok=True)
    
    # 读取URL列表并过滤
    urls = load_urls_from_file(url_file_path)
    
    # 过滤无效URL
    valid_urls = [url for url in urls if is_valid_image_url(url)]
    
    if not valid_urls:
        logger.info(f"角色 {role_pinyin} 没有有效的图片URL")
        return
    
    logger.info(f"开始下载角色 {role_pinyin} 的图片，共 {len(valid_urls)} 个有效URL")
    
    # 下载图片
    success_count = 0
    fail_count = 0
    
    for idx, url in enumerate(valid_urls):
        success, message = download_image(
            url, 
            role_dir,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        
        if success:
            success_count += 1
            logger.debug(f"下载成功: {message}")
        elif message == "文件已存在":
            success_count += 1
        else:
            fail_count += 1
            logger.warning(f"下载失败: {url[:50]}... - {message}")
    
    logger.info(f"角色 {role_pinyin} 下载完成: 成功 {success_count} 张, 失败 {fail_count} 张")


def main():
    """主函数"""
    logger = setup_logger("download_from_img_url")
    
    # 获取所有URL文件
    if not os.path.exists(URL_DIR):
        logger.error(f"URL目录不存在: {URL_DIR}")
        return
    
    url_files = [f for f in os.listdir(URL_DIR) if f.endswith('_img.txt')]
    
    if not url_files:
        logger.warning("未找到URL文件")
        return
    
    logger.info(f"找到 {len(url_files)} 个URL文件")
    
    # 逐个处理
    for url_file in url_files:
        url_file_path = os.path.join(URL_DIR, url_file)
        logger.info(f"处理文件: {url_file}")
        process_img_url_file(url_file_path)
        # 添加间隔避免被封
        time.sleep(config.delay)


if __name__ == '__main__':
    main()
