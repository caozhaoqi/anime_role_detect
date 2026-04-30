#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从爬虫系统的URL文件下载图片
"""

import os
import sys
import requests
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 配置
SPIDER_URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/href_url'
DOWNLOAD_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/role_images'
MAX_WORKERS = 5
TIMEOUT = 30
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建下载目录
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_image(url, save_path, retries=0):
    """下载单张图片"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36',
            'Referer': 'https://www.pixiv.net/'
        }
        response = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.debug(f"下载成功: {save_path}")
        return True, None
        
    except requests.exceptions.RequestException as e:
        retries += 1
        if retries >= MAX_RETRIES:
            return False, str(e)
        delay = 2 ** retries + random.uniform(0, 1)
        logger.warning(f"下载失败: {url}, 重试 {retries}/{MAX_RETRIES}, {delay:.2f}秒后重试")
        time.sleep(delay)
        return download_image(url, save_path, retries)
    except Exception as e:
        return False, str(e)


def process_role_url_file(url_file_path):
    """处理单个角色的URL文件"""
    # 从文件名提取角色拼音
    file_name = os.path.basename(url_file_path)
    if '_url.txt' not in file_name:
        return
    
    role_pinyin = file_name.replace('_url.txt', '')
    
    # 创建角色目录
    role_dir = os.path.join(DOWNLOAD_DIR, role_pinyin)
    os.makedirs(role_dir, exist_ok=True)
    
    # 读取URL列表
    with open(url_file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    if not urls:
        logger.info(f"角色 {role_pinyin} 的URL文件为空")
        return
    
    logger.info(f"开始下载角色 {role_pinyin} 的图片，共 {len(urls)} 个URL")
    
    # 下载图片
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for idx, url in enumerate(urls):
            # 生成保存路径
            ext = url.split('.')[-1].lower() if '.' in url else 'jpg'
            if ext not in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
                ext = 'jpg'
            save_path = os.path.join(role_dir, f"{role_pinyin}_{idx + 1}.{ext}")
            
            # 如果文件已存在，跳过
            if os.path.exists(save_path):
                success_count += 1
                continue
            
            future = executor.submit(download_image, url, save_path)
            futures[future] = (url, save_path)
        
        for future in as_completed(futures):
            url, save_path = futures[future]
            try:
                success, error = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    logger.error(f"下载失败: {url} - {error}")
            except Exception as e:
                fail_count += 1
                logger.error(f"下载异常: {url} - {str(e)}")
    
    logger.info(f"角色 {role_pinyin} 下载完成: 成功 {success_count} 张, 失败 {fail_count} 张")


def main():
    """主函数"""
    # 获取所有URL文件
    url_files = [f for f in os.listdir(SPIDER_URL_DIR) if f.endswith('_url.txt')]
    
    if not url_files:
        logger.warning("未找到URL文件")
        return
    
    logger.info(f"找到 {len(url_files)} 个URL文件")
    
    # 逐个处理
    for url_file in url_files:
        url_file_path = os.path.join(SPIDER_URL_DIR, url_file)
        logger.info(f"处理文件: {url_file}")
        process_role_url_file(url_file_path)
        # 添加间隔避免被封
        time.sleep(2)


if __name__ == '__main__':
    main()
