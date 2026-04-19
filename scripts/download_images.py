#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从URL文件下载图片
"""

import os
import sys
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("download_images")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("download_images")

URL_FILE = './data/img_url/arona_img.txt'
OUTPUT_DIR = './data/downloaded_images/arona'
MAX_WORKERS = 5
TIMEOUT = 30

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.pixiv.net/',
    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

lock = threading.Lock()
downloaded_count = 0
failed_count = 0


def is_valid_image_url(url):
    """检查URL是否为有效的图片URL"""
    if not url.startswith('http'):
        return False
    parsed = urlparse(url)
    if 'vv50.de' in parsed.netloc or 'sd.vv50.de' in parsed.netloc:
        return False
    if not any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
        return False
    return True


def get_filename_from_url(url):
    """从URL提取文件名"""
    parsed = urlparse(url)
    path = parsed.path
    filename = os.path.basename(path)
    return filename if filename else f"img_{hash(url)}.jpg"


def download_image(url, output_dir):
    """下载单个图片"""
    global downloaded_count, failed_count

    if not is_valid_image_url(url):
        return False

    try:
        filename = get_filename_from_url(url)
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logger.info(f"文件已存在，跳过: {filename}")
            with lock:
                downloaded_count += 1
            return True

        response = requests.get(url, timeout=TIMEOUT, stream=True, headers=HEADERS)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with lock:
            downloaded_count += 1
        logger.info(f"下载成功: {filename}")
        return True

    except Exception as e:
        with lock:
            failed_count += 1
        logger.error(f"下载失败 [{url}]: {e}")
        return False


def main():
    global downloaded_count, failed_count

    logger.info("=" * 60)
    logger.info("开始下载图片")
    logger.info("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(URL_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    valid_urls = [url for url in urls if is_valid_image_url(url)]
    logger.info(f"有效图片URL数量: {len(valid_urls)} / {len(urls)}")

    if not valid_urls:
        logger.warning("没有找到有效的图片URL")
        return

    logger.info(f"下载目录: {OUTPUT_DIR}")
    logger.info(f"并发数: {MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_image, url, OUTPUT_DIR): url for url in valid_urls}

        for future in as_completed(futures):
            pass

    logger.info("=" * 60)
    logger.info(f"下载完成: 成功 {downloaded_count}, 失败 {failed_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()