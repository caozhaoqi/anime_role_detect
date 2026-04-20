#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
萝莉角色数据采集脚本
根据萝莉.txt中的角色下载图片
"""

import os
import sys
import requests
from PIL import Image
import io
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("collect_loli")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("collect_loli")

LOLI_FILE = "./auto_spider_img/classified_all_v2/萝莉.txt"
IMG_URL_DIR = "./spider_image_system/data/img_url"
OUTPUT_DIR = "./data/downloaded_images"
MAX_IMAGES_PER_ROLE = 100
TIMEOUT = 15
DELAY = 0.5
MAX_WORKERS = 3

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.pixiv.net/',
    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

lock = threading.Lock()
stats = {'success': 0, 'failed': 0, 'skipped': 0}


def is_valid_image(content):
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except:
        return False


def get_pinyin_mapping():
    return {
        '莉莉娅·阿琳': 'li4li4ya3_a1lin2',
        '青雀': 'qing1que4',
        '古明地恋': 'gu3ming2di4lian4',
        '梅比乌斯': 'mei2bi3wu3si1',
        '纳西妲': 'na4xi1da4',
        '雾雨魔理沙': 'wu4yu3mo2li3sha1',
        '可莉': 'ke3li2',
        '安卡希雅': 'an1ka3xi1ya3',
        '迪奥娜': 'di2ao4na4',
        '可琳·威克斯': 'ke3lin2_wei1ke4si1',
        '瑶瑶': 'yao2yao2',
        '希格雯': 'xi1ge2wen2',
        '蕾贝': 'lei3bei4',
        '黑塔': 'hei1ta3',
        '符玄': 'fu2xuan2',
        '菲谢尔': 'fei1xie4er3',
        '萝莎莉娅·阿琳': 'luo2sha1li4ya3_a1lin2',
        '丛雨': 'cong2yu3',
    }


def download_image(url, save_dir, role_name):
    global stats
    try:
        response = requests.get(url, timeout=TIMEOUT, stream=True, headers=HEADERS)
        if response.status_code != 200:
            return False

        if not is_valid_image(response.content):
            return False

        filename = f"{abs(hash(url)) % 10000000:07d}.jpg"
        filepath = os.path.join(save_dir, filename)

        if os.path.exists(filepath):
            with lock:
                stats['skipped'] += 1
            return True

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with lock:
            stats['success'] += 1
        return True

    except Exception as e:
        with lock:
            stats['failed'] += 1
        return False


def collect_role(role_name, pinyin_key):
    role_dir = os.path.join(OUTPUT_DIR, pinyin_key)
    os.makedirs(role_dir, exist_ok=True)

    existing = len([f for f in os.listdir(role_dir) if f.endswith('.jpg')])
    if existing >= MAX_IMAGES_PER_ROLE:
        logger.info(f"[{role_name}] 已有 {existing} 张图片，跳过")
        return

    img_file = os.path.join(IMG_URL_DIR, f"{pinyin_key}_img.txt")
    if not os.path.exists(img_file):
        logger.warning(f"[{role_name}] URL文件不存在: {img_file}")
        return

    with open(img_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip().startswith('http')]

    urls = urls[:MAX_IMAGES_PER_ROLE - existing]
    if not urls:
        return

    logger.info(f"[{role_name}] 开始下载 {len(urls)} 张图片...")
    success = 0
    for url in urls:
        if download_image(url, role_dir, role_name):
            success += 1
        time.sleep(DELAY)

    logger.info(f"[{role_name}] 完成，成功 {success} 张")


def main():
    logger.info("=" * 60)
    logger.info("开始采集萝莉角色图片")
    logger.info("=" * 60)

    if not os.path.exists(LOLI_FILE):
        logger.error(f"角色文件不存在: {LOLI_FILE}")
        return

    with open(LOLI_FILE, 'r', encoding='utf-8') as f:
        characters = [line.strip() for line in f if line.strip()]

    logger.info(f"加载了 {len(characters)} 个角色")

    pinyin_map = get_pinyin_mapping()

    for char in characters:
        pinyin = pinyin_map.get(char)
        if pinyin:
            collect_role(char, pinyin)
        else:
            logger.warning(f"未找到角色映射: {char}")

    logger.info("=" * 60)
    logger.info(f"采集完成: 成功 {stats['success']}, 失败 {stats['failed']}, 跳过 {stats['skipped']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()