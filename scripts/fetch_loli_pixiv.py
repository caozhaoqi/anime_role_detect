#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接从pixiv镜像站获取萝莉角色图片URL
"""

import os
import sys
import requests
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("fetch_pixiv_urls")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fetch_pixiv_urls")

LOLI_FILE = "./auto_spider_img/classified_all_v2/萝莉.txt"
IMG_URL_DIR = "./spider_image_system/data/img_url"
OUTPUT_DIR = "./data/downloaded_images"
MAX_IMAGES = 100
TIMEOUT = 15

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'https://www.pixiv.net/',
}

lock = threading.Lock()
stats = {'saved': 0, 'failed': 0}


def get_pinyin_map():
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


def search_pixiv_sdvv50(keyword):
    """从sd.vv50.de搜索图片"""
    img_urls = []

    try:
        # 使用URL编码
        encoded_keyword = requests.utils.quote(keyword)
        url = f"https://sd.vv50.de/search?q={encoded_keyword}"

        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)

        if response.status_code == 200:
            # 匹配图片URL模式
            patterns = [
                r'src="(https://i\.pximg\.net/[^"]+master1200\.jpg)"',
                r'data-src="(https://i\.pximg\.net/[^"]+master1200\.jpg)"',
                r'"(https://sd\.vv50\.de/artworks/\d+)"',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, response.text)
                img_urls.extend(matches)

            # 去重
            img_urls = list(dict.fromkeys(img_urls))

            logger.info(f"[{keyword}] 找到 {len(img_urls)} 个URL")

        else:
            logger.warning(f"[{keyword}] 搜索失败: {response.status_code}")

    except Exception as e:
        logger.error(f"[{keyword}] 异常: {e}")

    return img_urls


def save_to_file(role_name, urls):
    """保存URL到文件"""
    pinyin_map = get_pinyin_map()
    pinyin = pinyin_map.get(role_name)
    if not pinyin:
        logger.warning(f"未找到角色映射: {role_name}")
        return 0

    os.makedirs(IMG_URL_DIR, exist_ok=True)
    filepath = os.path.join(IMG_URL_DIR, f"{pinyin}_img.txt")

    # 读取已存在的URL
    existing = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            existing = set(line.strip() for line in f if line.strip())

    # 添加新URL
    new_urls = [u for u in urls if u not in existing]

    if new_urls:
        with open(filepath, 'a', encoding='utf-8') as f:
            for url in new_urls:
                f.write(url + '\n')
        logger.info(f"[{role_name}] 保存了 {len(new_urls)} 个新URL")

    return len(new_urls)


def main():
    logger.info("=" * 60)
    logger.info("开始获取萝莉角色图片URL")
    logger.info("=" * 60)

    if not os.path.exists(LOLI_FILE):
        logger.error(f"角色文件不存在: {LOLI_FILE}")
        return

    with open(LOLI_FILE, 'r', encoding='utf-8') as f:
        characters = [line.strip() for line in f if line.strip()]

    logger.info(f"加载了 {len(characters)} 个角色")

    total_saved = 0

    for char in characters:
        logger.info(f"正在搜索: {char}")
        urls = search_pixiv_sdvv50(char)
        if urls:
            saved = save_to_file(char, urls)
            total_saved += saved
        time.sleep(random.uniform(1, 2))

    logger.info("=" * 60)
    logger.info(f"完成: 共保存 {total_saved} 个URL")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()