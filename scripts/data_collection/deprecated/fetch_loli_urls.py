#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从sd.vv50.de获取萝莉角色的图片URL
"""

import os
import sys
import requests
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("fetch_loli_urls")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fetch_loli_urls")

LOLI_FILE = "./auto_spider_img/loli-role.txt"
IMG_URL_DIR = "./spider_image_system/data/img_url"
HREF_URL_DIR = "./spider_image_system/data/href_url"
MAX_WORKERS = 3

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

PIXIV_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.pixiv.net/',
    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

lock = threading.Lock()
stats = {'found': 0, 'failed': 0}


def search_pixiv_for_role(role_name):
    """从pixiv搜索角色图片"""
    urls = []
    try:
        search_url = f"https://www.pixiv.net/tags/{requests.utils.quote(role_name)}/artworks"
        response = requests.get(search_url, headers=HEADERS, timeout=15)

        if response.status_code == 200:
            import re
            img_pattern = r'https://i\.pximg\.net/c/\d+x\d+/img-master/img/\d+/\d+/\d+/\d+/\d+/\d+/(\d+)_p\d+_master1200\.jpg'
            matches = re.findall(img_pattern, response.text)
            for match in matches[:50]:
                urls.append(f"https://pi.326688.xyz/img-master/img/??/{match}_p0_master1200.jpg")

        time.sleep(random.uniform(1, 2))
    except Exception as e:
        logger.error(f"[{role_name}] 搜索失败: {e}")

    return role_name, urls


def get_url_from_sdvv50(role_name):
    """从sd.vv50.de获取图片URL"""
    img_urls = []
    href_urls = []

    try:
        search_url = f"https://sd.vv50.de/search?q={requests.utils.quote(role_name)}"
        response = requests.get(search_url, headers=HEADERS, timeout=15)

        if response.status_code == 200:
            import re
            href_pattern = r'href="(https://sd\.vv50\.de/artworks/\d+)"'
            img_pattern = r'src="(https://i\.pximg\.net/[^"]+master1200\.jpg)"'

            hrefs = re.findall(href_pattern, response.text)
            imgs = re.findall(img_pattern, response.text)

            href_urls.extend(hrefs)
            img_urls.extend(imgs)

        time.sleep(random.uniform(0.5, 1.5))
    except Exception as e:
        logger.error(f"[{role_name}] sd.vv50.de搜索失败: {e}")

    return role_name, href_urls, img_urls


def save_urls(role_name, href_urls, img_urls):
    global stats

    pinyin_map = {
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

    pinyin = pinyin_map.get(role_name)
    if not pinyin:
        # 如果没有拼音映射，使用角色名作为文件名
        pinyin = role_name
        logger.info(f"使用角色名作为文件名: {role_name}")

    os.makedirs(IMG_URL_DIR, exist_ok=True)
    os.makedirs(HREF_URL_DIR, exist_ok=True)

    img_file = os.path.join(IMG_URL_DIR, f"{pinyin}_img.txt")
    href_file = os.path.join(HREF_URL_DIR, f"{pinyin}_url.txt")

    with open(img_file, 'w', encoding='utf-8') as f:
        for url in img_urls:
            f.write(url + '\n')

    with open(href_file, 'w', encoding='utf-8') as f:
        for url in href_urls:
            f.write(url + '\n')

    with lock:
        stats['found'] += 1

    logger.info(f"[{role_name}] 保存了 {len(img_urls)} 个图片URL, {len(href_urls)} 个链接")


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

    for char in characters:
        logger.info(f"正在搜索: {char}")
        name, hrefs, imgs = get_url_from_sdvv50(char)
        save_urls(name, hrefs, imgs)
        time.sleep(random.uniform(0.5, 1))

    logger.info("=" * 60)
    logger.info(f"完成: 处理 {stats['found']} 个角色")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()