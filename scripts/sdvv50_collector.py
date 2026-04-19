#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从sd.vv50.de采集萝莉角色图片URL并下载
"""

import os
import sys
import requests
import re
import time
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("sdvv50_collector")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("sdvv50_collector")

LOLI_FILE = "./auto_spider_img/classified_all_v2/萝莉.txt"
IMG_URL_DIR = "./spider_image_system/data/img_url"
OUTPUT_DIR = "./data/downloaded_images"
MAX_IMAGES = 50
TIMEOUT = 30

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0',
}

lock = threading.Lock()
stats = {'saved': 0, 'downloaded': 0, 'failed': 0}


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


def search_sdvv50(keyword):
    """从sd.vv50.de搜索图片URL"""
    img_urls = []

    try:
        # 搜索URL
        search_url = f"https://sd.vv50.de/search?q={requests.utils.quote(keyword)}"

        session = requests.Session()
        session.headers.update(HEADERS)

        # 先访问主页获取cookie
        session.get("https://sd.vv50.de/", timeout=TIMEOUT)
        time.sleep(1)

        # 再搜索
        response = session.get(search_url, timeout=TIMEOUT)

        if response.status_code == 200:
            # 解析图片URL
            patterns = [
                r'src="(https://i\.pximg\.net/[^"]+master1200\.jpg)"',
                r'data-src="(https://i\.pximg\.net/[^"]+master1200\.jpg)"',
                r'"url":"(https://i\.pximg\.net/[^"]+master1200\.jpg)"',
                r'"original":"(https://i\.pximg\.net/[^"]+master1200\.jpg)"',
            ]

            found = set()
            for pattern in patterns:
                matches = re.findall(pattern, response.text)
                for m in matches:
                    if m not in found and 'master1200' in m:
                        found.add(m)
                        img_urls.append(m)

            logger.info(f"[{keyword}] 找到 {len(img_urls)} 个图片URL")

        elif response.status_code == 400:
            logger.warning(f"[{keyword}] 请求被拒绝(400)，尝试其他方式")
            # 尝试用.artworks/方式
            artwork_url = f"https://sd.vv50.de/artworks/{requests.utils.quote(keyword)}"
            resp = session.get(artwork_url, timeout=TIMEOUT)
            if resp.status_code == 200:
                matches = re.findall(r'src="(https://i\.pximg\.net/[^"]+master1200\.jpg)"', resp.text)
                img_urls.extend(matches)
        else:
            logger.warning(f"[{keyword}] 状态码: {response.status_code}")

    except Exception as e:
        logger.error(f"[{keyword}] 异常: {e}")

    return list(set(img_urls))


def download_image(url, save_dir, role_name):
    """下载图片"""
    global stats

    try:
        filename = f"{abs(hash(url)) % 10000000:07d}.jpg"
        filepath = os.path.join(save_dir, filename)

        if os.path.exists(filepath):
            return True

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://sd.vv50.de/',
        }

        response = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)

        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            with lock:
                stats['downloaded'] += 1
            return True

    except Exception as e:
        logger.error(f"下载失败 [{url}]: {e}")

    with lock:
        stats['failed'] += 1
    return False


def save_urls_to_file(role_name, urls):
    """保存URL到文件"""
    pinyin_map = get_pinyin_map()
    pinyin = pinyin_map.get(role_name)
    if not pinyin:
        logger.warning(f"未找到角色映射: {role_name}")
        return 0

    os.makedirs(IMG_URL_DIR, exist_ok=True)
    filepath = os.path.join(IMG_URL_DIR, f"{pinyin}_img.txt")

    existing = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            existing = set(line.strip() for line in f if line.strip())

    new_urls = [u for u in urls if u and u not in existing]

    if new_urls:
        with open(filepath, 'a', encoding='utf-8') as f:
            for url in new_urls:
                f.write(url + '\n')

        with lock:
            stats['saved'] += len(new_urls)

        logger.info(f"[{role_name}] 保存了 {len(new_urls)} 个URL")

    return len(new_urls)


def process_role(role_name):
    """处理单个角色"""
    logger.info(f"正在处理: {role_name}")

    # 搜索图片URL
    urls = search_sdvv50(role_name)

    if urls:
        # 保存URL
        save_urls_to_file(role_name, urls)

        # 下载图片
        pinyin_map = get_pinyin_map()
        pinyin = pinyin_map.get(role_name, role_name)
        save_dir = os.path.join(OUTPUT_DIR, pinyin)
        os.makedirs(save_dir, exist_ok=True)

        for url in urls[:MAX_IMAGES]:
            download_image(url, save_dir, role_name)
            time.sleep(random.uniform(0.5, 1.5))
    else:
        logger.warning(f"[{role_name}] 未找到图片")

    time.sleep(random.uniform(2, 4))


def main():
    logger.info("=" * 60)
    logger.info("从sd.vv50.de采集萝莉角色图片")
    logger.info("=" * 60)

    if not os.path.exists(LOLI_FILE):
        logger.error(f"角色文件不存在: {LOLI_FILE}")
        return

    with open(LOLI_FILE, 'r', encoding='utf-8') as f:
        characters = [line.strip() for line in f if line.strip()]

    logger.info(f"加载了 {len(characters)} 个角色")

    for char in characters:
        process_role(char)

    logger.info("=" * 60)
    logger.info(f"完成: 保存URL {stats['saved']}, 下载 {stats['downloaded']}, 失败 {stats['failed']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()