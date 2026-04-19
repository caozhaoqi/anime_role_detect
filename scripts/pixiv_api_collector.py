#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过Pixiv API采集萝莉角色图片并下载
"""

import os
import sys
import requests
import time
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("pixiv_collector")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pixiv_collector")

LOLI_FILE = "./auto_spider_img/classified_all_v2/萝莉.txt"
IMG_URL_DIR = "./spider_image_system/data/img_url"
OUTPUT_DIR = "./data/downloaded_images"
MAX_IMAGES = 50
TIMEOUT = 30

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': 'https://www.pixiv.net/',
    'Cookie': ''
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


def search_pixiv_api(keyword):
    """通过Pixiv API搜索图片"""
    img_urls = []

    try:
        # Pixiv API 搜索端点
        url = f"https://www.pixiv.net/ajax/search/artworks/{requests.utils.quote(keyword)}"
        params = {
            'word': keyword,
            'order': 'date_d',
            'mode': 'all',
            'p': 1,
            'type': 'all'
        }

        response = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)

        if response.status_code == 200:
            data = response.json()

            if 'body' in data and 'illustManga' in data['body']:
                items = data['body']['illustManga']['data']
                for item in items[:MAX_IMAGES]:
                    illust_id = item.get('id', '')
                    # 构建图片URL
                    if 'url' in item and item['url']:
                        img_urls.append(item['url'])
                    elif illust_id:
                        # 尝试构建URL
                        img_url = f"https://i.pximg.net/img-master/img/{item.get('createDate', '').replace('-', '/')}/{illust_id}_p0_master1200.jpg"
                        img_urls.append(img_url)

            logger.info(f"[{keyword}] API找到 {len(img_urls)} 个结果")
        else:
            logger.warning(f"[{keyword}] API请求失败: {response.status_code}")

    except Exception as e:
        logger.error(f"[{keyword}] 异常: {e}")

    return img_urls


def search_pixiv_html(keyword):
    """通过Pixiv HTML页面搜索图片（备选方案）"""
    img_urls = []

    try:
        url = f"https://www.pixiv.net/tags/{requests.utils.quote(keyword)}/artworks"

        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)

        if response.status_code == 200:
            import re
            # 匹配图片ID模式
            pattern = r'"id":"(\d+)","title"'
            matches = re.findall(pattern, response.text)

            for illust_id in matches[:MAX_IMAGES]:
                img_url = f"https://i.pximg.net/img-master/img/{illust_id}_p0_master1200.jpg"
                img_urls.append(img_url)

            logger.info(f"[{keyword}] HTML找到 {len(img_urls)} 个结果")
        else:
            logger.warning(f"[{keyword}] HTML请求失败: {response.status_code}")

    except Exception as e:
        logger.error(f"[{keyword}] HTML异常: {e}")

    return img_urls


def download_image(url, save_dir, role_name):
    """下载图片"""
    global stats

    try:
        filename = f"{abs(hash(url)) % 10000000:07d}.jpg"
        filepath = os.path.join(save_dir, filename)

        if os.path.exists(filepath):
            return True

        # 使用Pixiv Referer
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://www.pixiv.net/',
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

    # 尝试API搜索
    urls = search_pixiv_api(role_name)

    # 如果API失败，尝试HTML
    if not urls:
        urls = search_pixiv_html(role_name)

    if urls:
        save_urls_to_file(role_name, urls)

        # 下载图片
        pinyin_map = get_pinyin_map()
        pinyin = pinyin_map.get(role_name, role_name)
        save_dir = os.path.join(OUTPUT_DIR, pinyin)
        os.makedirs(save_dir, exist_ok=True)

        for url in urls[:10]:  # 限制下载数量
            download_image(url, save_dir, role_name)
            time.sleep(0.5)
    else:
        logger.warning(f"[{role_name}] 未找到图片")

    time.sleep(random.uniform(1, 2))


def main():
    logger.info("=" * 60)
    logger.info("开始通过Pixiv API采集萝莉角色图片")
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