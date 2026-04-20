#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过spider_image_system的API采集萝莉角色图片并下载
"""

import os
import sys
import requests
import time
import random
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("sis_api_collector")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("sis_api_collector")

LOLI_FILE = "./auto_spider_img/classified_all_v2/萝莉.txt"
IMG_URL_DIR = "./spider_image_system/data/img_url"
OUTPUT_DIR = "./data/downloaded_images"
MAX_IMAGES = 30
API_BASE = "http://localhost:33333/api/v1.2.5.260305"
TIMEOUT = 30

stats = {'saved': 0, 'downloaded': 0, 'failed': 0}


def get_pinyin_map():
    return {
        '莉莉娅·阿琳': 'li4li4ya4·a1lin2',
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
        '丛雨': 'cong2yu3'
    }


def check_api_server():
    """检查API服务是否运行"""
    try:
        resp = requests.get(f"{API_BASE}/sis/spider_image/config", timeout=5)
        if resp.status_code == 200:
            return True
    except:
        pass
    return False


def start_sis_server():
    """启动SIS API服务"""
    sis_path = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/src"

    logger.info("正在启动spider_image_system API服务...")

    proc = subprocess.Popen(
        [sys.executable, "-m", "run.sis_main_process"],
        cwd=sis_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # 等待服务启动
    for i in range(30):
        time.sleep(1)
        if check_api_server():
            logger.info("SIS API服务已启动")
            return proc
        logger.info(f"等待服务启动... ({i+1}/30)")

    logger.error("SIS API服务启动失败")
    return None


def spider_single_keyword(keyword):
    """通过API爬取单个关键词的图片URL"""
    try:
        resp = requests.post(
            f"{API_BASE}/sis/spider_start/single",
            params={"key_word": keyword},
            timeout=60
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        logger.error(f"API调用失败 [{keyword}]: {e}")
        return None


def get_config():
    """获取配置"""
    try:
        resp = requests.get(f"{API_BASE}/sis/spider_image/config", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
    return None


def download_image(url, save_dir, role_name):
    """下载图片"""
    try:
        filename = f"{abs(hash(url)) % 10000000:07d}.jpg"
        filepath = os.path.join(save_dir, filename)

        if os.path.exists(filepath):
            return True

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://www.pixiv.net/',
        }

        response = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)

        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True

    except Exception as e:
        logger.error(f"下载失败 [{url}]: {e}")

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

    new_urls = [u for u in urls if u and u.startswith('http') and u not in existing]

    if new_urls:
        with open(filepath, 'a', encoding='utf-8') as f:
            for url in new_urls:
                f.write(url + '\n')
        logger.info(f"[{role_name}] 保存了 {len(new_urls)} 个URL")

        return len(new_urls)

    return 0


def process_role(role_name):
    """处理单个角色"""
    global stats

    logger.info(f"正在处理: {role_name}")

    # 通过API爬取
    result = spider_single_keyword(role_name)

    if result:
        logger.info(f"[{role_name}] API调用成功")
    else:
        logger.warning(f"[{role_name}] API调用失败")

    # 等待爬取完成
    time.sleep(5)

    # 下载图片
    pinyin_map = get_pinyin_map()
    pinyin = pinyin_map.get(role_name, role_name)
    save_dir = os.path.join(OUTPUT_DIR, pinyin)
    os.makedirs(save_dir, exist_ok=True)

    # 读取已保存的URL
    img_file = os.path.join(IMG_URL_DIR, f"{pinyin}_img.txt")
    if os.path.exists(img_file):
        with open(img_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and line.startswith('http')]

        logger.info(f"[{role_name}] 从URL文件加载了 {len(urls)} 个URL")

        for url in urls[:MAX_IMAGES]:
            if download_image(url, save_dir, role_name):
                stats['downloaded'] += 1
            time.sleep(0.5)

    time.sleep(random.uniform(2, 4))


def main():
    logger.info("=" * 60)
    logger.info("通过spider_image_system API采集萝莉角色图片")
    logger.info("=" * 60)

    # 检查API服务
    if not check_api_server():
        logger.info("API服务未运行，正在启动...")
        proc = start_sis_server()
        if not proc:
            logger.error("无法启动API服务")
            return
    else:
        logger.info("API服务已在运行")
        proc = None

    # 获取当前配置
    config = get_config()
    if config:
        logger.info(f"当前配置: visit_url = {config.get('spider_config', {}).get('visit_url', 'unknown')}")

    if not os.path.exists(LOLI_FILE):
        logger.error(f"角色文件不存在: {LOLI_FILE}")
        return

    with open(LOLI_FILE, 'r', encoding='utf-8') as f:
        characters = [line.strip() for line in f if line.strip()]

    logger.info(f"加载了 {len(characters)} 个角色")

    for char in characters:
        process_role(char)

    logger.info("=" * 60)
    logger.info(f"完成: 下载 {stats['downloaded']} 张图片")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()