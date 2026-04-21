#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过spider_image_system的API为缺少数据的角色采集图片
"""

import os
import sys
import requests
import time
import random
import json
import subprocess
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("api_collector")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("api_collector")

# 配置参数
IMG_URL_DIR = "./spider_image_system/data/img_url"
OUTPUT_DIR = "./data/downloaded_images"
MAX_IMAGES_PER_CHARACTER = 50
API_BASE = "http://localhost:33333/api/v1.2.5.260305"
TIMEOUT = 30

# 要采集的角色（中文名称 -> 英文目录名）
CHARACTERS_TO_COLLECT = {
    '琮玉': 'cong2yu3',
    '迪奥娜': 'di2ao4na4',
    '菲谢尔': 'fei1xie4er3',
    '符玄': 'fu2xuan2',
    '孤明德莲': 'gu3ming2di4lian4',
    '黑塔': 'hei1ta3',
    '可莉': 'ke3li2',
    '科林·维克斯': 'ke3lin2_wei1ke4si1',
    '莉莉娅·艾琳': 'li4li4ya3_a1lin2',
    '罗莎莉娅·艾琳': 'luo2sha1li4ya3_a1lin2',
    '梅比乌斯': 'mei2bi3wu3si1',
    '纳西妲': 'na4xi1da4',
    '西格温': 'xi1ge2wen2',
    '瑶瑶': 'yao2yao2'
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
        filename = f"api_{abs(hash(url)) % 10000000:07d}.jpg"
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

def save_urls_to_file(role_name, pinyin, urls):
    """保存URL到文件"""
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

def get_image_count(directory):
    """获取目录中的图像数量"""
    if not os.path.exists(directory):
        return 0
    count = 0
    for file in os.listdir(directory):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            count += 1
    return count

def process_character(role_name, pinyin):
    """处理单个角色"""
    logger.info(f"正在处理: {role_name} ({pinyin})")

    # 检查当前图像数量
    save_dir = os.path.join(OUTPUT_DIR, pinyin)
    os.makedirs(save_dir, exist_ok=True)
    current_count = get_image_count(save_dir)
    
    if current_count >= MAX_IMAGES_PER_CHARACTER:
        logger.info(f"[{role_name}] 已有 {current_count} 张图像，达到目标数量")
        return

    needed = MAX_IMAGES_PER_CHARACTER - current_count
    logger.info(f"[{role_name}] 需要采集 {needed} 张图像")

    # 生成搜索关键词
    keywords = [
        role_name,
        f"{role_name} 动漫",
        f"{role_name} 二次元",
        f"{role_name} 插画",
        f"{role_name} 角色"
    ]

    total_urls = []
    for keyword in keywords:
        logger.info(f"[{role_name}] 正在搜索: {keyword}")
        
        # 通过API爬取
        result = spider_single_keyword(keyword)
        
        if result:
            logger.info(f"[{role_name}] API调用成功")
        else:
            logger.warning(f"[{role_name}] API调用失败")
        
        # 等待爬取完成
        time.sleep(3)

    # 读取已保存的URL
    img_file = os.path.join(IMG_URL_DIR, f"{pinyin}_img.txt")
    if os.path.exists(img_file):
        with open(img_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and line.startswith('http')]
        
        logger.info(f"[{role_name}] 从URL文件加载了 {len(urls)} 个URL")
        
        # 下载图片
        downloaded = 0
        for url in urls[:needed]:
            if download_image(url, save_dir, role_name):
                downloaded += 1
                if downloaded >= needed:
                    break
            time.sleep(0.5)
        
        logger.info(f"[{role_name}] 成功下载 {downloaded} 张图像")
    else:
        logger.warning(f"[{role_name}] 未找到URL文件")

    # 最终统计
    final_count = get_image_count(save_dir)
    logger.info(f"[{role_name}] 采集完成，总计 {final_count} 张图像")
    
    time.sleep(random.uniform(2, 4))

def main():
    logger.info("=" * 60)
    logger.info("通过API采集缺少数据的角色图像")
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

    # 处理每个角色
    for role_name, pinyin in CHARACTERS_TO_COLLECT.items():
        logger.info(f"\n" + "-" * 40)
        process_character(role_name, pinyin)

    # 最终统计
    logger.info("\n" + "=" * 60)
    logger.info("采集完成")
    logger.info("=" * 60)
    
    total_images = 0
    for role_name, pinyin in CHARACTERS_TO_COLLECT.items():
        save_dir = os.path.join(OUTPUT_DIR, pinyin)
        count = get_image_count(save_dir)
        total_images += count
        logger.info(f"{role_name}: {count} 张图像")
    
    logger.info(f"\n最终总计: {total_images} 张图像")
    logger.info(f"平均每个角色: {total_images / len(CHARACTERS_TO_COLLECT):.1f} 张图像")

if __name__ == "__main__":
    main()
