#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为缺少数据的角色采集更多图像
"""

import os
import sys
import requests
import time
import random
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("character_collector")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("character_collector")

# 配置参数
DATA_DIR = './data/downloaded_images'
MIN_SAMPLES = 50  # 每个角色的最小样本数
MAX_IMAGES_PER_CHARACTER = 100  # 每个角色的最大图像数

# 用户代理列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
]

# 角色映射（中文名称 -> 英文目录名）
CHARACTER_MAPPING = {
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
    '希格雯': 'xi1ge2wen2',
    '瑶瑶': 'yao2yao2'
}

def get_image_count(directory):
    """获取目录中的图像数量"""
    if not os.path.exists(directory):
        return 0
    count = 0
    for file in os.listdir(directory):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            count += 1
    return count

def download_image(url, save_path):
    """下载图像"""
    try:
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')
        image.save(save_path, 'JPEG', quality=90)
        return True
    except Exception as e:
        logger.error(f"下载失败 {url}: {e}")
        return False

def search_images_bing(query, num_images=50):
    """使用Bing搜索图像"""
    images = []
    try:
        url = f"https://www.bing.com/images/search?q={query}&count={num_images}"
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for img in soup.find_all('img', class_='mimg'):
            if 'src' in img.attrs:
                images.append(img['src'])
            elif 'data-src' in img.attrs:
                images.append(img['data-src'])
    except Exception as e:
        logger.error(f"Bing搜索失败: {e}")
    return images[:num_images]

def collect_character_images(character_name, directory_name, target_count):
    """为单个角色采集图像"""
    output_dir = os.path.join(DATA_DIR, directory_name)
    os.makedirs(output_dir, exist_ok=True)
    
    current_count = get_image_count(output_dir)
    if current_count >= target_count:
        logger.info(f"{character_name} 已有 {current_count} 张图像，达到目标数量")
        return
    
    needed = target_count - current_count
    logger.info(f"为 {character_name} 采集 {needed} 张图像")
    
    # 生成搜索查询
    queries = [
        f"{character_name} 动漫",
        f"{character_name} 二次元",
        f"{character_name} 插画",
        f"{character_name} 角色",
        f"{character_name} fanart"
    ]
    
    collected = 0
    for query in queries:
        if collected >= needed:
            break
        
        logger.info(f"  搜索: {query}")
        image_urls = search_images_bing(query, num_images=20)
        
        for url in image_urls:
            if collected >= needed:
                break
            
            if url.startswith('http'):
                filename = f"{directory_name}_{collected + current_count:04d}.jpg"
                save_path = os.path.join(output_dir, filename)
                
                if download_image(url, save_path):
                    collected += 1
                    logger.info(f"  已下载 {collected}/{needed}")
                    time.sleep(random.uniform(1, 3))  # 避免被封禁
    
    final_count = get_image_count(output_dir)
    logger.info(f"{character_name} 采集完成，总计 {final_count} 张图像")

def main():
    logger.info("=" * 60)
    logger.info("开始采集缺少数据的角色图像")
    logger.info("=" * 60)
    
    # 分析当前数据状态
    logger.info("当前数据状态:")
    total_images = 0
    for character_cn, character_dir in CHARACTER_MAPPING.items():
        count = get_image_count(os.path.join(DATA_DIR, character_dir))
        total_images += count
        logger.info(f"  {character_cn}: {count} 张图像")
    
    logger.info(f"\n总计: {total_images} 张图像")
    
    # 为每个角色采集图像
    for character_cn, character_dir in CHARACTER_MAPPING.items():
        logger.info(f"\n" + "-" * 40)
        collect_character_images(character_cn, character_dir, MAX_IMAGES_PER_CHARACTER)
    
    # 最终统计
    logger.info("\n" + "=" * 60)
    logger.info("采集完成")
    logger.info("=" * 60)
    
    final_total = 0
    for character_cn, character_dir in CHARACTER_MAPPING.items():
        count = get_image_count(os.path.join(DATA_DIR, character_dir))
        final_total += count
        logger.info(f"{character_cn}: {count} 张图像")
    
    logger.info(f"\n最终总计: {final_total} 张图像")
    logger.info(f"平均每个角色: {final_total / len(CHARACTER_MAPPING):.1f} 张图像")

if __name__ == '__main__':
    main()
