#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
萝莉角色数据采集脚本 v2
根据分类出的萝莉角色进行数据采集，支持角色名匹配和配置文件读取
"""

import os
import requests
from PIL import Image
import io
import time
import random
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('loli_data_collection_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置参数
LOLI_CHARACTERS_FILE = "./auto_spider_img/lolis/loli_characters.txt"
DATA_DIR = "./data/loli_training_data"
CONFIG_DIR = "./spider_image_system/data/config"
IMG_URL_DIR = "./spider_image_system/data/img_url"
MAX_IMAGES_PER_ROLE = 100  # 每个角色最多采集的图片数量
TIMEOUT = 15  # 请求超时时间（秒）
DELAY = 1  # 请求延迟时间（秒）
MAX_WORKERS = 5  # 最大并发数

# 创建数据目录
os.makedirs(DATA_DIR, exist_ok=True)

def is_valid_image(content):
    """检查是否为有效图片"""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except:
        return False

def download_image(url, save_dir, role_name, timeout=15):
    """下载单张图片"""
    try:
        headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
            ])
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            if is_valid_image(response.content):
                # 生成文件名
                url_hash = abs(hash(url)) % 1000000
                filename = f"{url_hash:06d}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                # 避免重复下载
                if os.path.exists(filepath):
                    return False, "文件已存在"
                
                # 保存图片
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                return True, f"{filename}"
            else:
                return False, "无效图片"
        else:
            return False, f"HTTP {response.status_code}"
            
    except Exception as e:
        return False, str(e)

def find_config_file(character):
    """查找角色的配置文件"""
    if not os.path.exists(CONFIG_DIR):
        return None
    
    # 尝试精确匹配
    config_file = f"{character.lower()}_config.json"
    if os.path.exists(os.path.join(CONFIG_DIR, config_file)):
        return config_file
    
    # 尝试模糊匹配
    for file in os.listdir(CONFIG_DIR):
        if file.endswith('_config.json'):
            config_name = file.replace('_config.json', '')
            if character.lower() in config_name.lower() or config_name.lower() in character.lower():
                return file
    
    return None

def collect_from_config(character):
    """从配置文件采集图片"""
    config_file = find_config_file(character)
    if not config_file:
        logger.info(f"角色 {character} 未找到配置文件")
        return []
    
    config_path = os.path.join(CONFIG_DIR, config_file)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"读取配置文件 {config_path} 失败: {e}")
        return []
    
    # 创建角色目录
    role_dir = os.path.join(DATA_DIR, character)
    os.makedirs(role_dir, exist_ok=True)
    
    # 统计现有图片数量
    existing_images = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if existing_images >= MAX_IMAGES_PER_ROLE:
        logger.info(f"角色 {character} 已有 {existing_images} 张图片，跳过采集")
        return []
    
    # 从配置中获取搜索URL
    search_url = config.get('search_url', '')
    if not search_url:
        logger.info(f"角色 {character} 配置文件中未找到搜索URL")
        return []
    
    logger.info(f"角色 {character}: 使用配置文件 {config_file}")
    logger.info(f"搜索URL: {search_url}")
    
    # 这里需要实现从搜索URL获取图片URL的逻辑
    # 由于需要解析网页，这里只是示例
    logger.info(f"角色 {character}: 从配置文件采集（功能待完善）")
    return []

def load_loli_characters():
    """加载萝莉角色列表"""
    if not os.path.exists(LOLI_CHARACTERS_FILE):
        logger.error(f"萝莉角色文件不存在: {LOLI_CHARACTERS_FILE}")
        return []
    
    characters = []
    try:
        with open(LOLI_CHARACTERS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                character = line.strip()
                if character:
                    characters.append(character)
    except Exception as e:
        logger.error(f"读取萝莉角色文件失败: {e}")
        return []
    
    logger.info(f"共加载 {len(characters)} 个萝莉角色")
    return characters

def collect_loli_data(characters):
    """采集萝莉角色数据
    
    Args:
        characters: 角色名列表
        
    Returns:
        dict: 角色名到图片路径列表的映射
    """
    results = {}
    
    for i, character in enumerate(characters, 1):
        logger.info(f"处理角色 {i}/{len(characters)}: {character}")
        
        # 从配置文件采集
        images = collect_from_config(character)
        
        if images:
            results[character] = images
            logger.info(f"角色 {character}: 共采集 {len(images)} 张图片")
        else:
            logger.warning(f"角色 {character}: 未采集到图片")
        
        # 延迟，避免请求过于频繁
        time.sleep(DELAY)
    
    return results

def main():
    """主函数"""
    print("=" * 60)
    print("萝莉角色数据采集 v2")
    print("=" * 60)
    
    # 加载萝莉角色
    characters = load_loli_characters()
    if not characters:
        print("未找到萝莉角色")
        return
    
    print(f"共加载 {len(characters)} 个萝莉角色")
    print(f"数据目录: {DATA_DIR}")
    print(f"每个角色最多采集 {MAX_IMAGES_PER_ROLE} 张图片")
    print()
    
    # 采集数据
    results = collect_loli_data(characters)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("数据采集完成")
    print("=" * 60)
    print(f"成功采集 {len(results)} 个角色的图片")
    
    total_images = sum(len(images) for images in results.values())
    print(f"共采集 {total_images} 张图片")
    
    if results:
        print("\n角色图片统计:")
        for character, images in results.items():
            print(f"  {character}: {len(images)} 张")
    
    print(f"\n数据已保存到: {DATA_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
