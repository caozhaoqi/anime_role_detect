#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
萝莉角色数据采集脚本
根据分类出的萝莉角色进行数据采集
"""

import os
import requests
from PIL import Image
import io
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('loli_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置参数
LOLI_CHARACTERS_FILE = "./auto_spider_img/lolis/loli_characters.txt"
DATA_DIR = "./data/loli_training_data"
SPIDER_DATA_DIR = "./spider_image_system/data/img_url"
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

def collect_from_spider_data(role_name):
    """从spider_image_system数据源采集图片"""
    # 查找匹配的角色文件
    spider_files = []
    if os.path.exists(SPIDER_DATA_DIR):
        for file in os.listdir(SPIDER_DATA_DIR):
            if file.endswith('_img.txt'):
                spider_files.append(file)
    
    # 尝试匹配角色名
    matched_files = []
    for spider_file in spider_files:
        # 简单的匹配逻辑：检查文件名是否包含角色名的一部分
        if role_name.lower() in spider_file.lower():
            matched_files.append(spider_file)
    
    if not matched_files:
        logger.info(f"角色 {role_name} 未找到匹配的spider数据文件")
        return []
    
    # 创建角色目录
    role_dir = os.path.join(DATA_DIR, role_name)
    os.makedirs(role_dir, exist_ok=True)
    
    # 统计现有图片数量
    existing_images = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if existing_images >= MAX_IMAGES_PER_ROLE:
        logger.info(f"角色 {role_name} 已有 {existing_images} 张图片，跳过采集")
        return []
    
    # 下载图片
    urls = []
    for matched_file in matched_files:
        file_path = os.path.join(SPIDER_DATA_DIR, matched_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_urls = [line.strip() for line in f if line.strip()]
                urls.extend(file_urls)
        except Exception as e:
            logger.error(f"读取文件 {file_path} 失败: {e}")
    
    if not urls:
        logger.info(f"角色 {role_name} 未找到图片URL")
        return []
    
    # 限制图片数量
    urls = urls[:MAX_IMAGES_PER_ROLE - existing_images]
    
    # 下载图片
    success_count = 0
    fail_count = 0
    
    for i, url in enumerate(urls):
        success, message = download_image(url, role_dir, role_name)
        if success:
            success_count += 1
            logger.info(f"角色 {role_name}: 下载成功 ({success_count}/{len(urls)}) - {message}")
        else:
            fail_count += 1
            logger.warning(f"角色 {role_name}: 下载失败 ({fail_count}/{len(urls)}) - {message}")
        
        # 延迟，避免请求过于频繁
        time.sleep(DELAY)
    
    logger.info(f"角色 {role_name}: 采集完成，成功 {success_count} 张，失败 {fail_count} 张")
    return [os.path.join(role_dir, f) for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

def collect_from_web_search(role_name):
    """通过网络搜索采集图片"""
    # 创建角色目录
    role_dir = os.path.join(DATA_DIR, role_name)
    os.makedirs(role_dir, exist_ok=True)
    
    # 统计现有图片数量
    existing_images = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if existing_images >= MAX_IMAGES_PER_ROLE:
        logger.info(f"角色 {role_name} 已有 {existing_images} 张图片，跳过采集")
        return []
    
    # 使用百度图片搜索
    search_url = f"https://image.baidu.com/search/index?tn=baiduimage&word={role_name}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers, timeout=TIMEOUT)
        
        if response.status_code == 200:
            # 解析HTML，提取图片URL
            # 这里需要根据实际的HTML结构进行解析
            # 由于百度图片的HTML结构比较复杂，这里只是示例
            logger.info(f"角色 {role_name}: 从网络搜索采集（功能待完善）")
            return []
        else:
            logger.error(f"角色 {role_name}: 网络搜索失败，状态码 {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"角色 {role_name}: 网络搜索异常: {e}")
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

def collect_loli_data(characters, use_spider=True, use_web=False):
    """采集萝莉角色数据
    
    Args:
        characters: 角色名列表
        use_spider: 是否使用spider_image_system数据源
        use_web: 是否使用网络搜索
        
    Returns:
        dict: 角色名到图片路径列表的映射
    """
    results = {}
    
    for i, character in enumerate(characters, 1):
        logger.info(f"处理角色 {i}/{len(characters)}: {character}")
        
        images = []
        
        if use_spider:
            spider_images = collect_from_spider_data(character)
            images.extend(spider_images)
        
        if use_web:
            web_images = collect_from_web_search(character)
            images.extend(web_images)
        
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
    print("萝莉角色数据采集")
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
    
    # 询问数据源
    use_spider = input("是否使用spider_image_system数据源 (y/n): ").strip().lower() == "y"
    use_web = input("是否使用网络搜索 (y/n): ").strip().lower() == "y"
    
    if not use_spider and not use_web:
        print("必须选择至少一个数据源")
        return
    
    # 采集数据
    results = collect_loli_data(characters, use_spider, use_web)
    
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
