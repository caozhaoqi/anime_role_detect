#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多角色数据采集扩展系统
为多个角色同时采集数据
"""

import os
import requests
from PIL import Image
import io
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 配置参数
DATA_DIR = "./data/downloaded_images"
TARGET_ROLES = ["阿罗娜", "日奈", "普拉娜", "亚子", "伊织", "千夏", "伊吕波", "阿露", "睦月", "佳代子"]
TARGET_IMAGES_PER_ROLE = 200
MAX_THREADS = 8

# 搜索关键词映射
SEARCH_KEYWORDS = {
    "阿罗娜": ["arona", "アロナ", "阿罗娜", "blue archive arona"],
    "日奈": ["hina", "ヒナ", "日奈", "blue archive hina"],
    "普拉娜": ["plana", "プラナ", "普拉娜", "blue archive plana"],
    "亚子": ["ako", "アコ", "亚子", "blue archive ako"],
    "伊织": ["iori", "イオリ", "伊织", "blue archive iori"],
    "千夏": ["chinatsu", "チナツ", "千夏", "blue archive chinatsu"],
    "伊吕波": ["iroha", "イロハ", "伊吕波", "blue archive iroha"],
    "阿露": ["aru", "アル", "阿露", "blue archive aru"],
    "睦月": ["mutsuki", "ムツキ", "睦月", "blue archive mutsuki"],
    "佳代子": ["kayoko", "カヨコ", "佳代子", "blue archive kayoko"]
}

# 图片搜索API（示例）
IMAGE_SEARCH_APIS = [
    "https://api.example.com/search?q={keyword}&type=image",
    "https://search.example.com/images?query={keyword}"
]

def is_valid_image(content):
    """检查是否为有效图片"""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except:
        return False

def download_image(url, save_dir, role_name):
    """下载单张图片"""
    try:
        headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ])
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            if is_valid_image(response.content):
                # 生成文件名
                url_hash = abs(hash(url)) % 1000000
                filename = f"{url_hash:06d}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                # 避免重复
                if os.path.exists(filepath):
                    return False, "文件已存在"
                
                # 保存图片
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                return True, filename
            else:
                return False, "无效图片"
        else:
            return False, f"HTTP {response.status_code}"
            
    except Exception as e:
        return False, str(e)

def collect_from_existing_sources():
    """从现有数据源采集"""
    urls = []
    
    # 从现有URL文件收集
    url_files = [
        "./data/img_url/arona_img.txt",
        "./data/href_url/arona_result_url.txt",
        "./data/href_url/arona_url.txt"
    ]
    
    for url_file in url_files:
        if os.path.exists(url_file):
            with open(url_file, 'r', encoding='utf-8') as f:
                file_urls = [line.strip() for line in f if line.strip()]
                urls.extend(file_urls)
    
    return list(set(urls))  # 去重

def classify_url_to_role(url, role_keywords):
    """将URL分类到对应角色"""
    url_lower = url.lower()
    
    for role_name, keywords in role_keywords.items():
        for keyword in keywords:
            if keyword.lower() in url_lower:
                return role_name
    
    return None

def process_role_data_collection(role_name, urls):
    """为单个角色采集数据"""
    print(f"\n开始为 {role_name} 采集数据...")
    
    # 创建角色目录
    role_dir = os.path.join(DATA_DIR, role_name)
    os.makedirs(role_dir, exist_ok=True)
    
    # 检查现有数量
    existing_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  现有图片: {existing_count} 张")
    
    if existing_count >= TARGET_IMAGES_PER_ROLE:
        print(f"  已达到目标数量")
        return 0
    
    needed = TARGET_IMAGES_PER_ROLE - existing_count
    print(f"  需要采集: {needed} 张")
    
    # 筛选相关URL
    role_keywords = SEARCH_KEYWORDS.get(role_name, [])
    relevant_urls = []
    
    for url in urls:
        classified_role = classify_url_to_role(url, SEARCH_KEYWORDS)
        if classified_role == role_name:
            relevant_urls.append(url)
    
    print(f"  找到相关URL: {len(relevant_urls)} 个")
    
    if not relevant_urls:
        print(f"  没有找到相关URL")
        return 0
    
    # 下载图片
    downloaded_count = 0
    failed_count = 0
    
    for i, url in enumerate(relevant_urls[:needed], start=1):
        if downloaded_count >= needed:
            break
        
        success, result = download_image(url, role_dir, role_name)
        
        if success:
            downloaded_count += 1
            print(f"  ✓ [{i}/{min(needed, len(relevant_urls))}] {result}")
        else:
            failed_count += 1
            if "文件已存在" not in result:
                print(f"  ✗ [{i}/{min(needed, len(relevant_urls))}] {result}")
        
        time.sleep(0.1)
    
    print(f"  成功下载: {downloaded_count} 张")
    print(f"  下载失败: {failed_count} 张")
    
    return downloaded_count

def main():
    """主函数"""
    print("=" * 60)
    print("多角色数据采集扩展系统")
    print("=" * 60)
    
    # 从现有数据源采集URL
    print("\n从现有数据源采集URL...")
    all_urls = collect_from_existing_sources()
    print(f"总共收集到 {len(all_urls)} 个URL")
    
    # 为每个角色采集数据
    total_downloaded = 0
    
    for role_name in TARGET_ROLES:
        downloaded = process_role_data_collection(role_name, all_urls)
        total_downloaded += downloaded
    
    # 最终统计
    print("\n" + "=" * 60)
    print("数据采集完成")
    print("=" * 60)
    print(f"总共下载: {total_downloaded} 张图片")
    
    print("\n各角色统计:")
    for role_name in TARGET_ROLES:
        role_dir = os.path.join(DATA_DIR, role_name)
        if os.path.exists(role_dir):
            count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {role_name}: {count} 张")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
