#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级数据采集脚本
从多个URL源下载图片并自动分类到不同的角色目录
支持多种数据源和智能分类
"""

import os
import requests
from pathlib import Path
from urllib.parse import urlparse, urljoin
from PIL import Image
import io
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# 配置参数
DATA_DIR = "./data/downloaded_images"
URL_SOURCES = [
    "./data/img_url/arona_img.txt",
    "./data/href_url/arona_result_url.txt",
    "./data/href_url/arona_url.txt"
]

# 角色配置
ROLES = {
    "a1luo2na4": {
        "dir": "a1luo2na4",
        "keywords": ["arona", "阿罗娜", "a1luo2na4", "alona"],
        "chinese_name": "阿罗娜"
    },
    "ri4nai4": {
        "dir": "ri4nai4", 
        "keywords": ["rinai", "日奈", "ri4nai4", "rinai"],
        "chinese_name": "日奈"
    },
    "plana": {
        "dir": "plana",
        "keywords": ["plana", "普拉娜", "plana"],
        "chinese_name": "普拉娜"
    }
}

# 创建目录
for role_name, role_config in ROLES.items():
    role_dir = os.path.join(DATA_DIR, role_config["dir"])
    os.makedirs(role_dir, exist_ok=True)

# 下载统计
download_stats = {
    "total_urls": 0,
    "successful_downloads": 0,
    "failed_downloads": 0,
    "invalid_images": 0,
    "role_distribution": {role_name: 0 for role_name in ROLES.keys()}
}

def is_valid_image(content):
    """检查是否为有效的图片"""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except:
        return False

def classify_url(url):
    """根据URL分类图片到对应角色"""
    url_lower = url.lower()
    
    for role_name, role_config in ROLES.items():
        for keyword in role_config["keywords"]:
            if keyword.lower() in url_lower:
                return role_name
    
    # 默认分类到阿罗娜
    return "a1luo2na4"

def extract_image_urls_from_page(page_url):
    """从页面URL中提取图片URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(page_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # 使用正则表达式提取图片URL
            img_pattern = r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|gif|webp)'
            img_urls = re.findall(img_pattern, response.text)
            
            # 过滤掉小图标和其他不相关的图片
            filtered_urls = []
            for url in img_urls:
                if any(size in url.lower() for size in ['master', 'original', 'large']):
                    if not any(exclude in url.lower() for exclude in ['icon', 'logo', 'thumb', 'small']):
                        filtered_urls.append(url)
            
            return filtered_urls[:10]  # 限制每个页面最多10张图片
            
    except Exception as e:
        print(f"  ✗ 提取图片URL失败: {page_url} - {str(e)}")
    
    return []

def download_image(url, save_dir, role_name, timeout=15):
    """下载单张图片"""
    try:
        # 发送请求
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            # 检查是否为有效图片
            if is_valid_image(response.content):
                # 生成文件名
                url_hash = abs(hash(url)) % 100000
                filename = f"{role_name}_{url_hash:05d}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                # 避免重复下载
                if os.path.exists(filepath):
                    return False, "文件已存在"
                
                # 保存图片
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                chinese_name = ROLES[role_name]["chinese_name"]
                return True, f"{filename} ({chinese_name})"
            else:
                return False, "无效图片"
        else:
            return False, f"HTTP {response.status_code}"
            
    except Exception as e:
        return False, str(e)

def process_url_file(file_path):
    """处理单个URL文件"""
    print(f"\n处理文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"  ✗ 文件不存在")
        return []
    
    urls = []
    with open(file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"  找到 {len(urls)} 个URL")
    
    # 区分直接图片URL和页面URL
    image_urls = []
    page_urls = []
    
    for url in urls:
        if url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            image_urls.append(url)
        else:
            page_urls.append(url)
    
    print(f"  直接图片URL: {len(image_urls)}")
    print(f"  页面URL: {len(page_urls)}")
    
    all_urls = image_urls.copy()
    
    # 从页面URL中提取图片URL
    if page_urls:
        print(f"  正在从页面提取图片URL...")
        for page_url in page_urls[:20]:  # 限制处理20个页面
            extracted_urls = extract_image_urls_from_page(page_url)
            all_urls.extend(extracted_urls)
            time.sleep(0.5)  # 避免请求过快
    
    print(f"  总共提取到 {len(all_urls)} 个图片URL")
    return all_urls

def main():
    """主函数"""
    print("=" * 60)
    print("开始高级数据采集")
    print("=" * 60)
    
    # 收集所有URL
    all_urls = []
    for url_source in URL_SOURCES:
        urls = process_url_file(url_source)
        all_urls.extend(urls)
    
    # 去重
    all_urls = list(set(all_urls))
    print(f"\n去重后总共 {len(all_urls)} 个唯一URL")
    
    # 统计现有图片数量
    print("\n现有数据统计:")
    for role_name, role_config in ROLES.items():
        role_dir = os.path.join(DATA_DIR, role_config["dir"])
        existing_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {role_config['chinese_name']}: {existing_count} 张")
        download_stats["role_distribution"][role_name] = existing_count
    
    # 下载图片
    print(f"\n开始下载 {len(all_urls)} 张图片...")
    
    # 使用多线程下载
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for i, url in enumerate(all_urls, start=1):
            # 分类URL到对应角色
            role_name = classify_url(url)
            save_dir = os.path.join(DATA_DIR, ROLES[role_name]["dir"])
            
            # 提交下载任务
            future = executor.submit(download_image, url, save_dir, role_name)
            futures.append((future, i, len(all_urls), url))
        
        # 处理下载结果
        for future, current, total, url in futures:
            try:
                success, result = future.result(timeout=30)
                
                if success:
                    print(f"✓ [{current}/{total}] 下载成功: {result}")
                    download_stats["successful_downloads"] += 1
                    # 更新角色分布统计
                    role_name = result.split('(')[1].split(')')[0]
                    for role_key, role_config in ROLES.items():
                        if role_config["chinese_name"] == role_name:
                            download_stats["role_distribution"][role_key] += 1
                            break
                else:
                    if "无效图片" in result:
                        download_stats["invalid_images"] += 1
                    download_stats["failed_downloads"] += 1
                    print(f"✗ [{current}/{total}] 下载失败: {url} - {result}")
                
                download_stats["total_urls"] += 1
                
                # 避免请求过快
                time.sleep(0.2)
                
            except Exception as e:
                download_stats["failed_downloads"] += 1
                print(f"✗ [{current}/{total}] 下载错误: {url} - {str(e)}")
    
    # 统计结果
    print()
    print("=" * 60)
    print("数据采集完成")
    print("=" * 60)
    print(f"处理URL总数: {download_stats['total_urls']}")
    print(f"成功下载: {download_stats['successful_downloads']} 张")
    print(f"无效图片: {download_stats['invalid_images']} 张")
    print(f"下载失败: {download_stats['failed_downloads']} 张")
    print()
    
    print("角色分布统计:")
    for role_name, count in download_stats["role_distribution"].items():
        chinese_name = ROLES[role_name]["chinese_name"]
        print(f"  {chinese_name}: {count} 张")
    
    print()
    print("最终数据统计:")
    total_images = 0
    for role_name, role_config in ROLES.items():
        role_dir = os.path.join(DATA_DIR, role_config["dir"])
        final_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {role_config['chinese_name']}: {final_count} 张")
        total_images += final_count
    
    print(f"总图片数量: {total_images} 张")
    print("=" * 60)
    
    # 下载统计信息
    stats_file = "./data/download_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(download_stats, f, ensure_ascii=False, indent=2)
    print(f"\n下载统计已保存到: {stats_file}")

if __name__ == "__main__":
    main()
