#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蔚蓝档案角色数据采集系统
支持多角色、大规模、智能分类的数据采集
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
import random
from datetime import datetime

# 配置参数
DATA_DIR = "./data/downloaded_images"
BLUE_ARCHIVE_ROLES_FILE = "./auto_spider_img/blda_spider_img_keyword.txt"
MAX_IMAGES_PER_ROLE = 200  # 每个角色最多采集200张图片
MAX_THREADS = 8  # 最大并发线程数

# 数据源配置
DATA_SOURCES = [
    {
        "name": "现有URL文件",
        "files": [
            "./data/img_url/arona_img.txt",
            "./data/href_url/arona_result_url.txt",
            "./data/href_url/arona_url.txt"
        ]
    },
    {
        "name": "自动爬虫数据",
        "base_url": "https://sd.vv50.de",
        "search_pattern": "https://sd.vv50.de/artworks/{}"
    }
]

def load_blue_archive_roles():
    """加载蔚蓝档案角色列表"""
    roles = {}
    
    if not os.path.exists(BLUE_ARCHIVE_ROLES_FILE):
        print(f"警告: 角色文件不存在: {BLUE_ARCHIVE_ROLES_FILE}")
        return roles
    
    with open(BLUE_ARCHIVE_ROLES_FILE, 'r', encoding='utf-8') as f:
        role_names = [line.strip() for line in f if line.strip()]
    
    # 为每个角色创建目录和配置
    for i, role_name in enumerate(role_names):
        # 生成拼音目录名（简化版）
        pinyin_name = role_name.lower().replace(" ", "_").replace("·", "_")
        role_dir = os.path.join(DATA_DIR, pinyin_name)
        os.makedirs(role_dir, exist_ok=True)
        
        roles[pinyin_name] = {
            "chinese_name": role_name,
            "dir": pinyin_name,
            "keywords": [role_name, pinyin_name],
            "target_count": MAX_IMAGES_PER_ROLE
        }
    
    return roles

def is_valid_image(content):
    """检查是否为有效的图片"""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except:
        return False

def classify_url_to_role(url, roles):
    """根据URL分类到对应角色"""
    url_lower = url.lower()
    
    for role_key, role_info in roles.items():
        for keyword in role_info["keywords"]:
            if keyword.lower() in url_lower:
                return role_key
    
    # 默认分类到阿罗娜
    return "阿罗娜".lower().replace(" ", "_").replace("·", "_")

def download_image(url, save_dir, role_key, timeout=15):
    """下载单张图片"""
    try:
        headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
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
                
                return True, filename
            else:
                return False, "无效图片"
        else:
            return False, f"HTTP {response.status_code}"
            
    except Exception as e:
        return False, str(e)

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
            
            # 过滤高质量图片
            filtered_urls = []
            for url in img_urls:
                if any(size in url.lower() for size in ['master', 'original', 'large']):
                    if not any(exclude in url.lower() for exclude in ['icon', 'logo', 'thumb', 'small', 'avatar']):
                        filtered_urls.append(url)
            
            return filtered_urls[:15]  # 每个页面最多15张图片
            
    except Exception as e:
        print(f"  ✗ 提取图片URL失败: {page_url} - {str(e)}")
    
    return []

def collect_from_url_files(roles):
    """从现有URL文件采集数据"""
    print("\n" + "=" * 60)
    print("从现有URL文件采集数据")
    print("=" * 60)
    
    all_urls = []
    
    # 收集所有URL文件
    url_files = []
    for source in DATA_SOURCES:
        if source["name"] == "现有URL文件":
            url_files.extend(source["files"])
    
    for url_file in url_files:
        if not os.path.exists(url_file):
            continue
            
        print(f"\n处理文件: {url_file}")
        
        with open(url_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
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
        
        all_urls.extend(image_urls)
        
        # 从页面URL中提取图片URL
        if page_urls:
            print(f"  正在从页面提取图片URL...")
            for page_url in page_urls[:30]:  # 限制处理30个页面
                extracted_urls = extract_image_urls_from_page(page_url)
                all_urls.extend(extracted_urls)
                time.sleep(0.3)
    
    # 去重
    all_urls = list(set(all_urls))
    print(f"\n总共提取到 {len(all_urls)} 个唯一URL")
    
    return all_urls

def download_and_classify_images(urls, roles):
    """下载并分类图片"""
    print(f"\n开始下载 {len(urls)} 张图片...")
    
    download_stats = {
        "total": len(urls),
        "successful": 0,
        "failed": 0,
        "invalid": 0,
        "role_distribution": {role_key: 0 for role_key in roles.keys()}
    }
    
    # 使用多线程下载
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []
        
        for i, url in enumerate(urls, start=1):
            # 分类URL到对应角色
            role_key = classify_url_to_role(url, roles)
            save_dir = os.path.join(DATA_DIR, roles[role_key]["dir"])
            
            # 检查是否达到目标数量
            current_count = len([f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if current_count >= roles[role_key]["target_count"]:
                continue
            
            # 提交下载任务
            future = executor.submit(download_image, url, save_dir, role_key)
            futures.append((future, i, len(urls), url, role_key))
        
        # 处理下载结果
        for future, current, total, url, role_key in futures:
            try:
                success, result = future.result(timeout=30)
                
                if success:
                    print(f"✓ [{current}/{total}] {roles[role_key]['chinese_name']}: {result}")
                    download_stats["successful"] += 1
                    download_stats["role_distribution"][role_key] += 1
                else:
                    if "无效图片" in result:
                        download_stats["invalid"] += 1
                    download_stats["failed"] += 1
                    print(f"✗ [{current}/{total}] {url} - {result}")
                
                # 避免请求过快
                time.sleep(0.1)
                
            except Exception as e:
                download_stats["failed"] += 1
                print(f"✗ [{current}/{total}] {url} - {str(e)}")
    
    return download_stats

def main():
    """主函数"""
    print("=" * 60)
    print("蔚蓝档案角色数据采集系统")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载角色列表
    print("\n加载蔚蓝档案角色列表...")
    roles = load_blue_archive_roles()
    print(f"加载了 {len(roles)} 个角色")
    
    # 统计现有数据
    print("\n现有数据统计:")
    for role_key, role_info in roles.items():
        role_dir = os.path.join(DATA_DIR, role_info["dir"])
        existing_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {role_info['chinese_name']}: {existing_count} 张")
    
    # 从现有URL文件采集数据
    urls = collect_from_url_files(roles)
    
    if urls:
        # 下载并分类图片
        download_stats = download_and_classify_images(urls, roles)
        
        # 统计结果
        print("\n" + "=" * 60)
        print("数据采集完成")
        print("=" * 60)
        print(f"处理URL总数: {download_stats['total']}")
        print(f"成功下载: {download_stats['successful']} 张")
        print(f"无效图片: {download_stats['invalid']} 张")
        print(f"下载失败: {download_stats['failed']} 张")
        
        print("\n角色分布统计:")
        for role_key, count in download_stats["role_distribution"].items():
            if count > 0:
                print(f"  {roles[role_key]['chinese_name']}: +{count} 张")
        
        print("\n最终数据统计:")
        total_images = 0
        for role_key, role_info in roles.items():
            role_dir = os.path.join(DATA_DIR, role_info["dir"])
            final_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {role_info['chinese_name']}: {final_count} 张")
            total_images += final_count
        
        print(f"\n总图片数量: {total_images} 张")
        
        # 保存统计信息
        stats_file = "./data/blue_archive_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "download_stats": download_stats,
                "final_distribution": {role_key: {"chinese_name": role_info["chinese_name"], "count": len([f for f in os.listdir(os.path.join(DATA_DIR, role_info["dir"])) if f.endswith(('.jpg', '.jpeg', '.png'))])} for role_key, role_info in roles.items()}
            }, f, ensure_ascii=False, indent=2)
        print(f"\n统计信息已保存到: {stats_file}")
    else:
        print("\n没有找到可下载的URL")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
