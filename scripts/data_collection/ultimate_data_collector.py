#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极数据采集系统
支持多游戏、多角色、大规模、智能分类的数据采集
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
MAX_IMAGES_PER_ROLE = 300  # 每个角色最多采集300张图片
MAX_THREADS = 10  # 最大并发线程数
REQUEST_TIMEOUT = 20  # 请求超时时间

# 游戏角色配置
GAME_ROLES = {
    "蔚蓝档案": "./auto_spider_img/blda_spider_img_keyword.txt",
    "原神": "./auto_spider_img/0307_spider_img_keyword.txt",
    "崩坏星穹铁道": "./auto_spider_img/1_p_top_spider_img_keyword.txt",
    "崩坏三": "./auto_spider_img/6_honkai3_chinese_spider_img_keyword.txt",
    "绝区零": "./auto_spider_img/zzz_spider_img_keyword.txt"
}

# 现有数据源
EXISTING_DATA_SOURCES = [
    "./data/img_url/arona_img.txt",
    "./data/href_url/arona_result_url.txt",
    "./data/href_url/arona_url.txt"
]

# 下载统计
download_stats = {
    "total_processed": 0,
    "successful_downloads": 0,
    "failed_downloads": 0,
    "invalid_images": 0,
    "duplicate_files": 0,
    "game_distribution": {},
    "role_distribution": {},
    "start_time": None,
    "end_time": None
}

def load_game_roles():
    """加载所有游戏的角色列表"""
    all_roles = {}
    
    for game_name, role_file in GAME_ROLES.items():
        if not os.path.exists(role_file):
            print(f"警告: {game_name} 角色文件不存在: {role_file}")
            continue
        
        with open(role_file, 'r', encoding='utf-8') as f:
            role_names = [line.strip() for line in f if line.strip()]
        
        # 为每个角色创建配置
        for role_name in role_names:
            # 生成拼音目录名
            pinyin_name = role_name.lower().replace(" ", "_").replace("·", "_").replace("'", "")
            role_dir = os.path.join(DATA_DIR, pinyin_name)
            os.makedirs(role_dir, exist_ok=True)
            
            all_roles[pinyin_name] = {
                "chinese_name": role_name,
                "game": game_name,
                "dir": pinyin_name,
                "keywords": [role_name, pinyin_name],
                "target_count": MAX_IMAGES_PER_ROLE
            }
    
    return all_roles

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
    
    # 如果没有匹配到，返回None
    return None

def download_image(url, save_dir, role_key, timeout=REQUEST_TIMEOUT):
    """下载单张图片"""
    try:
        # 随机User-Agent
        headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15'
            ]),
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.google.com/',
            'Accept-Encoding': 'gzip, deflate, br'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            # 检查内容类型
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                return False, "非图片内容"
            
            # 检查是否为有效图片
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
            
    except requests.exceptions.Timeout:
        return False, "请求超时"
    except requests.exceptions.ConnectionError:
        return False, "连接错误"
    except Exception as e:
        return False, str(e)

def extract_image_urls_from_page(page_url):
    """从页面URL中提取图片URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(page_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            # 使用正则表达式提取图片URL
            img_pattern = r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|gif|webp)'
            img_urls = re.findall(img_pattern, response.text)
            
            # 过滤高质量图片
            filtered_urls = []
            for url in img_urls:
                if any(size in url.lower() for size in ['master', 'original', 'large', 'high']):
                    if not any(exclude in url.lower() for exclude in ['icon', 'logo', 'thumb', 'small', 'avatar', 'profile']):
                        filtered_urls.append(url)
            
            return filtered_urls[:20]  # 每个页面最多20张图片
            
    except Exception as e:
        pass
    
    return []

def collect_from_existing_sources(roles):
    """从现有数据源采集"""
    print("\n" + "=" * 60)
    print("从现有数据源采集")
    print("=" * 60)
    
    all_urls = []
    
    for source_file in EXISTING_DATA_SOURCES:
        if not os.path.exists(source_file):
            continue
            
        print(f"\n处理文件: {source_file}")
        
        with open(source_file, 'r', encoding='utf-8') as f:
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
            for page_url in page_urls[:50]:  # 限制处理50个页面
                extracted_urls = extract_image_urls_from_page(page_url)
                all_urls.extend(extracted_urls)
                time.sleep(0.2)
    
    # 去重
    all_urls = list(set(all_urls))
    print(f"\n总共提取到 {len(all_urls)} 个唯一URL")
    
    return all_urls

def process_downloads(urls, roles):
    """处理下载任务"""
    print(f"\n开始处理 {len(urls)} 个URL...")
    
    processed_count = 0
    
    # 使用多线程下载
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []
        
        for i, url in enumerate(urls, start=1):
            # 分类URL到对应角色
            role_key = classify_url_to_role(url, roles)
            
            # 如果无法分类，跳过
            if not role_key:
                continue
            
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
                processed_count += 1
                download_stats["total_processed"] += 1
                
                if success:
                    print(f"✓ [{processed_count}/{len(futures)}] {roles[role_key]['chinese_name']}: {result}")
                    download_stats["successful_downloads"] += 1
                    
                    # 更新统计
                    game_name = roles[role_key]["game"]
                    if game_name not in download_stats["game_distribution"]:
                        download_stats["game_distribution"][game_name] = 0
                    download_stats["game_distribution"][game_name] += 1
                    
                    if role_key not in download_stats["role_distribution"]:
                        download_stats["role_distribution"][role_key] = 0
                    download_stats["role_distribution"][role_key] += 1
                else:
                    if "文件已存在" in result:
                        download_stats["duplicate_files"] += 1
                    elif "无效图片" in result:
                        download_stats["invalid_images"] += 1
                    else:
                        download_stats["failed_downloads"] += 1
                    
                    # 只显示重要错误
                    if not any(keyword in result for keyword in ["文件已存在", "非图片内容"]):
                        print(f"✗ [{processed_count}/{len(futures)}] {url[:50]}... - {result}")
                
                # 控制请求速度
                time.sleep(0.05)
                
            except Exception as e:
                download_stats["failed_downloads"] += 1
                print(f"✗ [{processed_count}/{len(futures)}] {url[:50]}... - {str(e)}")

def print_statistics(roles):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("数据采集统计")
    print("=" * 60)
    print(f"处理URL总数: {download_stats['total_processed']}")
    print(f"成功下载: {download_stats['successful_downloads']} 张")
    print(f"重复文件: {download_stats['duplicate_files']} 张")
    print(f"无效图片: {download_stats['invalid_images']} 张")
    print(f"下载失败: {download_stats['failed_downloads']} 张")
    
    print("\n游戏分布:")
    for game_name, count in download_stats["game_distribution"].items():
        print(f"  {game_name}: {count} 张")
    
    print("\n角色分布 (前10个):")
    sorted_roles = sorted(download_stats["role_distribution"].items(), key=lambda x: x[1], reverse=True)
    for role_key, count in sorted_roles[:10]:
        if count > 0:
            print(f"  {roles[role_key]['chinese_name']}: {count} 张")
    
    print("\n最终数据统计:")
    total_images = 0
    role_count = 0
    for role_key, role_info in roles.items():
        role_dir = os.path.join(DATA_DIR, role_info["dir"])
        final_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        if final_count > 0:
            role_count += 1
            total_images += final_count
    
    print(f"  有数据的角色: {role_count} 个")
    print(f"  总图片数量: {total_images} 张")
    print("=" * 60)

def main():
    """主函数"""
    global download_stats
    
    download_stats["start_time"] = datetime.now()
    
    print("=" * 60)
    print("终极数据采集系统")
    print("=" * 60)
    print(f"开始时间: {download_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载所有游戏角色
    print("\n加载游戏角色列表...")
    roles = load_game_roles()
    print(f"加载了 {len(roles)} 个角色")
    
    # 统计现有数据
    print("\n现有数据统计:")
    existing_total = 0
    for role_key, role_info in roles.items():
        role_dir = os.path.join(DATA_DIR, role_info["dir"])
        existing_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        if existing_count > 0:
            print(f"  {role_info['chinese_name']} ({role_info['game']}): {existing_count} 张")
            existing_total += existing_count
    
    print(f"\n现有总图片数: {existing_total} 张")
    
    # 从现有数据源采集
    urls = collect_from_existing_sources(roles)
    
    if urls:
        # 处理下载
        process_downloads(urls, roles)
        
        # 打印统计
        print_statistics(roles)
        
        # 保存统计信息
        download_stats["end_time"] = datetime.now()
        stats_file = "./data/ultimate_collection_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(download_stats, f, ensure_ascii=False, indent=2)
        print(f"\n统计信息已保存到: {stats_file}")
    else:
        print("\n没有找到可下载的URL")
    
    print(f"\n结束时间: {download_stats['end_time'].strftime('%Y-%m-%d %H:%M:%S') if download_stats['end_time'] else 'N/A'}")
    duration = (download_stats['end_time'] - download_stats['start_time']).total_seconds() if download_stats['end_time'] else 0
    print(f"总耗时: {duration:.2f} 秒")
    print("=" * 60)

if __name__ == "__main__":
    main()
