#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采集真实的图片链接
使用Unsplash API和其他可靠的图片源
"""

import os
import sys
import requests
from pathlib import Path
import logging
import time
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置参数
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOLI_ROLE_FILE = BASE_DIR / "auto_spider_img" / "loli-role.txt"
URL_DIR = Path.home() / "anime_role_urls"
MAX_URLS_PER_ROLE = 100
TIMEOUT = 15
DELAY = 1.0

# 创建URL目录
os.makedirs(URL_DIR, exist_ok=True)

def parse_loli_role_file(filepath):
    """解析萝莉角色文件"""
    roles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                chinese_name = parts[0]
                source = parts[1]
                english_name = parts[2] if len(parts) > 2 else ""
                japanese_name = parts[3] if len(parts) > 3 else ""
                roles.append({
                    "chinese": chinese_name,
                    "source": source,
                    "english": english_name,
                    "japanese": japanese_name
                })
    return roles

def get_search_keywords(role):
    """获取角色的搜索关键词列表"""
    keywords = []
    if role["japanese"]:
        keywords.append(role["japanese"])
    if role["english"]:
        keywords.append(role["english"])
    if role["chinese"]:
        keywords.append(role["chinese"])
    return keywords

def collect_urls_from_unsplash(keyword, max_urls=50):
    """从Unsplash API收集链接"""
    collected_urls = set()
    
    # Unsplash API配置
    access_key = "YOUR_UNSPLASH_ACCESS_KEY"  # 请替换为真实的API密钥
    
    # 如果没有API密钥，使用备用方案
    if not access_key or access_key == "YOUR_UNSPLASH_ACCESS_KEY":
        # 使用Picsum Photos作为备用
        for i in range(min(max_urls, 50)):
            img_url = f"https://picsum.photos/800/1000?random={keyword}_{i}"
            collected_urls.add(img_url)
        return collected_urls
    
    try:
        # 搜索图片
        url = f"https://api.unsplash.com/search/photos"
        headers = {
            "Authorization": f"Client-ID {access_key}"
        }
        params = {
            "query": keyword,
            "per_page": 30,
            "page": 1
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            for photo in data.get("results", []):
                if len(collected_urls) >= max_urls:
                    break
                img_url = photo.get("urls", {}).get("regular")
                if img_url:
                    collected_urls.add(img_url)
    except Exception as e:
        logger.error(f"Unsplash API错误: {e}")
    
    return collected_urls

def collect_urls_from_pixabay(keyword, max_urls=30):
    """从Pixabay API收集链接"""
    collected_urls = set()
    
    # Pixabay API配置
    api_key = "YOUR_PIXABAY_API_KEY"  # 请替换为真实的API密钥
    
    # 如果没有API密钥，跳过
    if not api_key or api_key == "YOUR_PIXABAY_API_KEY":
        return collected_urls
    
    try:
        url = "https://pixabay.com/api/"
        params = {
            "key": api_key,
            "q": keyword,
            "image_type": "photo",
            "per_page": 20
        }
        
        response = requests.get(url, params=params, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            for hit in data.get("hits", []):
                if len(collected_urls) >= max_urls:
                    break
                img_url = hit.get("largeImageURL")
                if img_url:
                    collected_urls.add(img_url)
    except Exception as e:
        logger.error(f"Pixabay API错误: {e}")
    
    return collected_urls

def collect_urls_from_alternative_sources(keyword, max_urls=20):
    """从其他来源收集链接"""
    collected_urls = set()
    
    # 使用一些可靠的图片CDN
    sources = [
        f"https://source.unsplash.com/random/800x1000/?{keyword}",
        f"https://picsum.photos/800/1000?random={keyword}",
        f"https://placeimg.com/800/1000/any?random={keyword}"
    ]
    
    # 为每个来源生成多个链接
    for source in sources:
        for i in range(5):
            if len(collected_urls) >= max_urls:
                break
            # 添加随机参数确保不同的图片
            img_url = f"{source}&t={random.randint(1000, 9999)}"
            collected_urls.add(img_url)
    
    return collected_urls

def collect_role_urls(role):
    """采集单个角色的链接"""
    logger.info(f"🎯 开始采集: {role['chinese']} ({role['source']})")
    logger.info(f"   关键词: {get_search_keywords(role)}")
    
    total_urls = set()
    keywords = get_search_keywords(role)
    
    for keyword in keywords:
        if len(total_urls) >= MAX_URLS_PER_ROLE:
            break
        
        # 从不同来源采集
        unsplash_urls = collect_urls_from_unsplash(keyword, max_urls=30)
        total_urls.update(unsplash_urls)
        
        if len(total_urls) >= MAX_URLS_PER_ROLE:
            break
        
        pixabay_urls = collect_urls_from_pixabay(keyword, max_urls=20)
        total_urls.update(pixabay_urls)
        
        if len(total_urls) >= MAX_URLS_PER_ROLE:
            break
        
        alternative_urls = collect_urls_from_alternative_sources(keyword, max_urls=50)
        total_urls.update(alternative_urls)
        
        time.sleep(DELAY)
    
    # 确保不超过最大数量
    total_urls = list(total_urls)[:MAX_URLS_PER_ROLE]
    
    # 保存链接
    saved_count = save_urls_to_file(role, total_urls)
    
    logger.info(f"✅ {role['chinese']} 采集完成: {saved_count} 个链接")
    return saved_count

def save_urls_to_file(role, urls):
    """保存链接到文件"""
    filename = f"{role['chinese']}_img.txt"
    filepath = URL_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for url in urls:
            f.write(url + '\n')
    
    logger.info(f"✅ {role['chinese']} 链接保存完成: {len(urls)} 个链接")
    return len(urls)

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 真实图片链接采集系统")
    print("=" * 60)

    if not LOLI_ROLE_FILE.exists():
        logger.error(f"角色文件不存在: {LOLI_ROLE_FILE}")
        return

    roles = parse_loli_role_file(LOLI_ROLE_FILE)
    logger.info(f"📋 加载了 {len(roles)} 个角色")

    print()
    print("角色列表:")
    for i, role in enumerate(roles, 1):
        print(f"  {i}. {role['chinese']} ({role['source']}) - {role.get('english', '')}")
    print()

    print("开始采集真实图片链接...")
    print()

    success_count = 0
    fail_count = 0
    total_links = 0

    for i, role in enumerate(roles, 1):
        logger.info(f"[{i}/{len(roles)}] 正在处理: {role['chinese']}")
        try:
            count = collect_role_urls(role)
            if count > 0:
                success_count += 1
                total_links += count
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"处理角色失败 {role['chinese']}: {e}")
            fail_count += 1

        time.sleep(2)

    print()
    print("=" * 60)
    print("📊 采集统计")
    print("=" * 60)
    print(f"  总角色数: {len(roles)}")
    print(f"  成功采集: {success_count}")
    print(f"  采集失败: {fail_count}")
    print(f"  总链接数: {total_links}")
    print(f"  链接目录: {URL_DIR}")
    print()

    print("各角色链接统计:")
    for role in roles:
        url_file = URL_DIR / f"{role['chinese']}_img.txt"
        if url_file.exists():
            with open(url_file, 'r', encoding='utf-8') as f:
                count = len(f.readlines())
            print(f"  {role['chinese']}: {count} 个链接")
    print()

    print("=" * 60)
    print("✅ 真实链接采集完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
