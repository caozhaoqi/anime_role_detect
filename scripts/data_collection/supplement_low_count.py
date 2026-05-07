#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充采集低于50张图片的角色"""
import os
import sys
import time
import hashlib
import requests
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent.parent
URL_DIR = PROJECT_ROOT / "spider_image_system" / "data" / "img_url"
OUTPUT_DIR = PROJECT_ROOT / "data" / "organized_images"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.pixiv.net/'
}

# 目标图片数量
TARGET_COUNT = 50

def get_role_image_count(role_name):
    """获取角色当前图片数量"""
    role_dir = OUTPUT_DIR / role_name
    if not role_dir.exists():
        return 0
    return len(list(role_dir.glob('*')))

def get_existing_hashes(role_name):
    """获取角色已下载图片的哈希集合"""
    hashes = set()
    role_dir = OUTPUT_DIR / role_name
    if role_dir.exists():
        for img in role_dir.glob('*'):
            if img.is_file():
                try:
                    with open(img, 'rb') as f:
                        hashes.add(hashlib.md5(f.read()).hexdigest())
                except:
                    pass
    return hashes

def download_image(url, role_name, session, existing_hashes):
    """下载单张图片"""
    try:
        response = session.get(url, timeout=30, headers=HEADERS, allow_redirects=True)
        if response.status_code != 200:
            return 'failed', None
        
        content = response.content
        if len(content) < 1000:
            return 'failed', None
        
        img_hash = hashlib.md5(content).hexdigest()
        if img_hash in existing_hashes:
            return 'skipped', None
        
        ext = Path(urlparse(url).path).suffix or '.jpg'
        if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
            ext = '.jpg'
        
        role_dir = OUTPUT_DIR / role_name
        role_dir.mkdir(exist_ok=True)
        
        filepath = role_dir / f"{img_hash}{ext}"
        with open(filepath, 'wb') as f:
            f.write(content)
        
        existing_hashes.add(img_hash)
        return 'success', img_hash
    except Exception as e:
        return 'error', None

def download_role_images(role_name, urls, target_count, session):
    """下载角色图片直到达到目标数量"""
    existing_hashes = get_existing_hashes(role_name)
    current_count = len(existing_hashes)
    
    if current_count >= target_count:
        print(f"   ⏭️ {role_name}: 已有 {current_count} 张，无需补充")
        return 0, 0, 0
    
    needed = target_count - current_count
    print(f"   📥 {role_name}: 需要补充 {needed} 张")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for url in urls:
        if downloaded >= needed:
            break
        
        result, _ = download_image(url, role_name, session, existing_hashes)
        
        if result == 'success':
            downloaded += 1
        elif result == 'skipped':
            skipped += 1
        else:
            failed += 1
        
        if downloaded % 10 == 0 and downloaded > 0:
            print(f"      已下载 {downloaded}/{needed}...")
    
    return downloaded, skipped, failed

def main():
    print("=" * 60)
    print("🚀 开始补充采集低于50张图片的角色")
    print("=" * 60)
    
    # 1. 找出图片数量低于50张的角色
    roles_to_supplement = []
    for folder in OUTPUT_DIR.iterdir():
        if not folder.is_dir() or folder.name in ['其他', 'trash', 'trash_nsfw', 'trash_multi_face']:
            continue
        
        count = get_role_image_count(folder.name)
        if count < TARGET_COUNT:
            roles_to_supplement.append((folder.name, count))
    
    print(f"\n📊 发现 {len(roles_to_supplement)} 个角色图片数低于 {TARGET_COUNT} 张:")
    for role, count in sorted(roles_to_supplement, key=lambda x: x[1]):
        print(f"   {role}: {count} 张")
    
    # 2. 检查URL文件并下载
    session = requests.Session()
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    
    for role, current_count in sorted(roles_to_supplement, key=lambda x: x[1]):
        url_file = URL_DIR / f"{role}_img.txt"
        
        if not url_file.exists():
            print(f"\n❌ {role}: URL文件不存在")
            continue
        
        with open(url_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        print(f"\n📁 {role}: {len(urls)} 个URL, 当前 {current_count} 张")
        
        downloaded, skipped, failed = download_role_images(role, urls, TARGET_COUNT, session)
        
        new_count = current_count + downloaded
        print(f"   ✅ 新增 {downloaded} 张, 跳过 {skipped}, 失败 {failed}")
        print(f"   📊 总计: {new_count} 张")
        
        total_downloaded += downloaded
        total_skipped += skipped
        total_failed += failed
        
        time.sleep(0.3)
    
    print("\n" + "=" * 60)
    print("✅ 补充采集完成!")
    print("=" * 60)
    print(f"新增下载: {total_downloaded} 张")
    print(f"跳过(已存在): {total_skipped} 张")
    print(f"失败: {total_failed} 张")

if __name__ == '__main__':
    main()