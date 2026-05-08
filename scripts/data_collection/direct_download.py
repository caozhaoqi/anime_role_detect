#!/usr/bin/env python3
"""直接下载图片不足20张的角色图片"""
import os
import sys
import requests
import hashlib
import time
from pathlib import Path

sys.path.insert(0, 'spider_image_system/src/run')
from constants import PINYIN_MAPPING

# 需要下载的角色
LOW_COUNT_ROLES = [
    '芙丽希娅',
    '洛茜', 
    '克萝萝',
    '德丽莎'
]

# 请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.pixiv.net/',
    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

def download_image(url, save_path, max_retries=3):
    """下载单张图片"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            else:
                print(f"   ⚠️ 尝试 {attempt+1}/{max_retries}: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ⚠️ 尝试 {attempt+1}/{max_retries}: {e}")
        time.sleep(1)
    return False

def download_role_images(role_name):
    """下载单个角色的图片"""
    pinyin = PINYIN_MAPPING.get(role_name)
    if not pinyin:
        print(f"❌ 未找到 {role_name} 的拼音映射")
        return 0
    
    # 读取img_url文件
    img_url_file = f'spider_image_system/data/img_url/{pinyin}_img.txt'
    if not os.path.exists(img_url_file):
        print(f"❌ {role_name} 的图片URL文件不存在")
        return 0
    
    with open(img_url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    urls = list(set(urls))  # 去重
    print(f"📥 {role_name}: 找到 {len(urls)} 个图片URL")
    
    # 目标目录
    target_dir = Path(f'data/organized_images/{pinyin}')
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取已有图片数量
    existing_images = len(list(target_dir.glob('*.jpg'))) + len(list(target_dir.glob('*.png'))) + len(list(target_dir.glob('*.webp')))
    print(f"   已有图片: {existing_images}张")
    
    # 下载图片
    downloaded = 0
    failed = 0
    
    for i, url in enumerate(urls, 1):
        # 生成文件名
        url_hash = hashlib.md5(url.encode()).hexdigest()
        ext = '.jpg' if '.jpg' in url or 'jpg' in url else '.png'
        save_path = target_dir / f"{url_hash}{ext}"
        
        # 如果文件已存在，跳过
        if save_path.exists():
            continue
        
        print(f"   [{i}/{len(urls)}] 下载中...", end='\r')
        
        if download_image(url, save_path):
            downloaded += 1
            if downloaded >= 50 - existing_images:  # 下载到50张为止
                break
        else:
            failed += 1
        
        time.sleep(0.5)  # 间隔0.5秒
    
    # 最终统计
    final_count = len(list(target_dir.glob('*.jpg'))) + len(list(target_dir.glob('*.png'))) + len(list(target_dir.glob('*.webp')))
    print(f"   ✅ 成功下载: {downloaded}张, 失败: {failed}张, 总计: {final_count}张")
    
    return downloaded

def main():
    print("=" * 60)
    print("📷 直接下载图片")
    print("=" * 60)
    
    total_downloaded = 0
    for role in LOW_COUNT_ROLES:
        print(f"\n📋 {role}")
        downloaded = download_role_images(role)
        total_downloaded += downloaded
    
    print(f"\n📊 共下载 {total_downloaded} 张图片")
    print("=" * 60)

if __name__ == '__main__':
    main()
