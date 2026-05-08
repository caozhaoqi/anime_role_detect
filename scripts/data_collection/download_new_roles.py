#!/usr/bin/env python3
import os
import hashlib
import requests
from pathlib import Path

IMG_DIR = Path('data/organized_images')
URL_DIR = Path('spider_image_system/data/img_url')
TARGET_COUNT = 50

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_existing_hashes(role_name):
    role_dir = IMG_DIR / role_name
    hashes = set()
    if role_dir.exists():
        for img_path in role_dir.glob('*'):
            if img_path.is_file():
                try:
                    hashes.add(calculate_md5(img_path))
                except:
                    pass
    return hashes

def download_image(url, role_name, session, existing_hashes):
    role_dir = IMG_DIR / role_name
    role_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        
        content = response.content
        file_hash = hashlib.md5(content).hexdigest()
        
        if file_hash in existing_hashes:
            return 'skipped', None
        
        ext = '.jpg'
        if url.lower().endswith('.png'):
            ext = '.png'
        elif url.lower().endswith('.webp'):
            ext = '.webp'
        
        filename = f"{len([f for f in role_dir.glob('*')])}{ext}"
        filepath = role_dir / filename
        
        with open(filepath, 'wb') as f:
            f.write(content)
        
        existing_hashes.add(file_hash)
        return 'success', filepath
    
    except Exception as e:
        return 'failed', str(e)

def download_role_images(role_name, urls, target_count, session):
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
    print("🚀 为有URL的新角色下载图片")
    print("=" * 60)
    
    roles_to_download = [
        'lu4mu4yuan2',   # 鹿目圆
        'ni2dou4zi5',    # 祢豆子
        'qi2ta3',        # 奇塔
        'shen1yue4',     # 神乐
        'xue4xiao3ban3', # 血小板
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    total_downloaded = 0
    
    for role_name in roles_to_download:
        url_file = URL_DIR / f'{role_name}_img.txt'
        
        if not url_file.exists():
            print(f"\n❌ {role_name}: 未找到URL文件")
            continue
        
        with open(url_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        current_count = len(list((IMG_DIR / role_name).glob('*')))
        print(f"\n📁 {role_name}: {len(urls)} 个URL, 当前 {current_count} 张")
        
        downloaded, skipped, failed = download_role_images(role_name, urls, TARGET_COUNT, session)
        total_downloaded += downloaded
        
        new_count = current_count + downloaded
        print(f"   ✅ 新增 {downloaded} 张, 跳过 {skipped}, 失败 {failed}")
        print(f"   📊 总计: {new_count} 张")
    
    print("\n" + "=" * 60)
    print(f"🎉 下载完成！共新增 {total_downloaded} 张图片")
    print("=" * 60)

if __name__ == '__main__':
    main()
