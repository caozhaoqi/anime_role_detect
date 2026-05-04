#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充下载 - 针对URL充足但图片不足的角色"""
import os
import sys
import time
import hashlib
import requests
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
URL_DIR = PROJECT_ROOT / "spider_image_system" / "data" / "img_url"
OUTPUT_DIR = PROJECT_ROOT / "data" / "organized_images"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Referer': 'https://www.pixiv.net/'
}

# 需要补充的角色 (URL充足但下载不足)
SUPPLEMENT_ROLES = {
    'a1luo4na4': 87,      # 阿洛娜 - 87/457 = 19%
    'na4xi1da2': 56,      # 纳西妲 - 56/427 = 13%
    'xiao3mei3yan4': 155, # 晓美焰 - 155/519 = 30%
    'an1ke3': 154,        # 安可 - 154/515 = 30%
    'luo4ke3ke3': 190,    # 洛可可 - 190/555 = 34%
    'fu2xuan2': 155,      # 符玄 - 155/384 = 40%
    'lei3bei4': 217,      # 蕾贝 - 217/~400 = 54%
    'ke1xie4ni2ya4': 74,  # 科谢尼娅
    'zao3wu4': 68,        # 早雾
    'yue4qian1ye4': 47,   # 月千夜
    'ai4li4er3': 41,      # 爱丽儿
    'xiao3shan3': 29,     # 小闪
    'ke4luo2luo2': 20,    # 克萝萝
    'fu2li4xi1ya4': 11,   # 芙丽希娅
    'shen2le4': 3,        # 神乐
}

def get_existing_hashes(folder):
    """获取已下载图片的哈希集合"""
    hashes = set()
    folder_path = OUTPUT_DIR / folder
    if folder_path.exists():
        for img in folder_path.glob('*'):
            if img.is_file():
                try:
                    with open(img, 'rb') as f:
                        hashes.add(hashlib.md5(f.read()).hexdigest())
                except:
                    pass
    return hashes

def download_image(url, role_name, session):
    """下载单张图片"""
    try:
        response = session.get(url, timeout=30, headers=HEADERS)
        if response.status_code == 200:
            content = response.content
            img_hash = hashlib.md5(content).hexdigest()

            ext = Path(urlparse(url).path).suffix or '.jpg'
            if len(ext) > 5:
                ext = '.jpg'

            role_dir = OUTPUT_DIR / role_name
            role_dir.mkdir(exist_ok=True)

            filepath = role_dir / f"{img_hash}{ext}"
            if filepath.exists():
                return 'skipped', img_hash

            with open(filepath, 'wb') as f:
                f.write(content)
            return 'success', img_hash
        else:
            return 'failed', None
    except Exception as e:
        return 'error', None

def main():
    print("=" * 60)
    print("🚀 开始补充下载")
    print("=" * 60)

    session = requests.Session()
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for role, current_count in sorted(SUPPLEMENT_ROLES.items(), key=lambda x: x[1]):
        url_file = URL_DIR / f"{role}_img.txt"
        if not url_file.exists():
            print(f"❌ {role}: URL文件不存在")
            continue

        with open(url_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]

        existing_hashes = get_existing_hashes(role)
        print(f"\n📥 {role}: {len(urls)} URLs, 已有 {current_count} 张, 哈希去重 {len(existing_hashes)} 个")

        downloaded = 0
        skipped = 0
        failed = 0

        for url in urls:
            if 'pixiv' in url.lower() or 'img' in url.lower():
                result, img_hash = download_image(url, role, session)

                if result == 'success':
                    existing_hashes.add(img_hash)
                    downloaded += 1
                elif result == 'skipped':
                    skipped += 1
                else:
                    failed += 1

                if downloaded % 20 == 0 and downloaded > 0:
                    print(f"   下载进度: {downloaded} 张...")

        new_total = current_count + downloaded
        print(f"   ✅ {role}: 新增 {downloaded} 张, 跳过 {skipped}, 失败 {failed}")
        print(f"   📊 总计: 约 {new_total} 张")

        total_downloaded += downloaded
        total_skipped += skipped
        total_failed += failed

        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("📊 补充下载完成")
    print("=" * 60)
    print(f"新增下载: {total_downloaded} 张")
    print(f"跳过(已存在): {total_skipped} 张")
    print(f"失败: {total_failed} 张")

if __name__ == '__main__':
    main()
