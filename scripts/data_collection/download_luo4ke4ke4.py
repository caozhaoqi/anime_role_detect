#!/usr/bin/env python3
import requests
import hashlib
from pathlib import Path

role_name = 'luo4ke4ke4'
img_dir = Path('data/organized_images') / role_name
img_dir.mkdir(parents=True, exist_ok=True)

url_file = Path(f'spider_image_system/data/img_url/{role_name}_img.txt')
with open(url_file) as f:
    urls = [l.strip() for l in f if l.strip()]

print(f'洛可可: {len(urls)} 个URL')

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://www.pixiv.net/'
})

existing_hashes = set()
downloaded = 0

for url in urls:
    if downloaded >= 50:
        break

    try:
        resp = session.get(url, timeout=15)
        content = resp.content
        file_hash = hashlib.md5(content).hexdigest()

        if file_hash in existing_hashes:
            continue

        ext = '.jpg'
        if url.lower().endswith('.png'):
            ext = '.png'

        filepath = img_dir / f'{downloaded}{ext}'
        with open(filepath, 'wb') as f:
            f.write(content)

        existing_hashes.add(file_hash)
        downloaded += 1

        if downloaded % 10 == 0:
            print(f'已下载 {downloaded}/50...')
    except Exception as e:
        continue

print(f'✅ 洛可可下载完成: {downloaded} 张')
print(f'文件夹中实际文件数: {len(list(img_dir.glob("*")))}')
