#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析下载进度 - URL vs 已下载图片"""
import os
from pathlib import Path

URL_DIR = Path('spider_image_system/data/img_url')
IMG_DIR = Path('data/organized_images')

# 获取所有URL文件
url_stats = []
for f in URL_DIR.glob('*.txt'):
    with open(f, 'r', encoding='utf-8') as file:
        count = sum(1 for line in file if line.strip())
    name = f.stem.replace('_img', '')
    url_stats.append((name, count))

# 获取所有图片文件夹
img_stats = {}
for d in IMG_DIR.iterdir():
    if d.is_dir():
        img_stats[d.name] = len(list(d.glob('*')))

# 映射URL文件名到文件夹名
MAPPING = {
    'a1luo4na4': 'a1luo4na4',
    'bu4luo4ni2ya4': 'bu4luo4ni2ya4',
    'ke3li4': 'ke3li4',
    'mi2dou4zi': 'mi2dou4zi',
    'shen2le4': 'shen2le4',
    # 可以继续添加映射
}

# 合并统计
print('=' * 70)
print('📊 下载进度分析')
print('=' * 70)
print(f'{"角色":<25} {"URL数":>8} {"已下载":>8} {"进度":>10}')
print('-' * 70)

total_urls = 0
total_imgs = 0

for name, url_count in sorted(url_stats, key=lambda x: x[1], reverse=True)[:30]:
    folder = MAPPING.get(name, name)
    img_count = img_stats.get(folder, 0)
    progress = f"{img_count/url_count*100:.0f}%" if url_count > 0 else "N/A"
    total_urls += url_count
    total_imgs += img_count
    print(f'{name:<25} {url_count:>8} {img_count:>8} {progress:>10}')

print('-' * 70)
print(f'总计 (前30): {total_urls} URLs, {total_imgs} 图片')

# 检查缺失的角色
print('\n=== 检查URL文件缺失的角色 ===')
all_url_names = set(f.stem.replace('_img', '') for f in URL_DIR.glob('*.txt'))
print(f'URL文件数: {len(all_url_names)}')
