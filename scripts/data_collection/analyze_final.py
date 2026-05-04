#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计名单角色的图片 - 使用文件夹实际的拼音"""
import os
from pathlib import Path

ROLE_FILE = Path('auto_spider_img/loli-role.txt')
IMG_DIR = Path('data/organized_images')

# 读取名单角色
roles = []
with open(ROLE_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(' ')
            roles.append(parts[0])

# 文件夹名和角色的映射 - 基于实际URL文件名
FOLDER_MAPPING = {
    # 阿洛娜
    '阿洛娜': 'a1luo4na4',
    # 阿尼亚
    '阿尼亚': 'a1ni4ya4',
    # 布洛妮娅
    '布洛妮娅': 'bu4luo4ni2ya4',
    # 爱丽儿
    '爱丽儿': 'ai4li4er3',
    # 希儿
    '希儿': 'xi1er3',
    # 祢豆子
    '祢豆子': 'mi2dou4zi',
}

# 统计
stats = []
for role in roles:
    # 优先用映射
    folder = FOLDER_MAPPING.get(role)
    if not folder:
        # 找所有包含该角色相关名称的文件夹
        matching_folders = []
        for f in IMG_DIR.glob('*'):
            if f.is_dir():
                name = f.name.lower()
                if role in name:
                    matching_folders.append(f)
        
        if matching_folders:
            # 取图片最多的那个文件夹
            best = max(matching_folders, key=lambda x: len(list(x.glob('*'))))
            folder = best.name
        else:
            # 实在找不到，试试直接拼音
            from pypinyin import lazy_pinyin, Style
            folder = ''.join(lazy_pinyin(role, style=Style.TONE3))
    
    dir_path = IMG_DIR / folder
    if dir_path.exists():
        count = len(list(dir_path.glob('*')))
    else:
        count = 0
    stats.append((role, folder, count))

stats.sort(key=lambda x: x[2], reverse=True)

print('=' * 60)
print('📊 名单角色图片统计 (最终版)')
print('=' * 60)
good = [s for s in stats if s[2] >= 100]
low = [s for s in stats if 0 < s[2] < 100]
missing = [s for s in stats if s[2] == 0]

print(f'✅ 图片充足(>=100): {len(good)} 个')
print(f'⚠️ 图片不足(<100): {len(low)} 个')
print(f'❌ 无图片: {len(missing)} 个')
print('=' * 60)

if missing:
    print('\n❌ 无图片的角色:')
    for role, folder, _ in missing:
        print(f'  {role} (需要文件夹: {folder})')

if low:
    print('\n⚠️ 图片不足的角色:')
    for role, folder, cnt in low:
        print(f'  {role} ({folder}): {cnt} 张')

print('\n✅ 图片充足的角色 (前10):')
for role, folder, cnt in good[:10]:
    print(f'  {role} ({folder}): {cnt} 张')
