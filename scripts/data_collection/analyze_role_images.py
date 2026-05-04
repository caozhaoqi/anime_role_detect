#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')
from pypinyin import lazy_pinyin, Style
import os

ROLE_FILE = 'auto_spider_img/loli-role.txt'
IMG_DIR = 'data/organized_images'

# 读取名单角色
roles = []
with open(ROLE_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(' ')
            roles.append(parts[0])

# 检查每个角色的图片数量
stats = []
for role in roles:
    pinyin = ''.join(lazy_pinyin(role, style=Style.TONE3))
    dir_path = os.path.join(IMG_DIR, pinyin)
    if os.path.exists(dir_path):
        cnt = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        stats.append((role, cnt))
    else:
        stats.append((role, 0))

stats.sort(key=lambda x: x[1], reverse=True)

print('=' * 60)
print('📊 名单角色图片统计')
print('=' * 60)
good = [s for s in stats if s[1] >= 100]
low = [s for s in stats if 0 < s[1] < 100]
missing = [s for s in stats if s[1] == 0]

print(f'✅ 图片充足(>=100): {len(good)} 个')
print(f'⚠️ 图片不足(<100): {len(low)} 个')
print(f'❌ 无图片: {len(missing)} 个')
print('=' * 60)

if missing:
    print('\\n❌ 无图片的角色:')
    for role, cnt in missing:
        print(f'  {role}')

if low:
    print('\\n⚠️ 图片不足的角色:')
    for role, cnt in low:
        print(f'  {role}: {cnt}')
