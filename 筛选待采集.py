#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""筛选待采集角色"""

import os

# 读取已采集的角色
collected = set()
for d in os.listdir('data/danbooru_images'):
    if os.path.isdir(os.path.join('data/danbooru_images', d)):
        collected.add(d.lower())

# 读取角色标签列表
with open('archived/spider_image_system/角色标签列表.txt') as f:
    all_tags = [line.strip() for line in f if line.strip()]

# 筛选未采集的
missing = []
for tag in all_tags:
    if tag.lower() not in collected:
        missing.append(tag)

print(f'总角色数: {len(all_tags)}')
print(f'已采集: {len(collected)}')
print(f'待采集: {len(missing)}')
print()
print('待采集列表:')
for i, tag in enumerate(missing[:50], 1):
    print(f'{i}. {tag}')
if len(missing) > 50:
    print(f'... 还有 {len(missing) - 50} 个')

# 写入待采集列表
with open('archived/spider_image_system/待采集列表.txt', 'w') as f:
    for tag in missing:
        f.write(tag + '\n')
print()
print('已写入 archived/spider_image_system/待采集列表.txt')
