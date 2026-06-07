#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分析脚本
"""

import os
from pathlib import Path

data_dir = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset'

stats = {
    'total_images': 0,
    'total_roles': 0,
    'roles': [],
    'formats': {},
    'total_size_mb': 0,
}

for role_dir in sorted(Path(data_dir).iterdir()):
    if not role_dir.is_dir():
        continue
    
    role_name = role_dir.name
    images = [f for f in role_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    if not images:
        continue
    
    stats['total_roles'] += 1
    stats['total_images'] += len(images)
    
    for img in images:
        ext = img.suffix.lower()
        stats['formats'][ext] = stats['formats'].get(ext, 0) + 1
        try:
            stats['total_size_mb'] += os.path.getsize(img) / (1024 * 1024)
        except:
            pass
    
    stats['roles'].append({
        'name': role_name,
        'count': len(images),
    })

stats['roles'].sort(key=lambda x: x['count'], reverse=True)

print('='*60)
print('数据集分析报告')
print('='*60)
print(f'\n📊 概览')
print(f'  角色总数: {stats["total_roles"]}')
print(f'  图片总数: {stats["total_images"]}')
print(f'  平均每角色: {stats["total_images"] / stats["total_roles"]:.1f} 张')
print(f'  总大小: {stats["total_size_mb"]:.2f} MB')

print(f'\n📁 图片格式分布')
for fmt, count in sorted(stats['formats'].items(), key=lambda x: x[1], reverse=True):
    pct = count / stats['total_images'] * 100
    print(f'  {fmt}: {count} ({pct:.1f}%)')

print(f'\n🏆 图片最多的角色 (Top 10)')
for i, role in enumerate(stats['roles'][:10], 1):
    print(f'  {i}. {role["name"]}: {role["count"]} 张')

print(f'\n⚠️ 图片不足的角色')
不足 = [r for r in stats['roles'] if r['count'] < 5]
for i, role in enumerate(不足[:10], 1):
    print(f'  {i}. {role["name"]}: {role["count"]} 张')

print(f'\n📚 按作品分类')
work_stats = {}
for role in stats['roles']:
    name = role['name']
    if 'genshin_impact' in name:
        work = '原神'
    elif 'blue_archive' in name:
        work = '蔚蓝档案'
    elif 'honkai_star_rail' in name:
        work = '崩坏星穹铁道'
    elif 'honkai_impact' in name:
        work = '崩坏3'
    elif 'arknights' in name:
        work = '明日方舟'
    elif 're_zero' in name:
        work = 'Re:从零开始的异世界'
    elif 'kimetsu_no_yaiba' in name:
        work = '鬼灭之刃'
    elif 'hololive' in name:
        work = 'Hololive'
    else:
        work = '其他'
    
    if work not in work_stats:
        work_stats[work] = {'roles': 0, 'images': 0}
    work_stats[work]['roles'] += 1
    work_stats[work]['images'] += role['count']

for work, data in sorted(work_stats.items(), key=lambda x: x[1]['images'], reverse=True):
    print(f'  {work}: {data["roles"]} 角色, {data["images"]} 图片')

print('='*60)
