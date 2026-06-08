#!/usr/bin/env python3
"""分析 final_dataset 目录中的数据"""

import os
from pathlib import Path

def analyze_dataset(data_dir):
    data_path = Path(data_dir)
    
    # 统计角色目录
    char_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f'📁 角色总数: {len(char_dirs)}')

    # 统计每个角色的图片
    char_stats = []
    for char_dir in char_dirs:
        images = list(char_dir.glob('*'))
        jpg_count = len([f for f in images if f.suffix.lower() in ('.jpg', '.jpeg')])
        webp_count = len([f for f in images if f.suffix.lower() == '.webp'])
        other_count = len(images) - jpg_count - webp_count
        char_stats.append({
            'name': char_dir.name,
            'total': len(images),
            'jpg': jpg_count,
            'webp': webp_count,
            'other': other_count
        })

    # 按图片数量排序
    char_stats.sort(key=lambda x: x['total'], reverse=True)

    print(f'\n📊 各角色图片数量分布（Top 20）:')
    print('-' * 80)
    print(f'{"角色名称":<30} {"总数":>6} {"JPG":>6} {"WebP":>6} {"其他":>6}')
    print('-' * 80)
    for i, stat in enumerate(char_stats[:20], 1):
        print(f'{i:2d}. {stat["name"]:<28} {stat["total"]:>6} {stat["jpg"]:>6} {stat["webp"]:>6} {stat["other"]:>6}')

    # 总计
    total_images = sum(s['total'] for s in char_stats)
    total_jpg = sum(s['jpg'] for s in char_stats)
    total_webp = sum(s['webp'] for s in char_stats)
    total_other = sum(s['other'] for s in char_stats)

    print('-' * 80)
    print(f'{"总计":<30} {total_images:>6} {total_jpg:>6} {total_webp:>6} {total_other:>6}')

    # 统计文件格式占比
    print(f'\n🖼️ 文件格式分布:')
    print(f'  JPG: {total_jpg} ({total_jpg/total_images*100:.1f}%)')
    print(f'  WebP: {total_webp} ({total_webp/total_images*100:.1f}%)')
    print(f'  其他: {total_other} ({total_other/total_images*100:.1f}%)')

    # 统计角色图片数量分布区间
    print(f'\n📈 角色图片数量分布区间:')
    ranges = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, float('inf'))]
    for r_min, r_max in ranges:
        count = sum(1 for s in char_stats if r_min <= s['total'] <= r_max)
        r_max_str = '∞' if r_max == float('inf') else int(r_max)
        print(f'  {r_min}-{r_max_str} 张: {count} 个角色')

    # 检查是否有异常文件
    print(f'\n🔍 检查异常情况:')
    empty_dirs = [d.name for d in char_dirs if len(list(d.glob('*'))) == 0]
    if empty_dirs:
        print(f'  ⚠️ 空目录: {len(empty_dirs)} 个 - {empty_dirs}')
    else:
        print(f'  ✅ 无空目录')

if __name__ == '__main__':
    data_dir = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset'
    analyze_dataset(data_dir)