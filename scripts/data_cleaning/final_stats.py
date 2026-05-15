#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出数据集最终统计信息
"""
import os

def main():
    DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
    
    # 读取角色映射
    role_map = {}
    with open('/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                role_map[parts[2]] = parts[0]
    
    # 统计各角色图片数
    stats = []
    for d in sorted(os.listdir(DATASET_PATH)):
        dp = os.path.join(DATASET_PATH, d)
        if not os.path.isdir(dp) or d.startswith('.') or '.json' in d:
            continue
        cnt = len([f for f in os.listdir(dp) if f.lower().endswith('.jpg')])
        stats.append((d, cnt))
    
    stats.sort(key=lambda x: x[1])
    
    print('📊 数据集最终统计:')
    print('='*70)
    print(f'{"角色英文名":<20} {"中文名":<12} {"图片数":<6}')
    print('-'*70)
    
    total = 0
    low_count = []
    
    for role, cnt in stats:
        cn_name = role_map.get(role, '未知')
        print(f'{role:<20} {cn_name:<12} {cnt:<6}')
        total += cnt
        if cnt < 100:
            low_count.append((role, cn_name, cnt))
    
    print('-'*70)
    print(f'总计: {len(stats)} 个角色, {total:,} 张图片')
    print(f'满足要求(≥100张): {len(stats) - len(low_count)} 个 ({(len(stats) - len(low_count))/len(stats)*100:.1f}%)')
    
    if low_count:
        print('\n⚠️ 图片数不足100的角色:')
        print('-'*50)
        print(f'{"角色":<20} {"中文名":<12} {"当前":<6} {"还差":<4}')
        print('-'*50)
        for role, cn_name, cnt in low_count:
            print(f'{role:<20} {cn_name:<12} {cnt:<6} {100-cnt:<4}')

if __name__ == '__main__':
    main()