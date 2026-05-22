#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析最终数据集 - 角色占比统计
"""

import os
import json
from collections import defaultdict


def analyze_final_dataset(data_dir):
    """分析最终数据集（直接包含角色目录）"""
    stats = {
        'total_images': 0,
        'total_roles': 0,
        'role_stats': defaultdict(int),
        'image_formats': defaultdict(int),
        'size_distribution': defaultdict(int)
    }
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    # 遍历角色目录
    if not os.path.exists(data_dir):
        return stats
    
    for role in os.listdir(data_dir):
        role_dir = os.path.join(data_dir, role)
        
        if not os.path.isdir(role_dir):
            continue
        
        for filename in os.listdir(role_dir):
            if not filename.lower().endswith(image_extensions):
                continue
            
            filepath = os.path.join(role_dir, filename)
            
            stats['total_images'] += 1
            stats['role_stats'][role] += 1
            
            # 统计格式
            ext = os.path.splitext(filename)[1].lower()
            stats['image_formats'][ext] += 1
            
            # 统计文件大小
            try:
                size_kb = os.path.getsize(filepath) / 1024
                if size_kb < 50:
                    stats['size_distribution']['<50KB'] += 1
                elif size_kb < 100:
                    stats['size_distribution']['50-100KB'] += 1
                elif size_kb < 500:
                    stats['size_distribution']['100-500KB'] += 1
                elif size_kb < 1000:
                    stats['size_distribution']['500KB-1MB'] += 1
                else:
                    stats['size_distribution']['>1MB'] += 1
            except:
                pass
    
    stats['total_roles'] = len(stats['role_stats'])
    
    # 排序角色统计
    stats['role_stats'] = dict(sorted(stats['role_stats'].items(), 
                                     key=lambda x: x[1], reverse=True))
    
    return stats


def print_report(stats, output_file=None):
    """打印分析报告"""
    print("\n" + "=" * 70)
    print("📊 最终数据集分析报告")
    print("=" * 70)
    
    # 基本统计
    print(f"\n📈 基本统计:")
    print(f"   总图片数: {stats['total_images']:,}")
    print(f"   角色数量: {stats['total_roles']}")
    print(f"   平均/角色: {(stats['total_images']/stats['total_roles']):.1f}")
    
    # 格式统计
    print(f"\n🖼️ 图片格式:")
    for ext, count in stats['image_formats'].items():
        print(f"   {ext}: {count:,} ({(count/stats['total_images']*100):.1f}%)")
    
    # 大小分布
    print(f"\n📦 文件大小分布:")
    for size_range in ['<50KB', '50-100KB', '100-500KB', '500KB-1MB', '>1MB']:
        count = stats['size_distribution'].get(size_range, 0)
        print(f"   {size_range:10s}: {count:6,} ({(count/stats['total_images']*100):.1f}%)")
    
    # 角色统计 - 前20
    print(f"\n👥 角色图片数 TOP 20:")
    for i, (role, count) in enumerate(list(stats['role_stats'].items())[:20], 1):
        print(f"   {i:2d}. {role:20s}: {count:4,}")
    
    # 角色统计 - 后10（最少）
    print(f"\n👥 角色图片数 BOTTOM 10:")
    for i, (role, count) in enumerate(list(stats['role_stats'].items())[-10:], 1):
        print(f"   {i:2d}. {role:20s}: {count:4,}")
    
    # 角色分布区间
    print(f"\n📊 角色图片数分布:")
    buckets = defaultdict(int)
    for count in stats['role_stats'].values():
        if count < 10:
            buckets['1-9'] += 1
        elif count < 25:
            buckets['10-24'] += 1
        elif count < 50:
            buckets['25-49'] += 1
        elif count < 75:
            buckets['50-74'] += 1
        elif count < 100:
            buckets['75-99'] += 1
        elif count < 150:
            buckets['100-149'] += 1
        else:
            buckets['150+'] += 1
    
    for bucket in ['1-9', '10-24', '25-49', '50-74', '75-99', '100-149', '150+']:
        print(f"   {bucket:6s}: {buckets.get(bucket, 0)} 个角色")
    
    # 保存报告
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n💾 报告已保存到: {output_file}")


def main():
    data_dir = "data/final_dataset"
    output_file = "data/final_dataset_report.json"
    
    print("🚀 开始分析最终数据集...")
    stats = analyze_final_dataset(data_dir)
    print_report(stats, output_file)
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
