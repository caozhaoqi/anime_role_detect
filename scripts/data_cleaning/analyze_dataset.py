#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集分析报告 - 角色占比统计
"""

import os
import json
from collections import defaultdict
import argparse


def analyze_directory(data_dir):
    """分析数据目录"""
    stats = {
        'total_images': 0,
        'total_roles': 0,
        'role_stats': defaultdict(int),
        'directory_stats': defaultdict(int),
        'image_formats': defaultdict(int),
        'size_distribution': defaultdict(int)
    }
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    # 遍历目录
    for root, dirs, files in os.walk(data_dir):
        # 跳过非数据集目录
        dir_name = os.path.basename(root)
        parent_dir = os.path.basename(os.path.dirname(root))
        
        # 判断是否为角色目录（在数据目录下）
        data_parent_dirs = ['expanded_dataset', 'merged_dataset', 'training_dataset', 
                           'optimized_detection_v2', 'yunet_optimized_results']
        
        # 如果data_dir直接包含角色目录（如final_dataset），也视为角色目录
        is_role_dir = (parent_dir in data_parent_dirs) or (os.path.dirname(root) == data_dir and parent_dir == os.path.basename(data_dir))
        
        if is_role_dir or (parent_dir == os.path.basename(data_dir)):
            # 这是角色目录
            role_name = dir_name
            
            for filename in files:
                if filename.lower().endswith(image_extensions):
                    filepath = os.path.join(root, filename)
                    
                    stats['total_images'] += 1
                    stats['role_stats'][role_name] += 1
                    stats['directory_stats'][parent_dir] += 1
                    
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
    print("📊 数据集分析报告")
    print("=" * 70)
    
    # 基本统计
    print(f"\n📈 基本统计:")
    print(f"   总图片数: {stats['total_images']:,}")
    print(f"   角色数量: {stats['total_roles']}")
    
    # 目录统计
    print(f"\n📁 目录统计:")
    for dir_name, count in stats['directory_stats'].items():
        print(f"   {dir_name}: {count:,}")
    
    # 格式统计
    print(f"\n🖼️ 图片格式:")
    for ext, count in stats['image_formats'].items():
        print(f"   {ext}: {count:,} ({(count/stats['total_images']*100):.1f}%)")
    
    # 大小分布
    print(f"\n📦 文件大小分布:")
    for size_range, count in stats['size_distribution'].items():
        print(f"   {size_range}: {count:,} ({(count/stats['total_images']*100):.1f}%)")
    
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
        if count < 5:
            buckets['1-4'] += 1
        elif count < 10:
            buckets['5-9'] += 1
        elif count < 20:
            buckets['10-19'] += 1
        elif count < 50:
            buckets['20-49'] += 1
        elif count < 100:
            buckets['50-99'] += 1
        else:
            buckets['100+'] += 1
    
    for bucket in ['1-4', '5-9', '10-19', '20-49', '50-99', '100+']:
        print(f"   {bucket:5s}: {buckets.get(bucket, 0)} 个角色")
    
    # 保存报告
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n💾 报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='数据集分析 - 角色占比统计')
    parser.add_argument('--data-dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--output', type=str, default='dataset_report.json', help='输出报告')
    args = parser.parse_args()
    
    print("🚀 开始分析数据集...")
    stats = analyze_directory(args.data_dir)
    print_report(stats, args.output)
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
