#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速数据复查脚本 - 检查图片质量（快速版）
"""

import os
import json
from PIL import Image
from tqdm import tqdm
import argparse


def is_valid_image(filepath, min_size_kb=10, min_dim=100):
    """快速检查图片是否有效"""
    try:
        # 检查文件大小
        file_size_kb = os.path.getsize(filepath) / 1024
        if file_size_kb < min_size_kb:
            return False, f"文件过小 ({file_size_kb:.1f}KB)"
        
        # 尝试打开图片
        with Image.open(filepath) as img:
            img.verify()  # 快速验证
            width, height = img.size
            if width < min_dim or height < min_dim:
                return False, f"尺寸过小 ({width}x{height})"
        
        return True, ""
    except Exception as e:
        return False, f"损坏或无法读取: {str(e)[:50]}"


def quick_review(data_dir):
    """快速复查数据目录"""
    results = {
        'total': 0,
        'valid': 0,
        'invalid': [],
        'size_map': {}  # 文件大小 -> 路径列表，用于快速检测重复
    }
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                all_files.append(os.path.join(root, filename))
    
    print(f"📁 发现 {len(all_files)} 个图片文件")
    
    for filepath in tqdm(all_files, desc="快速检查"):
        results['total'] += 1
        
        # 检查文件大小用于重复检测
        file_size = os.path.getsize(filepath)
        if file_size not in results['size_map']:
            results['size_map'][file_size] = []
        results['size_map'][file_size].append(filepath)
        
        # 检查图片有效性
        is_ok, reason = is_valid_image(filepath)
        if is_ok:
            results['valid'] += 1
        else:
            results['invalid'].append({
                'path': filepath,
                'reason': reason
            })
    
    # 找出可能的重复（相同大小的文件）
    possible_duplicates = []
    for size, paths in results['size_map'].items():
        if len(paths) > 1 and size > 0:
            possible_duplicates.append({
                'size_bytes': size,
                'files': paths
            })
    
    results['possible_duplicates'] = possible_duplicates
    
    return results


def main():
    parser = argparse.ArgumentParser(description='快速数据复查')
    parser.add_argument('--data-dir', type=str, default='./data/merged_dataset', help='数据目录')
    args = parser.parse_args()
    
    print("🚀 快速数据复查")
    print("=" * 60)
    
    results = quick_review(args.data_dir)
    
    print("\n📊 复查结果:")
    print(f"   总文件数: {results['total']}")
    print(f"   有效文件: {results['valid']}")
    print(f"   无效文件: {len(results['invalid'])}")
    
    if results['invalid']:
        print("\n⚠️ 无效文件示例:")
        for i, item in enumerate(results['invalid'][:5]):
            print(f"   {i+1}. {os.path.basename(item['path'])} - {item['reason']}")
    
    # 统计可能重复的组
    dup_groups = [d for d in results['possible_duplicates'] if len(d['files']) > 1]
    print(f"\n🔍 可能重复组: {len(dup_groups)} 组")
    
    if dup_groups:
        print("   重复组示例:")
        for i, dup in enumerate(dup_groups[:3]):
            print(f"   {i+1}. {len(dup['files'])} 个文件 ({dup['size_bytes']} bytes)")
            for f in dup['files'][:3]:
                print(f"      - {os.path.basename(f)}")
    
    # 计算合格率
    if results['total'] > 0:
        rate = (results['valid'] / results['total']) * 100
        print(f"\n📈 合格率: {rate:.2f}%")
        
        if rate == 100:
            print("✅ 数据质量优秀！")
        elif rate >= 95:
            print("✅ 数据质量良好")
        elif rate >= 90:
            print("⚠️ 数据质量一般")
        else:
            print("❌ 数据质量较差")


if __name__ == '__main__':
    main()
