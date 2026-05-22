#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速筛选脚本 - 分层检查：大小过滤 → 质量检查 → MD5去重
"""

import os
import hashlib
import json
from PIL import Image
from tqdm import tqdm
import argparse

# 过滤参数
MIN_FILE_SIZE_KB = 10
MIN_WIDTH = 100
MIN_HEIGHT = 100


def filter_by_size(filepath):
    """按文件大小过滤"""
    file_size_kb = os.path.getsize(filepath) / 1024
    return file_size_kb >= MIN_FILE_SIZE_KB, file_size_kb


def check_quality(filepath):
    """检查图片质量"""
    try:
        with Image.open(filepath) as img:
            img.verify()  # 验证图片完整性
            img = Image.open(filepath)  # 重新打开
            width, height = img.size
            
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return False, f"尺寸过小 ({width}x{height})"
            
            # 检查是否为纯色图片
            img_rgb = img.convert('RGB')
            unique_colors = len(set(img_rgb.getdata()))
            if unique_colors < 10:
                return False, f"纯色图片 ({unique_colors}色)"
        
        return True, ""
    except Exception as e:
        return False, f"损坏: {str(e)[:30]}"


def compute_md5(filepath):
    """计算MD5哈希"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def quick_filter(data_dir, output_report):
    """快速筛选主函数"""
    results = {
        'total': 0,
        'passed_size': 0,
        'passed_quality': 0,
        'passed_all': 0,
        'failed_size': [],
        'failed_quality': [],
        'duplicates': [],
        'hash_map': {}
    }
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    # 阶段1: 收集所有图片文件
    print("📁 阶段1: 扫描目录...")
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                all_files.append(os.path.join(root, filename))
    
    results['total'] = len(all_files)
    print(f"   发现 {len(all_files)} 个图片文件")
    
    # 阶段2: 按大小过滤
    print("\n📐 阶段2: 大小过滤 (最小 {}KB)...".format(MIN_FILE_SIZE_KB))
    size_passed = []
    for filepath in tqdm(all_files, desc="大小检查"):
        passed, size_kb = filter_by_size(filepath)
        if passed:
            size_passed.append(filepath)
        else:
            results['failed_size'].append({
                'path': filepath,
                'reason': f"文件过小 ({size_kb:.1f}KB)"
            })
    
    results['passed_size'] = len(size_passed)
    print(f"   通过: {len(size_passed)} / 过滤: {len(results['failed_size'])}")
    
    # 阶段3: 质量检查
    print("\n✨ 阶段3: 质量检查...")
    quality_passed = []
    for filepath in tqdm(size_passed, desc="质量检查"):
        passed, reason = check_quality(filepath)
        if passed:
            quality_passed.append(filepath)
        else:
            results['failed_quality'].append({
                'path': filepath,
                'reason': reason
            })
    
    results['passed_quality'] = len(quality_passed)
    print(f"   通过: {len(quality_passed)} / 过滤: {len(results['failed_quality'])}")
    
    # 阶段4: MD5去重
    print("\n🔍 阶段4: MD5去重...")
    for filepath in tqdm(quality_passed, desc="MD5计算"):
        file_hash = compute_md5(filepath)
        
        if file_hash in results['hash_map']:
            results['duplicates'].append({
                'original': results['hash_map'][file_hash],
                'duplicate': filepath,
                'hash': file_hash
            })
        else:
            results['hash_map'][file_hash] = filepath
            results['passed_all'] += 1
    
    print(f"   唯一: {results['passed_all']} / 重复: {len(results['duplicates'])}")
    
    # 保存报告
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results


def print_summary(results):
    """打印筛选摘要"""
    print("\n" + "=" * 60)
    print("📊 快速筛选报告")
    print("=" * 60)
    
    print(f"\n📁 文件统计:")
    print(f"   总文件数: {results['total']}")
    print(f"   通过大小过滤: {results['passed_size']}")
    print(f"   通过质量检查: {results['passed_quality']}")
    print(f"   最终唯一文件: {results['passed_all']}")
    
    print(f"\n❌ 过滤统计:")
    print(f"   大小过滤: {len(results['failed_size'])} 个")
    print(f"   质量过滤: {len(results['failed_quality'])} 个")
    print(f"   重复文件: {len(results['duplicates'])} 个")
    
    if results['failed_size']:
        print(f"\n📉 大小过滤示例:")
        for i, item in enumerate(results['failed_size'][:3]):
            print(f"   {i+1}. {os.path.basename(item['path'])} - {item['reason']}")
    
    if results['failed_quality']:
        print(f"\n📉 质量过滤示例:")
        for i, item in enumerate(results['failed_quality'][:3]):
            print(f"   {i+1}. {os.path.basename(item['path'])} - {item['reason']}")
    
    if results['duplicates']:
        print(f"\n📉 重复示例:")
        for i, item in enumerate(results['duplicates'][:3]):
            print(f"   {i+1}. {os.path.basename(item['duplicate'])} ↔ {os.path.basename(item['original'])}")
    
    # 计算通过率
    if results['total'] > 0:
        rate = (results['passed_all'] / results['total']) * 100
        print(f"\n📈 最终通过率: {rate:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='快速筛选 - 分层检查：大小 → 质量 → MD5')
    parser.add_argument('--data-dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--output', type=str, default='quick_filter_report.json', help='输出报告')
    parser.add_argument('--min-size', type=int, default=10, help='最小文件大小(KB)')
    args = parser.parse_args()
    
    global MIN_FILE_SIZE_KB
    MIN_FILE_SIZE_KB = args.min_size
    
    print("🚀 快速筛选开始")
    print("=" * 60)
    print(f"参数: 最小文件大小={MIN_FILE_SIZE_KB}KB, 最小尺寸={MIN_WIDTH}x{MIN_HEIGHT}")
    
    results = quick_filter(args.data_dir, args.output)
    print_summary(results)
    
    print(f"\n📋 报告已保存到: {args.output}")


if __name__ == '__main__':
    main()
