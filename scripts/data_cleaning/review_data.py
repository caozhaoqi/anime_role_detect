#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据复查脚本 - 检查整个data目录中的图片质量
检测重复、低质量、损坏图片
"""

import os
import hashlib
import json
from PIL import Image
from tqdm import tqdm
import argparse


def get_image_hash(filepath):
    """计算图片文件的MD5哈希值"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def is_low_quality(filepath, min_size_kb=10, min_width=100, min_height=100):
    """检查图片是否为低质量"""
    try:
        # 检查文件大小
        file_size_kb = os.path.getsize(filepath) / 1024
        if file_size_kb < min_size_kb:
            return True, f"文件过小 ({file_size_kb:.1f}KB)"
        
        # 检查图片尺寸
        with Image.open(filepath) as img:
            width, height = img.size
            if width < min_width or height < min_height:
                return True, f"尺寸过小 ({width}x{height})"
            
            # 检查是否为纯色图片
            img = img.convert('RGB')
            pixels = img.getdata()
            unique_colors = set(pixels)
            if len(unique_colors) < 10:
                return True, f"纯色图片 ({len(unique_colors)}种颜色)"
        
        return False, ""
    except Exception as e:
        return True, f"无法读取: {str(e)}"


def review_data_directory(data_dir):
    """复查数据目录"""
    results = {
        'total_files': 0,
        'image_files': 0,
        'duplicates': [],
        'low_quality': [],
        'damaged': [],
        'hash_map': {}
    }
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')
    
    print(f"\n🔍 正在复查目录: {data_dir}")
    print("=" * 60)
    
    # 遍历目录
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                filepath = os.path.join(root, filename)
                all_files.append(filepath)
    
    print(f"📁 发现 {len(all_files)} 个图片文件")
    
    # 检查重复和质量
    for filepath in tqdm(all_files, desc="检查图片"):
        results['total_files'] += 1
        
        # 计算哈希值检测重复
        try:
            file_hash = get_image_hash(filepath)
            
            if file_hash in results['hash_map']:
                results['duplicates'].append({
                    'original': results['hash_map'][file_hash],
                    'duplicate': filepath
                })
            else:
                results['hash_map'][file_hash] = filepath
            
            # 检查质量
            is_bad, reason = is_low_quality(filepath)
            if is_bad:
                if "无法读取" in reason:
                    results['damaged'].append({
                        'path': filepath,
                        'reason': reason
                    })
                else:
                    results['low_quality'].append({
                        'path': filepath,
                        'reason': reason
                    })
            
            results['image_files'] += 1
        except Exception as e:
            results['damaged'].append({
                'path': filepath,
                'reason': f"处理异常: {str(e)}"
            })
    
    return results


def generate_report(results, output_file):
    """生成复查报告"""
    report = {
        'summary': {
            'total_files': results['total_files'],
            'image_files': results['image_files'],
            'duplicate_groups': len(results['duplicates']),
            'low_quality_count': len(results['low_quality']),
            'damaged_count': len(results['damaged']),
            'unique_images': len(results['hash_map'])
        },
        'duplicates': results['duplicates'],
        'low_quality': results['low_quality'],
        'damaged': results['damaged']
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report


def print_summary(report):
    """打印复查摘要"""
    print("\n📊 数据复查报告")
    print("=" * 60)
    
    print(f"\n📁 文件统计:")
    print(f"   总文件数: {report['summary']['total_files']}")
    print(f"   图片文件: {report['summary']['image_files']}")
    print(f"   唯一图片: {report['summary']['unique_images']}")
    
    print(f"\n⚠️ 问题统计:")
    print(f"   重复图片: {report['summary']['duplicate_groups']} 组")
    print(f"   低质量图片: {report['summary']['low_quality_count']} 张")
    print(f"   损坏图片: {report['summary']['damaged_count']} 张")
    
    if report['duplicates']:
        print(f"\n🔍 重复图片示例:")
        for i, dup in enumerate(report['duplicates'][:5]):
            print(f"   {i+1}. 重复: {os.path.basename(dup['duplicate'])}")
            print(f"      ↳ 原始: {os.path.basename(dup['original'])}")
    
    if report['low_quality']:
        print(f"\n🔍 低质量图片示例:")
        for i, lq in enumerate(report['low_quality'][:5]):
            print(f"   {i+1}. {os.path.basename(lq['path'])} - {lq['reason']}")
    
    if report['damaged']:
        print(f"\n🔍 损坏图片示例:")
        for i, dmg in enumerate(report['damaged'][:5]):
            print(f"   {i+1}. {os.path.basename(dmg['path'])} - {dmg['reason']}")
    
    # 计算脏数据比例
    total_issues = report['summary']['duplicate_groups'] + report['summary']['low_quality_count'] + report['summary']['damaged_count']
    if report['summary']['image_files'] > 0:
        dirty_ratio = (total_issues / report['summary']['image_files']) * 100
        print(f"\n📈 脏数据比例: {dirty_ratio:.2f}%")
        
        if dirty_ratio == 0:
            print("✅ 数据质量优秀！")
        elif dirty_ratio < 5:
            print("✅ 数据质量良好")
        elif dirty_ratio < 10:
            print("⚠️ 数据质量一般，建议清理")
        else:
            print("❌ 数据质量较差，需要清理")


def main():
    parser = argparse.ArgumentParser(description='数据复查脚本 - 检测重复、低质量、损坏图片')
    parser.add_argument('--data-dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--output', type=str, default='data_review_report.json', help='输出报告文件')
    args = parser.parse_args()
    
    print("🚀 数据复查开始")
    print("=" * 60)
    
    # 检查目录
    if not os.path.exists(args.data_dir):
        print(f"❌ 目录不存在: {args.data_dir}")
        return
    
    # 执行复查
    results = review_data_directory(args.data_dir)
    
    # 生成报告
    report = generate_report(results, args.output)
    
    # 打印摘要
    print_summary(report)
    
    print(f"\n📋 完整报告已保存到: {args.output}")


if __name__ == '__main__':
    main()
