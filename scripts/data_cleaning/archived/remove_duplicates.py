#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重复图片删除脚本 - 基于MD5哈希检测并删除重复图片
"""

import os
import hashlib
import json
import argparse
from tqdm import tqdm


def get_image_hash(img_path):
    """计算图片的MD5哈希值"""
    try:
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def find_duplicates(data_dir):
    """查找数据集中的重复图片"""
    hashes = {}
    duplicates = []
    
    print(f"🔍 正在扫描数据集: {data_dir}")
    
    # 遍历所有图片文件
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.png', '.webp', '.jpeg')):
                continue
            
            img_path = os.path.join(root, filename)
            img_hash = get_image_hash(img_path)
            
            if img_hash:
                if img_hash in hashes:
                    duplicates.append({
                        'original': hashes[img_hash],
                        'duplicate': img_path,
                        'hash': img_hash
                    })
                else:
                    hashes[img_hash] = img_path
    
    return duplicates, len(hashes)


def remove_duplicates(duplicates, dry_run=False):
    """删除重复图片"""
    removed_count = 0
    removed_files = []
    
    print(f"\n🗑️ 准备删除 {len(duplicates)} 张重复图片")
    
    for dup in tqdm(duplicates, desc="删除重复图片"):
        duplicate_path = dup['duplicate']
        
        if dry_run:
            print(f"  [模拟删除] {duplicate_path}")
        else:
            try:
                os.remove(duplicate_path)
                removed_count += 1
                removed_files.append({
                    'path': duplicate_path,
                    'original': dup['original'],
                    'hash': dup['hash']
                })
            except Exception as e:
                print(f"  ❌ 删除失败 {duplicate_path}: {e}")
    
    return removed_count, removed_files


def main():
    parser = argparse.ArgumentParser(description='删除数据集中的重复图片')
    parser.add_argument('--data-dir', type=str, default='./data/merged_dataset', help='数据集目录')
    parser.add_argument('--dry-run', action='store_true', help='模拟删除，不实际删除文件')
    parser.add_argument('--output', type=str, default='duplicate_report.json', help='输出报告路径')
    args = parser.parse_args()
    
    # 查找重复图片
    duplicates, unique_count = find_duplicates(args.data_dir)
    
    print(f"\n📊 扫描结果:")
    print(f"  发现重复图片: {len(duplicates)} 张")
    print(f"  唯一图片: {unique_count} 张")
    
    if duplicates:
        # 删除重复图片
        removed_count, removed_files = remove_duplicates(duplicates, args.dry_run)
        
        # 生成报告
        report = {
            'data_dir': args.data_dir,
            'total_duplicates_found': len(duplicates),
            'unique_images_before': unique_count + len(duplicates),
            'unique_images_after': unique_count,
            'removed_count': removed_count,
            'dry_run': args.dry_run,
            'removed_files': removed_files[:100]  # 只保存前100个删除记录
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 去重完成!")
        print(f"  删除重复图片: {removed_count} 张")
        print(f"  剩余唯一图片: {unique_count} 张")
        print(f"  报告已保存至: {args.output}")
    else:
        print("\n🎉 未发现重复图片，无需删除")


if __name__ == '__main__':
    main()
