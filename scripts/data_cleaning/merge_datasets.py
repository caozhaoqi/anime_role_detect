#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并数据集 - 将 expanded_dataset 和 merged_dataset 合并
"""

import os
import hashlib
import shutil
from tqdm import tqdm
import argparse


def get_image_hash(filepath):
    """计算文件MD5"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def merge_folders(src_dirs, dst_dir):
    """合并多个文件夹到目标目录"""
    stats = {
        'total_processed': 0,
        'copied': 0,
        'skipped_duplicates': 0,
        'errors': 0
    }
    
    seen_hashes = set()
    os.makedirs(dst_dir, exist_ok=True)
    
    # 首先收集所有角色目录
    all_roles = set()
    for src_dir in src_dirs:
        if os.path.exists(src_dir):
            for role in os.listdir(src_dir):
                if os.path.isdir(os.path.join(src_dir, role)):
                    all_roles.add(role)
    
    print(f"📁 发现 {len(all_roles)} 个角色")
    
    # 处理每个角色
    for role in tqdm(all_roles, desc="合并数据集"):
        dst_role_dir = os.path.join(dst_dir, role)
        os.makedirs(dst_role_dir, exist_ok=True)
        
        # 从每个源目录处理该角色
        for src_dir in src_dirs:
            src_role_dir = os.path.join(src_dir, role)
            
            if not os.path.exists(src_role_dir):
                continue
            
            # 遍历该角色的图片
            for filename in os.listdir(src_role_dir):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    continue
                
                src_path = os.path.join(src_role_dir, filename)
                stats['total_processed'] += 1
                
                # 计算哈希值去重
                try:
                    file_hash = get_image_hash(src_path)
                    
                    if file_hash in seen_hashes:
                        stats['skipped_duplicates'] += 1
                        continue
                    
                    # 检查目标目录是否已有同名文件
                    dst_path = os.path.join(dst_role_dir, filename)
                    
                    # 如果文件名已存在，添加序号
                    base_name, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dst_path):
                        new_filename = f"{base_name}_{counter}{ext}"
                        dst_path = os.path.join(dst_role_dir, new_filename)
                        counter += 1
                    
                    # 复制文件
                    shutil.copy2(src_path, dst_path)
                    seen_hashes.add(file_hash)
                    stats['copied'] += 1
                    
                except Exception as e:
                    print(f"❌ 处理失败 {src_path}: {e}")
                    stats['errors'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='合并数据集')
    parser.add_argument('--src1', type=str, default='data/merged_dataset', help='源目录1')
    parser.add_argument('--src2', type=str, default='data/expanded_dataset', help='源目录2')
    parser.add_argument('--dst', type=str, default='data/final_dataset', help='目标目录')
    args = parser.parse_args()
    
    print("🚀 开始合并数据集")
    print("=" * 60)
    
    # 检查源目录
    src_dirs = []
    for name, path in [('源1', args.src1), ('源2', args.src2)]:
        if os.path.exists(path):
            src_dirs.append(path)
            role_count = len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            print(f"📁 {name}: {path} ({role_count} 个角色)")
        else:
            print(f"⚠️ {name} 不存在: {path}")
    
    if len(src_dirs) < 1:
        print("❌ 没有有效的源目录")
        return
    
    # 执行合并
    stats = merge_folders(src_dirs, args.dst)
    
    # 统计结果
    print("\n" + "=" * 60)
    print("✅ 合并完成")
    print("=" * 60)
    
    print(f"\n📊 合并统计:")
    print(f"   总处理: {stats['total_processed']}")
    print(f"   复制文件: {stats['copied']}")
    print(f"   跳过重复: {stats['skipped_duplicates']}")
    print(f"   错误: {stats['errors']}")
    
    # 统计目标目录
    if os.path.exists(args.dst):
        roles = [d for d in os.listdir(args.dst) if os.path.isdir(os.path.join(args.dst, d))]
        total_images = 0
        for role in roles:
            total_images += len([f for f in os.listdir(os.path.join(args.dst, role))
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        
        print(f"\n📁 目标目录: {args.dst}")
        print(f"   角色数: {len(roles)}")
        print(f"   总图片: {total_images:,}")


if __name__ == '__main__':
    main()
