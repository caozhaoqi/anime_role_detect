#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测数据集中的重复图片并生成报告
"""
import os
import hashlib
import json
from collections import defaultdict

def calculate_md5(file_path):
    """计算文件MD5哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"❌ 计算MD5失败: {file_path} - {str(e)}")
        return None

def find_duplicates(dataset_path):
    """查找重复图片"""
    print("🔍 开始检测重复图片...")
    
    # 按哈希值分组
    hash_to_files = defaultdict(list)
    total_files = 0
    processed_files = 0
    
    # 遍历所有角色目录
    for role_dir in sorted(os.listdir(dataset_path)):
        role_path = os.path.join(dataset_path, role_dir)
        if not os.path.isdir(role_path) or role_dir.startswith('.') or role_dir.endswith('.json'):
            continue
        
        jpg_files = [f for f in os.listdir(role_path) if f.lower().endswith('.jpg')]
        total_files += len(jpg_files)
        
        for filename in jpg_files:
            file_path = os.path.join(role_path, filename)
            file_hash = calculate_md5(file_path)
            
            if file_hash:
                hash_to_files[file_hash].append({
                    'role': role_dir,
                    'filename': filename,
                    'path': file_path,
                    'size': os.path.getsize(file_path)
                })
            processed_files += 1
            if processed_files % 1000 == 0:
                print(f"  已处理: {processed_files}/{total_files}")
    
    # 筛选重复项（出现次数 >= 2）
    duplicates = []
    for file_hash, files in hash_to_files.items():
        if len(files) >= 2:
            duplicates.append({
                'hash': file_hash,
                'count': len(files),
                'files': files
            })
    
    # 按重复数量排序
    duplicates.sort(key=lambda x: x['count'], reverse=True)
    
    return duplicates, total_files

def main():
    DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
    OUTPUT_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/duplicate_report.json'
    
    duplicates, total_files = find_duplicates(DATASET_PATH)
    
    # 输出报告
    print("\n" + "=" * 80)
    print("📊 重复图片检测报告")
    print("=" * 80)
    print(f"总文件数: {total_files:,}")
    print(f"重复组数: {len(duplicates)}")
    
    total_duplicate_files = sum(d['count'] for d in duplicates)
    print(f"重复文件数: {total_duplicate_files:,}")
    print(f"唯一文件数: {total_files - total_duplicate_files + len(duplicates):,}")
    
    # 输出前20组重复详情
    print("\n📋 重复详情（按重复数量排序，显示前20组）:")
    for i, dup in enumerate(duplicates[:20], 1):
        print(f"\n{i}. Hash: {dup['hash']}")
        print(f"   重复次数: {dup['count']}")
        print(f"   文件列表:")
        for f in dup['files']:
            print(f"     - {f['role']}/{f['filename']} ({f['size']:,} bytes)")
    
    if len(duplicates) > 20:
        print(f"\n... 还有 {len(duplicates) - 20} 组重复")
    
    # 保存报告到JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(duplicates, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 报告已保存到: {OUTPUT_FILE}")
    
    # 统计每个角色的重复情况
    print("\n" + "=" * 80)
    print("📈 各角色重复图片统计")
    print("=" * 80)
    
    role_duplicate_count = defaultdict(int)
    for dup in duplicates:
        for f in dup['files']:
            role_duplicate_count[f['role']] += 1
    
    # 按重复数量排序
    sorted_roles = sorted(role_duplicate_count.items(), key=lambda x: x[1], reverse=True)
    
    for role, count in sorted_roles[:10]:
        print(f"{role:<20} {count} 张重复")
    
    if len(sorted_roles) > 10:
        print(f"... 还有 {len(sorted_roles) - 10} 个角色有重复")

if __name__ == '__main__':
    main()