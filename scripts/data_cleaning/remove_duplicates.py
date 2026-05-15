#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除数据集中的重复图片（保留每组中的一个副本）
"""
import os
import json
import shutil

def main():
    DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
    REPORT_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/duplicate_report.json'
    LOG_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/deleted_duplicates.log'
    
    # 读取重复报告
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        duplicates = json.load(f)
    
    print("🗑️ 开始删除重复图片...")
    print(f"发现 {len(duplicates)} 组重复图片")
    
    deleted_count = 0
    deleted_files = []
    
    for idx, dup in enumerate(duplicates, 1):
        if idx % 100 == 0:
            print(f"  已处理: {idx}/{len(duplicates)} 组")
        
        # 保留第一个文件（按角色名排序后的第一个）
        # 或者可以选择保留最大的文件
        files = sorted(dup['files'], key=lambda x: (x['role'], x['filename']))
        keep_file = files[0]
        
        # 删除其他重复文件
        for f in files[1:]:
            file_path = f['path']
            try:
                os.remove(file_path)
                deleted_count += 1
                deleted_files.append({
                    'hash': dup['hash'],
                    'role': f['role'],
                    'filename': f['filename'],
                    'size': f['size'],
                    'kept_role': keep_file['role'],
                    'kept_filename': keep_file['filename']
                })
            except Exception as e:
                print(f"❌ 删除失败: {file_path} - {str(e)}")
    
    # 保存删除日志
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(deleted_files, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ 删除完成")
    print("=" * 80)
    print(f"删除重复图片数: {deleted_count} 张")
    print(f"删除日志已保存到: {LOG_FILE}")
    
    # 统计删除后的数据集状态
    total_files = 0
    role_counts = {}
    
    for role_dir in os.listdir(DATASET_PATH):
        role_path = os.path.join(DATASET_PATH, role_dir)
        if not os.path.isdir(role_path) or role_dir.startswith('.') or role_dir.endswith('.json'):
            continue
        
        count = len([f for f in os.listdir(role_path) if f.lower().endswith('.jpg')])
        total_files += count
        role_counts[role_dir] = count
    
    print(f"\n📊 删除后数据集统计:")
    print(f"总角色数: {len(role_counts)}")
    print(f"总图片数: {total_files:,}")
    
    # 检查是否有目录图片数少于100
    low_count_roles = [(role, cnt) for role, cnt in role_counts.items() if cnt < 100]
    if low_count_roles:
        print("\n⚠️ 图片数不足100的角色:")
        for role, cnt in sorted(low_count_roles, key=lambda x: x[1]):
            print(f"  {role}: {cnt} 张")

if __name__ == '__main__':
    main()