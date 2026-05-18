#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查并删除数据目录中的重复图片文件
"""
import os
import hashlib
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data')

def get_file_hash(file_path):
    """计算文件的MD5哈希值"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"❌ 读取文件失败 {file_path}: {e}")
        return None

def find_duplicates():
    """查找重复文件"""
    hash_to_files = defaultdict(list)
    
    print('🔍 正在扫描所有图片文件...')
    total_files = 0
    
    for img_file in DATA_DIR.rglob('*.jpg'):
        if img_file.is_file() and not img_file.name.startswith('.'):
            file_hash = get_file_hash(img_file)
            if file_hash:
                hash_to_files[file_hash].append(img_file)
                total_files += 1
    
    print(f'📊 共扫描 {total_files} 个文件')
    
    # 找出重复的文件组
    duplicates = []
    for file_hash, files in hash_to_files.items():
        if len(files) > 1:
            duplicates.append(files)
    
    return duplicates

def delete_duplicates(duplicates):
    """删除重复文件（保留第一个）"""
    total_deleted = 0
    
    for group in duplicates:
        # 保留第一个文件，删除其余的
        keep_file = group[0]
        delete_files = group[1:]
        
        print(f"\n📁 重复组 ({len(group)}个文件):")
        print(f"   ✅ 保留: {keep_file.relative_to(DATA_DIR)}")
        
        for del_file in delete_files:
            try:
                os.remove(del_file)
                print(f"   🗑️ 删除: {del_file.relative_to(DATA_DIR)}")
                total_deleted += 1
            except Exception as e:
                print(f"   ❌ 删除失败 {del_file.name}: {e}")
    
    return total_deleted

def main():
    print('🚀 开始检查重复文件\n')
    
    duplicates = find_duplicates()
    
    if not duplicates:
        print('✅ 未发现重复文件')
        return
    
    print(f'\n⚠️ 发现 {len(duplicates)} 组重复文件，共 {sum(len(g) for g in duplicates)} 个文件')
    
    # 确认删除
    confirm = input('\n确定要删除重复文件吗？(y/n): ')
    if confirm.lower() != 'y':
        print('❌ 已取消操作')
        return
    
    total_deleted = delete_duplicates(duplicates)
    print(f'\n🎉 操作完成！共删除 {total_deleted} 个重复文件')

if __name__ == '__main__':
    main()