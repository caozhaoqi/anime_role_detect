#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 final_dataset 目录，只保留标准英文名的角色
"""

import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect")

# 角色列表文件
ROLE_LIST_FILE = PROJECT_ROOT / "auto_spider_img" / "loli-role.txt"

# 目标目录
TARGET_DIR = PROJECT_ROOT / "data" / "final_dataset"


def get_standard_role_names():
    """获取标准英文名列表"""
    standard_names = set()
    
    if not ROLE_LIST_FILE.exists():
        print(f"警告: 角色列表文件不存在: {ROLE_LIST_FILE}")
        return standard_names
    
    with open(ROLE_LIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                english_name = parts[2]
                standard_names.add(english_name)
    
    return standard_names


def cleanup_final_dataset():
    """清理 final_dataset 目录"""
    if not TARGET_DIR.exists():
        print(f"目标目录不存在: {TARGET_DIR}")
        return
    
    # 获取标准英文名列表
    standard_names = get_standard_role_names()
    print(f"标准英文名数量: {len(standard_names)}")
    
    # 统计信息
    stats = {
        'total_dirs': 0,
        'kept_dirs': 0,
        'removed_dirs': 0,
        'kept_files': 0,
        'removed_files': 0,
    }
    
    # 遍历所有角色目录
    for role_dir in TARGET_DIR.iterdir():
        if not role_dir.is_dir():
            continue
        
        stats['total_dirs'] += 1
        
        # 统计该目录的文件数
        file_count = len([f for f in role_dir.iterdir() if f.is_file()])
        
        # 检查是否为标准英文名
        if role_dir.name in standard_names:
            # 保留
            stats['kept_dirs'] += 1
            stats['kept_files'] += file_count
            print(f"保留: {role_dir.name} ({file_count} 个文件)")
        else:
            # 删除
            try:
                shutil.rmtree(role_dir)
                stats['removed_dirs'] += 1
                stats['removed_files'] += file_count
                print(f"删除: {role_dir.name} ({file_count} 个文件)")
            except Exception as e:
                print(f"删除失败: {role_dir.name}, 错误: {e}")
    
    # 输出统计报告
    print("\n" + "="*60)
    print("清理完成!")
    print("="*60)
    print(f"总角色目录数: {stats['total_dirs']}")
    print(f"保留目录数: {stats['kept_dirs']}")
    print(f"删除目录数: {stats['removed_dirs']}")
    print(f"保留文件数: {stats['kept_files']}")
    print(f"删除文件数: {stats['removed_files']}")
    print("="*60)
    
    # 列出最终的角色目录
    print("\n最终角色目录列表:")
    final_dirs = [d for d in TARGET_DIR.iterdir() if d.is_dir()]
    for role_dir in sorted(final_dirs, key=lambda x: x.name):
        file_count = len([f for f in role_dir.iterdir() if f.is_file()])
        print(f"  {role_dir.name}: {file_count} 个文件")


if __name__ == "__main__":
    cleanup_final_dataset()
