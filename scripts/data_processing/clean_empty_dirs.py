#!/usr/bin/env python3
"""安全删除空目录，保留高质量数据"""

import os
import shutil
from pathlib import Path

def delete_empty_dirs(data_dir, backup_dir=None, dry_run=True):
    """
    删除空目录
    
    Args:
        data_dir: 数据目录路径
        backup_dir: 备份目录（可选），空目录会先移动到这里
        dry_run: 模拟运行，不实际删除
    """
    data_path = Path(data_dir)
    
    # 获取所有角色目录
    char_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    # 分类目录
    empty_dirs = []
    non_empty_dirs = []
    
    for char_dir in char_dirs:
        files = list(char_dir.glob('*'))
        if len(files) == 0:
            empty_dirs.append(char_dir)
        else:
            non_empty_dirs.append(char_dir)
    
    print(f'📁 总角色目录数: {len(char_dirs)}')
    print(f'🗑️ 空目录数: {len(empty_dirs)}')
    print(f'✅ 非空目录数: {len(non_empty_dirs)}')
    
    if empty_dirs:
        print(f'\n📋 空目录列表:')
        for i, d in enumerate(empty_dirs, 1):
            print(f'  {i:2d}. {d.name}')
    
    if dry_run:
        print(f'\n⚠️ 模拟模式：以上目录将被删除，但未实际执行')
        print(f'   使用 --execute 参数执行实际删除')
        return
    
    # 创建备份目录
    if backup_dir:
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        print(f'\n📦 将空目录移动到备份: {backup_path}')
    
    # 执行删除
    deleted_count = 0
    for empty_dir in empty_dirs:
        try:
            if backup_dir:
                # 移动到备份目录
                backup_target = backup_path / empty_dir.name
                shutil.move(str(empty_dir), str(backup_target))
                print(f'   🔄 移动: {empty_dir.name} -> {backup_target.name}')
            else:
                # 直接删除
                empty_dir.rmdir()
                print(f'   🗑️ 删除: {empty_dir.name}')
            deleted_count += 1
        except Exception as e:
            print(f'   ❌ 失败: {empty_dir.name} - {e}')
    
    print(f'\n✅ 完成！共处理 {deleted_count} 个空目录')
    
    # 统计剩余数据
    remaining_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    total_images = sum(len(list(d.glob('*'))) for d in remaining_dirs)
    
    print(f'\n📊 删除后统计:')
    print(f'   剩余角色目录: {len(remaining_dirs)} 个')
    print(f'   剩余图片总数: {total_images} 张')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='删除空目录，保留高质量数据')
    parser.add_argument('--execute', action='store_true', help='实际执行删除操作')
    parser.add_argument('--backup', action='store_true', help='将空目录移动到备份目录')
    
    args = parser.parse_args()
    
    data_dir = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset'
    backup_dir = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/empty_backup' if args.backup else None
    
    delete_empty_dirs(data_dir, backup_dir, dry_run=not args.execute)