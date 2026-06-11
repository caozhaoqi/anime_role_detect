#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片目录合并工具
合并多个图片目录，基于MD5去重，减少重复数据
"""

import os
import sys
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple
from loguru import logger
from collections import defaultdict


def calculate_md5(file_path: Path) -> str:
    """计算文件MD5值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def scan_directory(dir_path: Path) -> Dict[str, Tuple[Path, str]]:
    """
    扫描目录，返回MD5到文件路径的映射
    
    Returns:
        Dict[str, Tuple[Path, str]]: {md5: (file_path, character_name)}
    """
    md5_map = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    
    if not dir_path.exists():
        logger.warning(f"目录不存在: {dir_path}")
        return md5_map
    
    for file_path in dir_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            try:
                md5 = calculate_md5(file_path)
                # 获取角色名（父目录名）
                character_name = file_path.parent.name
                md5_map[md5] = (file_path, character_name)
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {e}")
    
    return md5_map


def merge_directories(source_dirs: list, target_dir: Path, 
                     move_files: bool = False, dry_run: bool = False) -> dict:
    """
    合并多个目录到目标目录
    
    Args:
        source_dirs: 源目录列表
        target_dir: 目标目录
        move_files: 是否移动文件（False则复制）
        dry_run: 是否只模拟运行
        
    Returns:
        dict: 统计信息
    """
    stats = {
        'total_files': 0,
        'unique_files': 0,
        'duplicate_files': 0,
        'errors': 0,
        'by_character': defaultdict(int),
    }
    
    # 目标目录的MD5集合
    target_md5_map = {}
    
    # 创建目标目录
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"[模拟] 创建目标目录: {target_dir}")
    
    # 如果目标目录已有文件，先扫描
    if target_dir.exists():
        logger.info(f"扫描目标目录: {target_dir}")
        target_md5_map = scan_directory(target_dir)
        logger.info(f"目标目录已有 {len(target_md5_map)} 个唯一文件")
    
    # 处理每个源目录
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.warning(f"源目录不存在: {source_dir}")
            continue
        
        logger.info(f"扫描源目录: {source_dir}")
        source_md5_map = scan_directory(source_path)
        logger.info(f"找到 {len(source_md5_map)} 个文件")
        
        for md5, (file_path, character_name) in source_md5_map.items():
            stats['total_files'] += 1
            
            # 检查是否已存在
            if md5 in target_md5_map:
                stats['duplicate_files'] += 1
                logger.debug(f"重复文件: {file_path.name} (MD5: {md5[:8]}...)")
                continue
            
            # 新文件
            stats['unique_files'] += 1
            stats['by_character'][character_name] += 1
            
            # 创建角色目录
            char_dir = target_dir / character_name
            if not dry_run:
                char_dir.mkdir(parents=True, exist_ok=True)
            
            # 目标文件路径
            target_file = char_dir / file_path.name
            
            # 处理文件名冲突
            if target_file.exists():
                base_name = file_path.stem
                ext = file_path.suffix
                counter = 1
                while target_file.exists():
                    target_file = char_dir / f"{base_name}_{counter}{ext}"
                    counter += 1
            
            # 复制或移动文件
            try:
                if not dry_run:
                    if move_files:
                        shutil.move(str(file_path), str(target_file))
                    else:
                        shutil.copy2(str(file_path), str(target_file))
                    
                    # 更新MD5映射
                    target_md5_map[md5] = (target_file, character_name)
                else:
                    logger.info(f"[模拟] {'移动' if move_files else '复制'}: {file_path} -> {target_file}")
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {e}")
                stats['errors'] += 1
    
    return stats


def analyze_duplicates(dir1: Path, dir2: Path) -> dict:
    """
    分析两个目录之间的重复情况
    
    Returns:
        dict: 分析结果
    """
    logger.info(f"分析目录: {dir1}")
    map1 = scan_directory(dir1)
    logger.info(f"  文件数: {len(map1)}")
    
    logger.info(f"分析目录: {dir2}")
    map2 = scan_directory(dir2)
    logger.info(f"  文件数: {len(map2)}")
    
    # 找出重复的MD5
    md5_set1 = set(map1.keys())
    md5_set2 = set(map2.keys())
    duplicates = md5_set1 & md5_set2
    
    # 按角色统计
    by_character = defaultdict(lambda: {'dir1': 0, 'dir2': 0, 'duplicate': 0})
    
    for md5, (path, char) in map1.items():
        by_character[char]['dir1'] += 1
        if md5 in md5_set2:
            by_character[char]['duplicate'] += 1
    
    for md5, (path, char) in map2.items():
        by_character[char]['dir2'] += 1
    
    return {
        'dir1_total': len(map1),
        'dir2_total': len(map2),
        'duplicates': len(duplicates),
        'unique_in_dir1': len(md5_set1 - md5_set2),
        'unique_in_dir2': len(md5_set2 - md5_set1),
        'by_character': dict(by_character),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='图片目录合并工具')
    parser.add_argument('--source', type=str, nargs='+', required=True,
                        help='源目录列表')
    parser.add_argument('--target', type=str, required=True,
                        help='目标目录')
    parser.add_argument('--move', action='store_true',
                        help='移动文件而非复制')
    parser.add_argument('--dry-run', action='store_true',
                        help='模拟运行，不实际操作文件')
    parser.add_argument('--analyze', action='store_true',
                        help='只分析重复情况，不合并')
    
    args = parser.parse_args()
    
    source_dirs = [Path(p) for p in args.source]
    target_dir = Path(args.target)
    
    if args.analyze and len(source_dirs) == 2:
        # 分析模式
        result = analyze_duplicates(source_dirs[0], source_dirs[1])
        
        print("\n=== 重复分析结果 ===")
        print(f"目录1 ({source_dirs[0]}): {result['dir1_total']} 个文件")
        print(f"目录2 ({source_dirs[1]}): {result['dir2_total']} 个文件")
        print(f"重复文件: {result['duplicates']} 个")
        print(f"目录1独有: {result['unique_in_dir1']} 个")
        print(f"目录2独有: {result['unique_in_dir2']} 个")
        
        print("\n=== 按角色统计 ===")
        for char, counts in sorted(result['by_character'].items()):
            if counts['duplicate'] > 0:
                print(f"{char}: 目录1={counts['dir1']}, 目录2={counts['dir2']}, 重复={counts['duplicate']}")
        
        return
    
    # 合并模式
    logger.info(f"源目录: {source_dirs}")
    logger.info(f"目标目录: {target_dir}")
    logger.info(f"模式: {'移动' if args.move else '复制'}")
    
    stats = merge_directories(source_dirs, target_dir, args.move, args.dry_run)
    
    print("\n=== 合并统计 ===")
    print(f"总文件数: {stats['total_files']}")
    print(f"唯一文件: {stats['unique_files']}")
    print(f"重复文件: {stats['duplicate_files']}")
    print(f"错误数: {stats['errors']}")
    
    print("\n=== 按角色统计 ===")
    for char, count in sorted(stats['by_character'].items()):
        print(f"{char}: {count} 个新文件")


if __name__ == '__main__':
    main()
