#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片MD5索引工具
生成和更新图片目录的MD5索引，用于爬虫去重
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Set
from loguru import logger


def calculate_md5(file_path: Path) -> str:
    """计算文件MD5值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def build_md5_index(image_dir: Path, index_file: Path = None) -> Dict:
    """
    构建图片目录的MD5索引
    
    Args:
        image_dir: 图片目录
        index_file: 索引文件路径（默认为 image_dir/.md5_index.json）
        
    Returns:
        Dict: 索引数据
    """
    if index_file is None:
        index_file = image_dir / '.md5_index.json'
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    
    # 加载已有索引
    existing_index = {}
    if index_file.exists():
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                existing_index = json.load(f)
            logger.info(f"加载已有索引: {len(existing_index)} 条记录")
        except Exception as e:
            logger.warning(f"加载索引失败: {e}")
    
    # 扫描目录
    md5_index = {}
    stats = {
        'total': 0,
        'new': 0,
        'unchanged': 0,
        'by_character': {},
    }
    
    for file_path in image_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            stats['total'] += 1
            
            # 获取相对路径
            rel_path = str(file_path.relative_to(image_dir))
            
            # 检查是否已索引且文件未修改
            if rel_path in existing_index:
                file_stat = file_path.stat()
                cached = existing_index[rel_path]
                if cached.get('size') == file_stat.st_size and cached.get('mtime') == file_stat.st_mtime:
                    md5_index[rel_path] = cached
                    stats['unchanged'] += 1
                    continue
            
            # 计算MD5
            try:
                md5 = calculate_md5(file_path)
                file_stat = file_path.stat()
                
                md5_index[rel_path] = {
                    'md5': md5,
                    'size': file_stat.st_size,
                    'mtime': file_stat.st_mtime,
                }
                stats['new'] += 1
                
                # 按角色统计
                character = file_path.parent.name
                stats['by_character'][character] = stats['by_character'].get(character, 0) + 1
                
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {e}")
    
    # 保存索引
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(md5_index, f, indent=2, ensure_ascii=False)
    
    logger.success(f"索引已保存: {index_file}")
    
    return {
        'index': md5_index,
        'stats': stats,
    }


def load_md5_set(index_file: Path) -> Set[str]:
    """
    从索引文件加载MD5集合
    
    Args:
        index_file: 索引文件路径
        
    Returns:
        Set[str]: MD5集合
    """
    if not index_file.exists():
        return set()
    
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        return {v['md5'] for v in index.values() if 'md5' in v}
    except Exception as e:
        logger.error(f"加载索引失败: {e}")
        return set()


def get_character_image_count(image_dir: Path) -> Dict[str, int]:
    """
    统计每个角色的图片数量
    
    Args:
        image_dir: 图片目录
        
    Returns:
        Dict[str, int]: {角色名: 图片数量}
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    counts = {}
    
    for char_dir in image_dir.iterdir():
        if char_dir.is_dir():
            count = sum(1 for f in char_dir.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_extensions)
            if count > 0:
                counts[char_dir.name] = count
    
    return counts


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='图片MD5索引工具')
    parser.add_argument('--image-dir', type=str, 
                        default='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/merged_images',
                        help='图片目录路径')
    parser.add_argument('--index-file', type=str, default=None,
                        help='索引文件路径（默认为 image_dir/.md5_index.json）')
    parser.add_argument('--stats', action='store_true',
                        help='只显示统计信息')
    
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir)
    index_file = Path(args.index_file) if args.index_file else image_dir / '.md5_index.json'
    
    if not image_dir.exists():
        logger.error(f"图片目录不存在: {image_dir}")
        return
    
    if args.stats:
        # 只显示统计
        counts = get_character_image_count(image_dir)
        total = sum(counts.values())
        
        print(f"\n=== 图片统计 ===")
        print(f"总图片数: {total}")
        print(f"角色数: {len(counts)}")
        print(f"\n=== 按角色统计 ===")
        for char, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{char}: {count}")
        return
    
    # 构建索引
    logger.info(f"构建MD5索引: {image_dir}")
    result = build_md5_index(image_dir, index_file)
    
    print(f"\n=== 索引统计 ===")
    print(f"总文件数: {result['stats']['total']}")
    print(f"新索引: {result['stats']['new']}")
    print(f"未变化: {result['stats']['unchanged']}")
    
    # 显示角色统计
    if result['stats']['by_character']:
        print(f"\n=== 新增文件按角色统计 ===")
        for char, count in sorted(result['stats']['by_character'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10]:
            print(f"{char}: {count}")


if __name__ == '__main__':
    main()
