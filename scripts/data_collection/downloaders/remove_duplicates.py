#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重复图片清理脚本
删除原始目录中与 organized_images 重复的图片
"""

import os
import sys
import hashlib
import logging

# 配置
DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data'
ORGANIZED_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

# 需要清理的目录（按优先级排序）
DIRS_TO_CLEAN = [
    os.path.join(DATA_DIR, 'loli_roles'),
    os.path.join(DATA_DIR, 'loli_roles_cleaned'),
    os.path.join(DATA_DIR, 'downloaded_images'),
    os.path.join(DATA_DIR, 'role_images'),
    os.path.join(DATA_DIR, 'loli_training_data'),
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_file_hash(file_path):
    """计算文件的MD5哈希值"""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"计算哈希失败 {file_path}: {str(e)}")
        return None


def collect_organized_hashes():
    """收集已整理图片的哈希值"""
    logger.info("收集 organized_images 中的图片哈希...")
    hashes = set()
    for dirpath, dirnames, filenames in os.walk(ORGANIZED_DIR):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                file_hash = get_file_hash(os.path.join(dirpath, filename))
                if file_hash:
                    hashes.add(file_hash)
    logger.info(f"收集到 {len(hashes)} 个唯一哈希")
    return hashes


def clean_duplicates(organized_hashes):
    """清理重复图片"""
    stats = {
        'total_scanned': 0,
        'total_deleted': 0,
        'total_skipped': 0,
        'total_failed': 0,
        'by_dir': {}
    }
    
    for target_dir in DIRS_TO_CLEAN:
        if not os.path.exists(target_dir):
            logger.info(f"目录不存在: {target_dir}")
            continue
        
        dir_stats = {'scanned': 0, 'deleted': 0, 'skipped': 0, 'failed': 0}
        
        logger.info(f"\n处理目录: {target_dir}")
        
        for dirpath, dirnames, filenames in os.walk(target_dir):
            for filename in filenames:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    file_path = os.path.join(dirpath, filename)
                    dir_stats['scanned'] += 1
                    stats['total_scanned'] += 1
                    
                    file_hash = get_file_hash(file_path)
                    if not file_hash:
                        dir_stats['failed'] += 1
                        stats['total_failed'] += 1
                        continue
                    
                    if file_hash in organized_hashes:
                        try:
                            os.remove(file_path)
                            dir_stats['deleted'] += 1
                            stats['total_deleted'] += 1
                        except Exception as e:
                            logger.error(f"删除失败 {file_path}: {str(e)}")
                            dir_stats['failed'] += 1
                            stats['total_failed'] += 1
                    else:
                        dir_stats['skipped'] += 1
                        stats['total_skipped'] += 1
        
        stats['by_dir'][target_dir] = dir_stats
        logger.info(f"  扫描: {dir_stats['scanned']}, 删除: {dir_stats['deleted']}, 跳过: {dir_stats['skipped']}, 失败: {dir_stats['failed']}")
    
    return stats


def main():
    """主函数"""
    # 收集已整理图片的哈希
    organized_hashes = collect_organized_hashes()
    
    # 清理重复图片
    stats = clean_duplicates(organized_hashes)
    
    # 输出统计结果
    logger.info("\n=== 清理完成 ===")
    logger.info(f"总扫描: {stats['total_scanned']} 张")
    logger.info(f"总删除: {stats['total_deleted']} 张")
    logger.info(f"总跳过: {stats['total_skipped']} 张")
    logger.info(f"总失败: {stats['total_failed']} 张")
    
    # 计算节省的空间
    saved_space = stats['total_deleted'] * 200  # 平均每张图片约200KB
    if saved_space > 1024:
        logger.info(f"预计节省空间: {saved_space / 1024:.2f} MB")
    else:
        logger.info(f"预计节省空间: {saved_space} KB")


if __name__ == '__main__':
    main()
