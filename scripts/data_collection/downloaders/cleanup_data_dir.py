#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
彻底清理 data 目录脚本
删除所有空目录、只有 .DS_Store 的目录
"""

import os
import logging

# 配置
DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data'

# 保留的目录（这些目录有用）
KEEP_DIRS = [
    'organized_images',
    'href_url',
    'img_url', 
    'auto_spider_img',
    'versions'
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_empty_or_only_dotds(dir_path):
    """检查目录是否为空或只有 .DS_Store 文件"""
    try:
        contents = os.listdir(dir_path)
        if len(contents) == 0:
            return True, "空目录"
        elif len(contents) == 1 and contents[0] == '.DS_Store':
            return True, "只有 .DS_Store"
        return False, f"有 {len(contents)} 个文件"
    except OSError:
        return False, "无法访问"


def clean_directory(root_dir):
    """清理目录"""
    deleted_count = 0
    skipped_count = 0
    
    # 先收集所有要删除的目录（从深到浅）
    to_delete = []
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # 跳过根目录
        if dirpath == root_dir:
            continue
        
        # 获取目录名
        dir_name = os.path.basename(dirpath)
        
        # 检查是否在保留列表中
        if dir_name in KEEP_DIRS:
            skipped_count += 1
            continue
        
        # 检查是否为空或只有 .DS_Store
        is_empty, reason = is_empty_or_only_dotds(dirpath)
        if is_empty:
            to_delete.append((dirpath, reason))
    
    # 删除目录
    for dir_path, reason in to_delete:
        try:
            # 删除 .DS_Store 文件（如果存在）
            ds_store = os.path.join(dir_path, '.DS_Store')
            if os.path.exists(ds_store):
                os.remove(ds_store)
            
            # 删除空目录
            os.rmdir(dir_path)
            logger.info(f"删除目录: {dir_path} ({reason})")
            deleted_count += 1
        except OSError as e:
            logger.warning(f"删除失败 {dir_path}: {str(e)}")
            skipped_count += 1
    
    return deleted_count, skipped_count


def main():
    """主函数"""
    logger.info(f"开始清理 {DATA_DIR} 目录...")
    
    deleted_count, skipped_count = clean_directory(DATA_DIR)
    
    logger.info(f"\n=== 清理完成 ===")
    logger.info(f"删除目录: {deleted_count} 个")
    logger.info(f"跳过目录: {skipped_count} 个")
    
    # 显示最终目录结构
    logger.info("\n=== 最终目录结构 ===")
    for entry in os.listdir(DATA_DIR):
        entry_path = os.path.join(DATA_DIR, entry)
        if os.path.isdir(entry_path):
            logger.info(f"  {entry}/")
        else:
            logger.info(f"  {entry}")


if __name__ == '__main__':
    main()
