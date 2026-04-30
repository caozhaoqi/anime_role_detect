#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片整理脚本 - 将分散的图片统一分类管理
"""

import os
import sys
import shutil
import hashlib
import logging

# 配置
DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data'
OUTPUT_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.gif')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_all_images(root_dir):
    """查找所有图片文件"""
    images = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(dirpath, filename))
    return images


def parse_role_name_from_path(file_path):
    """从路径中解析角色名"""
    # 尝试从目录名提取角色名
    dirname = os.path.dirname(file_path)
    basename = os.path.basename(dirname)
    
    # 处理不同格式的目录名
    # 格式1: 角色名_作品名 (如: 阿洛娜_蔚蓝档案)
    if '_' in basename:
        parts = basename.split('_')
        # 返回角色名（第一个部分）
        return parts[0]
    
    # 格式2: 纯拼音 (如: a1luo4na4)
    # 格式3: 中文角色名 (如: 阿洛娜)
    return basename


def organize_images():
    """整理所有图片"""
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 查找所有图片
    logger.info("正在搜索所有图片文件...")
    all_images = find_all_images(DATA_DIR)
    logger.info(f"找到 {len(all_images)} 个图片文件")
    
    # 统计信息
    stats = {
        'total_found': len(all_images),
        'total_copied': 0,
        'total_skipped': 0,
        'total_failed': 0,
        'roles': {}
    }
    
    # 用于去重的哈希集合
    seen_hashes = set()
    
    # 遍历所有图片
    for image_path in all_images:
        try:
            # 计算文件哈希
            file_hash = get_file_hash(image_path)
            
            # 去重检查
            if file_hash in seen_hashes:
                stats['total_skipped'] += 1
                continue
            seen_hashes.add(file_hash)
            
            # 解析角色名
            role_name = parse_role_name_from_path(image_path)
            
            # 创建角色目录
            role_dir = os.path.join(OUTPUT_DIR, role_name)
            os.makedirs(role_dir, exist_ok=True)
            
            # 复制文件
            filename = os.path.basename(image_path)
            # 添加哈希值避免文件名冲突
            name_without_ext, ext = os.path.splitext(filename)
            new_filename = f"{name_without_ext}_{file_hash}{ext}"
            dest_path = os.path.join(role_dir, new_filename)
            
            shutil.copy2(image_path, dest_path)
            stats['total_copied'] += 1
            
            # 更新角色统计
            if role_name not in stats['roles']:
                stats['roles'][role_name] = 0
            stats['roles'][role_name] += 1
            
            if stats['total_copied'] % 100 == 0:
                logger.info(f"已处理 {stats['total_copied']}/{len(all_images)} 张图片")
                
        except Exception as e:
            logger.error(f"处理文件失败 {image_path}: {str(e)}")
            stats['total_failed'] += 1
    
    # 输出统计结果
    logger.info("\n=== 整理完成 ===")
    logger.info(f"发现图片: {stats['total_found']} 张")
    logger.info(f"成功复制: {stats['total_copied']} 张")
    logger.info(f"重复跳过: {stats['total_skipped']} 张")
    logger.info(f"处理失败: {stats['total_failed']} 张")
    
    logger.info("\n=== 角色分类统计 ===")
    sorted_roles = sorted(stats['roles'].items(), key=lambda x: x[1], reverse=True)
    for role_name, count in sorted_roles:
        logger.info(f"  {role_name}: {count} 张")
    
    # 保存整理清单
    manifest_path = os.path.join(OUTPUT_DIR, 'manifest.txt')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(f"整理时间: {__import__('datetime').datetime.now()}\n")
        f.write(f"总图片数: {stats['total_copied']}\n")
        f.write(f"角色数: {len(stats['roles'])}\n")
        f.write("\n=== 角色清单 ===\n")
        for role_name, count in sorted_roles:
            f.write(f"{role_name}: {count} 张\n")
    
    logger.info(f"\n整理清单已保存: {manifest_path}")


if __name__ == '__main__':
    organize_images()
