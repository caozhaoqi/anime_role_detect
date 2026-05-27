#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 final_dataset：
1. 移除非图片文件（SVG等）
2. 移除损坏的图片
3. 基于内容哈希去重
4. 移除低质量图片（尺寸过小、宽高比异常、文件过小）
5. 限制每个角色的最大图片数量
"""

import os
import sys
import hashlib
import shutil
import logging
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# 配置
FINAL_DATASET = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset'
TRASH_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/trash_final'

# 清理参数
MIN_IMAGE_SIZE = 10 * 1024      # 最小文件大小 10KB
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 最大文件大小 10MB
MIN_WIDTH = 300                  # 最小宽度
MIN_HEIGHT = 300                 # 最小高度
MAX_ASPECT_RATIO = 3.0           # 最大宽高比
MIN_ASPECT_RATIO = 0.3           # 最小宽高比
MAX_IMAGES_PER_ROLE = 200        # 每个角色最大保留图片数
HASH_SIZE = (100, 100)           # 哈希计算时的图片缩放尺寸

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("clean_final_dataset")

def is_valid_image_file(filename):
    """检查是否为有效的图片文件扩展名"""
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif')
    return filename.lower().endswith(valid_exts)

def remove_non_images(role_dir):
    """移除非图片文件"""
    removed = 0
    for filename in os.listdir(role_dir):
        filepath = os.path.join(role_dir, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.svg':
                os.remove(filepath)
                removed += 1
                logger.debug(f"移除SVG文件: {filename}")
            elif not is_valid_image_file(filename):
                os.remove(filepath)
                removed += 1
                logger.debug(f"移除非图片文件: {filename}")
    return removed

def check_image_quality(img_path):
    """
    检查图片质量，返回 (是否合格, 原因)
    """
    try:
        file_size = os.path.getsize(img_path)

        if file_size < MIN_IMAGE_SIZE:
            return False, f"文件过小 ({file_size} bytes)"

        if file_size > MAX_IMAGE_SIZE:
            return False, f"文件过大 ({file_size} bytes)"

        with Image.open(img_path) as img:
            width, height = img.size
            ratio = width / height

            if ratio > MAX_ASPECT_RATIO or ratio < MIN_ASPECT_RATIO:
                return False, f"宽高比异常 {width}x{height} ({ratio:.2f})"

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return False, f"尺寸过小 {width}x{height}"

        return True, "合格"

    except Exception as e:
        return False, f"损坏: {str(e)}"

def calculate_image_hash(img_path):
    """计算图片内容哈希（基于缩放后的灰度图）"""
    try:
        with Image.open(img_path) as img:
            img = img.resize(HASH_SIZE)
            img = img.convert('L')
            hash_obj = hashlib.md5()
            hash_obj.update(img.tobytes())
            return hash_obj.hexdigest()
    except Exception as e:
        logger.warning(f"哈希计算失败: {img_path} - {e}")
        return None

def clean_role(role_dir, role_name):
    """清理单个角色的图片"""
    stats = {
        'non_images_removed': 0,
        'low_quality_removed': 0,
        'duplicates_removed': 0,
        'kept': 0,
        'total_original': 0
    }

    # 1. 移除非图片文件
    stats['non_images_removed'] = remove_non_images(role_dir)

    # 收集所有图片
    images = []
    for filename in os.listdir(role_dir):
        filepath = os.path.join(role_dir, filename)
        if os.path.isfile(filepath) and is_valid_image_file(filename):
            images.append((filename, filepath))

    stats['total_original'] = len(images)

    # 2. 检查低质量图片
    low_quality = []
    good_images = []

    for filename, filepath in images:
        is_good, reason = check_image_quality(filepath)
        if is_good:
            good_images.append((filename, filepath))
        else:
            low_quality.append((filename, filepath, reason))

    stats['low_quality_removed'] = len(low_quality)

    # 3. 去重（基于内容哈希）
    seen_hashes = {}
    duplicates = []
    unique_images = []

    for filename, filepath in good_images:
        img_hash = calculate_image_hash(filepath)
        if img_hash is None:
            duplicates.append((filename, filepath, "哈希计算失败"))
            continue

        if img_hash in seen_hashes:
            duplicates.append((filename, filepath, f"与 {seen_hashes[img_hash]} 重复"))
        else:
            seen_hashes[img_hash] = filename
            unique_images.append((filename, filepath))

    stats['duplicates_removed'] = len(duplicates)

    # 4. 限制每个角色的图片数量（保留质量最好的）
    final_images = unique_images

    if len(final_images) > MAX_IMAGES_PER_ROLE:
        final_images.sort(key=lambda x: os.path.getsize(x[1]), reverse=True)
        removed_excess = final_images[MAX_IMAGES_PER_ROLE:]
        final_images = final_images[:MAX_IMAGES_PER_ROLE]
        stats['duplicates_removed'] += len(removed_excess)

        for filename, filepath in removed_excess:
            try:
                os.remove(filepath)
            except Exception as e:
                logger.warning(f"删除多余图片失败: {filepath} - {e}")

    stats['kept'] = len(final_images)

    # 5. 执行删除操作
    all_to_remove = low_quality + duplicates

    for filename, filepath, reason in all_to_remove:
        try:
            os.remove(filepath)
            logger.debug(f"删除 [{role_name}]: {filename} - {reason}")
        except Exception as e:
            logger.warning(f"删除失败: {filepath} - {e}")

    return stats

def clean_dataset():
    """清理整个数据集"""
    logger.info("="*70)
    logger.info("🧹 开始清理 final_dataset")
    logger.info("="*70)
    logger.info(f"数据目录: {FINAL_DATASET}")
    logger.info(f"清理参数:")
    logger.info(f"  - 最小文件大小: {MIN_IMAGE_SIZE/1024:.0f}KB")
    logger.info(f"  - 最大文件大小: {MAX_IMAGE_SIZE/1024/1024:.0f}MB")
    logger.info(f"  - 最小尺寸: {MIN_WIDTH}x{MIN_HEIGHT}")
    logger.info(f"  - 宽高比范围: {MIN_ASPECT_RATIO} ~ {MAX_ASPECT_RATIO}")
    logger.info(f"  - 每角色最大保留: {MAX_IMAGES_PER_ROLE} 张")
    logger.info("="*70)

    os.makedirs(TRASH_DIR, exist_ok=True)

    role_dirs = sorted([d for d in os.listdir(FINAL_DATASET)
                       if os.path.isdir(os.path.join(FINAL_DATASET, d))])

    total_stats = {
        'roles': 0,
        'total_original': 0,
        'non_images_removed': 0,
        'low_quality_removed': 0,
        'duplicates_removed': 0,
        'kept': 0
    }

    for role_name in role_dirs:
        role_dir = os.path.join(FINAL_DATASET, role_name)
        stats = clean_role(role_dir, role_name)

        total_stats['roles'] += 1
        total_stats['total_original'] += stats['total_original']
        total_stats['non_images_removed'] += stats['non_images_removed']
        total_stats['low_quality_removed'] += stats['low_quality_removed']
        total_stats['duplicates_removed'] += stats['duplicates_removed']
        total_stats['kept'] += stats['kept']

        if stats['non_images_removed'] > 0 or stats['low_quality_removed'] > 0 or stats['duplicates_removed'] > 0:
            logger.info(f"✅ {role_name}: 保留 {stats['kept']} 张 "
                       f"(移除: 非图片 {stats['non_images_removed']}, "
                       f"低质量 {stats['low_quality_removed']}, "
                       f"重复 {stats['duplicates_removed']})")
        else:
            logger.info(f"  {role_name}: 保留 {stats['kept']} 张 (无需清理)")

    # 输出统计
    logger.info("\n" + "="*70)
    logger.info("📊 清理完成统计")
    logger.info("="*70)
    logger.info(f"处理角色数: {total_stats['roles']} 个")
    logger.info(f"原始图片总数: {total_stats['total_original']} 张")
    logger.info(f"移除非图片文件: {total_stats['non_images_removed']} 张")
    logger.info(f"移除低质量图片: {total_stats['low_quality_removed']} 张")
    logger.info(f"移除重复图片: {total_stats['duplicates_removed']} 张")
    logger.info(f"最终保留图片: {total_stats['kept']} 张")
    logger.info(f"总移除: {total_stats['non_images_removed'] + total_stats['low_quality_removed'] + total_stats['duplicates_removed']} 张")
    if total_stats['total_original'] > 0:
        logger.info(f"保留率: {total_stats['kept'] / total_stats['total_original'] * 100:.1f}%")
    logger.info("="*70)

    # 统计每个角色的最终图片数
    final_counts = []
    for role_name in sorted(os.listdir(FINAL_DATASET)):
        role_dir = os.path.join(FINAL_DATASET, role_name)
        if os.path.isdir(role_dir):
            count = len([f for f in os.listdir(role_dir)
                        if os.path.isfile(os.path.join(role_dir, f)) and is_valid_image_file(f)])
            final_counts.append((role_name, count))

    if final_counts:
        logger.info("\n📁 各角色最终图片数量 (Top 10):")
        for role_name, count in sorted(final_counts, key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {role_name}: {count} 张")

        low_count_roles = [r for r, c in final_counts if c < 30]
        if low_count_roles:
            logger.info(f"\n⚠️ 图片少于30张的角色: {len(low_count_roles)} 个")

    return total_stats

if __name__ == "__main__":
    clean_dataset()
