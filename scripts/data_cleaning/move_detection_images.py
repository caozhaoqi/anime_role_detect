#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将检测结果目录中的图片移动到对应角色目录
"""

import os
import shutil
import re

# 配置路径
BASE_DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data'
COMBINED_DATASET = os.path.join(BASE_DATA_DIR, 'combined_dataset')
YUNET_RESULTS = os.path.join(BASE_DATA_DIR, 'yunet_detection_results')
MULTI_FACE_DIR = os.path.join(BASE_DATA_DIR, 'multi_face_detected_anime')


def extract_role_name(filename):
    """从文件名中提取角色名"""
    # 格式1: face_2_Klee_Klee_0232.jpg -> Klee
    if filename.startswith('face_'):
        parts = filename.split('_')[2:]
        if len(parts) >= 2:
            # 可能是 face_2_Klee_Klee_0232.jpg 或 face_2_Klee_0232.jpg
            if parts[0] == parts[1]:
                return parts[0]
            else:
                # 尝试组合前几个部分
                return '_'.join(parts[:-1]) if len(parts) > 2 else parts[0]
        return filename.split('_')[2]
    
    # 格式2: Klee_Klee_0232.jpg -> Klee
    parts = filename.split('_')
    if len(parts) >= 3:
        if parts[0] == parts[1]:
            return parts[0]
        else:
            # 可能是 multi_name 如 Aris_wei4lan2dang4an4
            return '_'.join(parts[:-1])
    elif len(parts) == 2:
        return parts[0]
    
    # 格式3: Klee_0232.jpg -> Klee
    return filename.split('_')[0].split('.')[0]


def move_images_from_dir(source_dir, target_base_dir):
    """从源目录移动所有图片到目标目录"""
    moved_count = 0
    skipped_count = 0
    not_found_count = 0
    
    if not os.path.exists(source_dir):
        print(f"❌ 目录不存在: {source_dir}")
        return moved_count, skipped_count, not_found_count
    
    for filename in os.listdir(source_dir):
        if not (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.webp')):
            continue
        
        # 提取角色名
        role_name = extract_role_name(filename)
        
        # 创建目标目录
        target_dir = os.path.join(target_base_dir, role_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # 源路径和目标路径
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        # 如果目标已存在，跳过
        if os.path.exists(target_path):
            skipped_count += 1
            continue
        
        # 移动文件
        shutil.move(source_path, target_path)
        moved_count += 1
        print(f"✅ {source_path} -> {target_path}")
    
    return moved_count, skipped_count, not_found_count


def main():
    """主函数"""
    print("=" * 60)
    print("📦 开始移动检测结果图片到角色目录")
    print("=" * 60)
    
    # 创建目标目录
    os.makedirs(COMBINED_DATASET, exist_ok=True)
    
    total_moved = 0
    total_skipped = 0
    
    # 处理 multi_face_detected_anime/multi_face
    print("\n--- 处理 multi_face_detected_anime/multi_face ---")
    multi_face_dir = os.path.join(MULTI_FACE_DIR, 'multi_face')
    moved, skipped, _ = move_images_from_dir(multi_face_dir, COMBINED_DATASET)
    total_moved += moved
    total_skipped += skipped
    print(f"  移动: {moved} | 跳过(已存在): {skipped}")
    
    # 处理 multi_face_detected_anime/grayscale_multi_face
    print("\n--- 处理 multi_face_detected_anime/grayscale_multi_face ---")
    grayscale_dir = os.path.join(MULTI_FACE_DIR, 'grayscale_multi_face')
    moved, skipped, _ = move_images_from_dir(grayscale_dir, COMBINED_DATASET)
    total_moved += moved
    total_skipped += skipped
    print(f"  移动: {moved} | 跳过(已存在): {skipped}")
    
    # 处理 yunet_detection_results/multi_role
    print("\n--- 处理 yunet_detection_results/multi_role ---")
    multi_role_dir = os.path.join(YUNET_RESULTS, 'multi_role')
    moved, skipped, _ = move_images_from_dir(multi_role_dir, COMBINED_DATASET)
    total_moved += moved
    total_skipped += skipped
    print(f"  移动: {moved} | 跳过(已存在): {skipped}")
    
    # 处理 yunet_detection_results/no_role
    print("\n--- 处理 yunet_detection_results/no_role ---")
    no_role_dir = os.path.join(YUNET_RESULTS, 'no_role')
    moved, skipped, _ = move_images_from_dir(no_role_dir, COMBINED_DATASET)
    total_moved += moved
    total_skipped += skipped
    print(f"  移动: {moved} | 跳过(已存在): {skipped}")
    
    # 处理 yunet_detection_results/single_role
    print("\n--- 处理 yunet_detection_results/single_role ---")
    single_role_dir = os.path.join(YUNET_RESULTS, 'single_role')
    moved, skipped, _ = move_images_from_dir(single_role_dir, COMBINED_DATASET)
    total_moved += moved
    total_skipped += skipped
    print(f"  移动: {moved} | 跳过(已存在): {skipped}")
    
    print("\n" + "=" * 60)
    print(f"📊 移动完成")
    print(f"  - 成功移动: {total_moved} 个文件")
    print(f"  - 跳过(已存在): {total_skipped} 个文件")
    print("=" * 60)


if __name__ == '__main__':
    main()