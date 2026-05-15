#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查并清理数据集中的损坏图片
"""
import os
import sys
from PIL import Image

DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset'

def check_images():
    """检查数据集中的所有图片"""
    corrupted_files = []
    total_files = 0
    valid_files = 0
    
    for role in sorted(os.listdir(DATASET_PATH)):
        role_dir = os.path.join(DATASET_PATH, role)
        if not os.path.isdir(role_dir) or role.startswith('.'):
            continue
        
        for img_file in os.listdir(role_dir):
            img_path = os.path.join(role_dir, img_file)
            if not img_file.lower().endswith('.jpg'):
                continue
            
            total_files += 1
            
            try:
                # 尝试打开图片
                with Image.open(img_path) as img:
                    img.verify()
                # 再次打开确保可以读取
                with Image.open(img_path) as img:
                    img.load()
                valid_files += 1
            except Exception as e:
                corrupted_files.append((role, img_file, str(e)))
                print(f"❌ 损坏图片: {role}/{img_file} - {e}")
    
    print(f"\n检查完成:")
    print(f"  总文件数: {total_files}")
    print(f"  有效文件: {valid_files}")
    print(f"  损坏文件: {len(corrupted_files)}")
    
    return corrupted_files

def fix_corrupted_images(corrupted_files):
    """修复损坏的图片（删除它们）"""
    if not corrupted_files:
        print("没有损坏的图片需要修复")
        return
    
    print(f"\n删除 {len(corrupted_files)} 个损坏图片:")
    
    for role, img_file, _ in corrupted_files:
        img_path = os.path.join(DATASET_PATH, role, img_file)
        try:
            os.remove(img_path)
            print(f"✅ 删除: {role}/{img_file}")
        except Exception as e:
            print(f"❌ 删除失败: {role}/{img_file} - {e}")

def main():
    print("🔍 检查数据集中的损坏图片")
    print("=" * 60)
    
    corrupted_files = check_images()
    
    if corrupted_files:
        fix_corrupted_images(corrupted_files)
    
    print("\n✅ 检查和修复完成")

if __name__ == '__main__':
    main()