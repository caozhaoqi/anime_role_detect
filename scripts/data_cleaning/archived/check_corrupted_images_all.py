#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查整个数据集(combined_dataset)中的损坏图片
"""
import os
from PIL import Image

def check_and_remove_corrupted(dataset_path):
    """检查并删除损坏的图片"""
    corrupted_files = []
    total_files = 0
    
    for character in sorted(os.listdir(dataset_path)):
        char_dir = os.path.join(dataset_path, character)
        
        if not os.path.isdir(char_dir) or character.startswith('.'):
            continue
        
        for filename in os.listdir(char_dir):
            if filename.lower().endswith('.jpg'):
                file_path = os.path.join(char_dir, filename)
                total_files += 1
                
                try:
                    with Image.open(file_path) as img:
                        img.load()
                except Exception as e:
                    corrupted_files.append((character, filename))
    
    print(f"🔍 检查完成:")
    print(f"  总文件数: {total_files}")
    print(f"  损坏文件: {len(corrupted_files)}")
    
    if corrupted_files:
        print("\n删除损坏图片:")
        for char_name, filename in corrupted_files:
            file_path = os.path.join(dataset_path, char_name, filename)
            try:
                os.remove(file_path)
                print(f"  ✅ 删除: {char_name}/{filename}")
            except Exception as e:
                print(f"  ❌ 删除失败 {char_name}/{filename}: {e}")
    
    return corrupted_files

def main():
    """主函数"""
    DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
    
    print("🔍 检查 combined_dataset 中的损坏图片")
    print("=" * 60)
    
    corrupted = check_and_remove_corrupted(DATASET_PATH)
    
    print("-" * 60)
    print(f"已删除 {len(corrupted)} 个损坏图片")
    print("✅ 检查完成")

if __name__ == '__main__':
    main()