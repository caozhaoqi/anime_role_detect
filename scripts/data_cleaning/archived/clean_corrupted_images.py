#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清理脚本 - 检测并删除损坏的图片文件
"""

import os
from PIL import Image

DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'

def is_image_corrupted(img_path):
    """检测图片是否损坏"""
    try:
        with Image.open(img_path) as img:
            img.verify()
            # 尝试重新打开并转换为RGB，确保图片可以正常读取
            img = Image.open(img_path)
            img.convert('RGB')
            return False
    except Exception as e:
        return True

def clean_corrupted_images(data_dir):
    """清理所有损坏的图片"""
    corrupted_count = 0
    total_files = 0
    corrupted_files = []
    
    print("=" * 70)
    print("🔍 开始检测损坏图片")
    print("=" * 70)
    
    for role_dir in sorted(os.listdir(data_dir)):
        role_path = os.path.join(data_dir, role_dir)
        
        if not os.path.isdir(role_path) or role_dir.startswith('.'):
            continue
        
        for img_name in os.listdir(role_path):
            img_path = os.path.join(role_path, img_name)
            
            if not os.path.isfile(img_path):
                continue
            
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                continue
            
            total_files += 1
            
            if is_image_corrupted(img_path):
                corrupted_count += 1
                corrupted_files.append(img_path)
                print(f"❌ 损坏: {img_path}")
    
    print("\n" + "=" * 70)
    print(f"检测完成！")
    print(f"总文件数: {total_files}")
    print(f"损坏文件数: {corrupted_count}")
    print("=" * 70)
    
    if corrupted_files:
        print(f"\n🗑️ 正在删除 {corrupted_count} 个损坏文件...")
        
        for img_path in corrupted_files:
            try:
                os.remove(img_path)
                print(f"已删除: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"删除失败 {img_path}: {e}")
        
        print(f"\n✅ 成功删除 {corrupted_count} 个损坏文件")
    else:
        print("\n🎉 没有发现损坏的图片！")
    
    # 统计清理后的数据集
    print("\n" + "=" * 70)
    print("📊 清理后的数据集统计")
    print("=" * 70)
    
    total_after = 0
    role_count = 0
    
    for role_dir in sorted(os.listdir(data_dir)):
        role_path = os.path.join(data_dir, role_dir)
        
        if not os.path.isdir(role_path) or role_dir.startswith('.'):
            continue
        
        role_files = [f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
        count = len(role_files)
        total_after += count
        role_count += 1
        
        if count < 50:
            print(f"⚠️ {role_dir}: {count} 张")
        else:
            print(f"✅ {role_dir}: {count} 张")
    
    print("\n" + "=" * 70)
    print(f"角色总数: {role_count}")
    print(f"图片总数: {total_after}")
    print(f"平均每角色: {total_after / role_count:.2f} 张")
    print("=" * 70)

if __name__ == '__main__':
    clean_corrupted_images(DATA_DIR)
