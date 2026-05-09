#!/usr/bin/env python3
"""
创建每个角色20张图片的新目录
"""

import os
import shutil
from pathlib import Path
import random

def main():
    # 原目录和新目录
    source_dir = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images')
    target_dir = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_20_images')

    # 创建新目录
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 创建新目录: {target_dir}")

    # 排除的文件夹
    exclude = {'trash', 'trash_nsfw', 'trash_multi_face', '其他', '.DS_Store'}

    # 遍历所有角色文件夹
    role_folders = [f for f in source_dir.iterdir() if f.is_dir() and f.name not in exclude and not f.name.startswith('.')]
    
    print(f"\n开始整理 {len(role_folders)} 个角色...")

    total_images_copied = 0
    roles_with_less_than_20 = []

    for role_folder in sorted(role_folders):
        role_name = role_folder.name
        
        # 获取所有图片文件
        images = [f for f in role_folder.iterdir() 
                  if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        
        if not images:
            print(f"⚠️  {role_name}: 无图片")
            continue
        
        # 创建目标角色目录
        target_role_dir = target_dir / role_name
        target_role_dir.mkdir(exist_ok=True)
        
        # 确定要复制的图片数量（最多20张）
        count = min(len(images), 20)
        
        # 随机选择图片
        selected = random.sample(images, count)
        
        # 复制图片
        for img in selected:
            shutil.copy(str(img), str(target_role_dir / img.name))
        
        total_images_copied += count
        
        if count < 20:
            roles_with_less_than_20.append(f"{role_name} ({count}张)")
        
        print(f"✅ {role_name}: {count}张图片")

    print(f"\n🎉 整理完成！")
    print(f"📂 新目录: {target_dir}")
    print(f"👥 总角色数: {len(role_folders)}")
    print(f"🖼️  总图片数: {total_images_copied}")
    
    if roles_with_less_than_20:
        print(f"\n⚠️  以下角色图片不足20张:")
        for role in roles_with_less_than_20:
            print(f"  - {role}")

if __name__ == "__main__":
    main()
