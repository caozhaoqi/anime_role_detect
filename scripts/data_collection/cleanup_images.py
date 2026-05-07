#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理主目录中的低质量图片"""
import os
from pathlib import Path
from PIL import Image

IMG_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images')
TRASH_DIR = IMG_DIR / 'trash'

def cleanup_images():
    print("=" * 60)
    print("🗑️ 清理主目录低质量图片")
    print("=" * 60)

    TRASH_DIR.mkdir(exist_ok=True)
    
    deleted_count = 0
    small_files_deleted = 0
    low_res_deleted = 0
    invalid_deleted = 0
    
    for folder in IMG_DIR.iterdir():
        if not folder.is_dir() or folder.name in ['其他', 'trash']:
            continue
            
        for img_path in folder.glob('*'):
            if not img_path.is_file():
                continue
                
            try:
                # 检查是否为有效图片
                try:
                    img = Image.open(img_path)
                    img.verify()
                    img.close()
                except:
                    # 无效图片，删除
                    dest = TRASH_DIR / f"{folder.name}_{img_path.name}"
                    os.rename(img_path, dest)
                    print(f"❌ 无效图片: {folder.name}/{img_path.name}")
                    invalid_deleted += 1
                    deleted_count += 1
                    continue
                
                # 文件大小分析 - 删除小于5KB的
                file_size = os.path.getsize(img_path)
                if file_size < 5 * 1024:
                    dest = TRASH_DIR / f"{folder.name}_{img_path.name}"
                    os.rename(img_path, dest)
                    print(f"📏 小文件 ({file_size} bytes): {folder.name}/{img_path.name}")
                    small_files_deleted += 1
                    deleted_count += 1
                    continue
                
                # 图片分辨率分析 - 删除低分辨率图片
                with Image.open(img_path) as img:
                    width, height = img.size
                    if width < 200 or height < 200:
                        dest = TRASH_DIR / f"{folder.name}_{img_path.name}"
                        os.rename(img_path, dest)
                        print(f"📐 低分辨率 ({width}x{height}): {folder.name}/{img_path.name}")
                        low_res_deleted += 1
                        deleted_count += 1
                        continue
                        
            except Exception as e:
                print(f"⚠️ 处理失败 {img_path}: {e}")

    print("\n" + "=" * 60)
    print("✅ 清理完成!")
    print(f"总删除: {deleted_count}")
    print(f"  - 无效图片: {invalid_deleted}")
    print(f"  - 小文件: {small_files_deleted}")
    print(f"  - 低分辨率: {low_res_deleted}")
    print(f"删除的文件已移至: {TRASH_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    cleanup_images()