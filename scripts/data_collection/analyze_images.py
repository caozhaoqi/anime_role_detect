#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析主目录图片数据特征，为清理做准备"""
import os
from pathlib import Path
from PIL import Image
import math

IMG_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images')

def analyze_images():
    print("=" * 60)
    print("📊 分析主目录图片数据特征")
    print("=" * 60)

    total_images = 0
    small_files = []
    low_res = []
    suspicious_names = []
    
    for folder in IMG_DIR.iterdir():
        if not folder.is_dir() or folder.name == '其他':
            continue
            
        for img_path in folder.glob('*'):
            if not img_path.is_file():
                continue
                
            total_images += 1
            
            try:
                # 文件大小分析
                file_size = os.path.getsize(img_path)
                
                # 小于5KB的可能是低质量或缩略图
                if file_size < 5 * 1024:
                    small_files.append((img_path, file_size))
                
                # 图片分辨率分析
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                    # 低分辨率检测
                    if width < 200 or height < 200:
                        low_res.append((img_path, width, height))
                
                # 文件命名分析
                filename = img_path.name.lower()
                # 检测可能的问题文件命名
                suspicious_keywords = ['nsfw', 'hentai', 'porn', 'sex', 'nude', 'naked', 'erotic', 'censored']
                for keyword in suspicious_keywords:
                    if keyword in filename:
                        suspicious_names.append((img_path, keyword))
                        
            except Exception as e:
                print(f"⚠️ 无法分析 {img_path}: {e}")

    print(f"\n📋 总图片数: {total_images}")
    print(f"\n📏 小于5KB的小文件 ({len(small_files)}):")
    for img, size in sorted(small_files, key=lambda x: x[1])[:10]:
        print(f"   {size} bytes - {img.parent.name}/{img.name}")
    
    print(f"\n📐 低分辨率图片 ({len(low_res)}):")
    for img, w, h in sorted(low_res, key=lambda x: x[1]*x[2])[:10]:
        print(f"   {w}x{h} - {img.parent.name}/{img.name}")
    
    print(f"\n⚠️ 可疑命名文件 ({len(suspicious_names)}):")
    for img, keyword in suspicious_names[:10]:
        print(f"   [{keyword}] - {img.parent.name}/{img.name}")

    print("\n" + "=" * 60)
    print(f"分析完成！")
    print(f"需要关注的文件总数: {len(small_files) + len(low_res) + len(suspicious_names)}")
    print("=" * 60)

if __name__ == '__main__':
    analyze_images()