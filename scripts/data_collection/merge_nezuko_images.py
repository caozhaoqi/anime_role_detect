#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并祢豆子的图片到正确的文件夹"""
import os
import shutil

IMG_DIR = 'data/organized_images'
TARGET_DIR = os.path.join(IMG_DIR, 'mi3dou4zi5')
SOURCE_DIRS = [
    os.path.join(IMG_DIR, 'ni2dou4zi5'),
    os.path.join(IMG_DIR, 'Nezuko'),
]

# 创建目标目录
os.makedirs(TARGET_DIR, exist_ok=True)

# 从源目录移动图片
moved = 0
for src_dir in SOURCE_DIRS:
    if os.path.exists(src_dir):
        for filename in os.listdir(src_dir):
            src_path = os.path.join(src_dir, filename)
            if os.path.isfile(src_path):
                dst_path = os.path.join(TARGET_DIR, filename)
                # 避免同名冲突
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(TARGET_DIR, f'{base}_{counter}{ext}')
                    counter += 1
                shutil.move(src_path, dst_path)
                moved += 1
        print(f'从 {os.path.basename(src_dir)} 移动了 {len(os.listdir(src_dir))} 张图片')
        os.rmdir(src_dir)

# 统计最终数量
final_count = len([f for f in os.listdir(TARGET_DIR) if os.path.isfile(os.path.join(TARGET_DIR, f))])
print(f'\\n✅ 完成！共移动了 {moved} 张图片')
print(f'祢豆子 (mi3dou4zi5) 现在有 {final_count} 张图片')
