#!/usr/bin/env python3
"""移动已下载的图片到organized_images目录"""
import os
import shutil
import sys
sys.path.insert(0, 'spider_image_system/src/run')
from constants import PINYIN_MAPPING

# 需要移动的角色
LOW_COUNT_ROLES = [
    '芙丽希娅',
    '洛茜', 
    '克萝萝',
    '德丽莎'
]

def move_images(role_name):
    """移动单个角色的图片"""
    pinyin = PINYIN_MAPPING.get(role_name)
    if not pinyin:
        print(f"❌ 未找到 {role_name} 的拼音映射")
        return 0
    
    # 源目录（爬虫下载位置）
    src_dir = f'spider_image_system/src/run/data/downloaded_images/{pinyin}'
    # 目标目录
    dst_dir = f'data/organized_images/{pinyin}'
    
    if not os.path.exists(src_dir):
        print(f"❌ {role_name} 的源目录不存在: {src_dir}")
        return 0
    
    os.makedirs(dst_dir, exist_ok=True)
    
    # 获取所有图片文件
    images = []
    for ext in ['*.jpg', '*.png', '*.webp']:
        images.extend([f for f in os.listdir(src_dir) if f.lower().endswith(ext[1:])])
    
    if not images:
        print(f"❌ {role_name} 的源目录中没有图片")
        return 0
    
    moved = 0
    for img in images:
        src_path = os.path.join(src_dir, img)
        dst_path = os.path.join(dst_dir, img)
        
        # 如果目标文件已存在，重命名
        counter = 1
        while os.path.exists(dst_path):
            name, ext = os.path.splitext(img)
            dst_path = os.path.join(dst_dir, f"{name}_{counter}{ext}")
            counter += 1
        
        shutil.move(src_path, dst_path)
        moved += 1
    
    print(f"✅ {role_name}: 成功移动 {moved} 张图片")
    return moved

def main():
    print("=" * 60)
    print("📦 移动已下载的图片")
    print("=" * 60)
    
    total_moved = 0
    for role in LOW_COUNT_ROLES:
        print(f"\n📋 {role}")
        moved = move_images(role)
        total_moved += moved
    
    print(f"\n📊 共移动 {total_moved} 张图片")
    print("=" * 60)
    
    # 显示最终统计
    print("\n📈 最终图片统计:")
    for role in LOW_COUNT_ROLES:
        pinyin = PINYIN_MAPPING.get(role)
        if pinyin:
            dst_dir = f'data/organized_images/{pinyin}'
            count = len([f for f in os.listdir(dst_dir) if f.lower().endswith(('.jpg', '.png', '.webp'))])
            print(f"   {role}: {count}张")

if __name__ == '__main__':
    main()
