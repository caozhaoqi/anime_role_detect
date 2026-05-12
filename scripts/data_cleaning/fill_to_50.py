#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将每个角色补充到50张图片
"""

import os
import shutil

TARGET_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'

def fill_to_50():
    print("=" * 70)
    print("📦 将每个角色补充到50张图片")
    print("=" * 70)
    
    roles = sorted([d for d in os.listdir(TARGET_DIR) if os.path.isdir(os.path.join(TARGET_DIR, d)) and not d.startswith('.')])
    total_added = 0
    
    for role in roles:
        role_path = os.path.join(TARGET_DIR, role)
        imgs = [f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))]
        current_count = len(imgs)
        
        if current_count >= 50:
            print(f"  {role}: {current_count} 张 (已达标)")
            continue
        
        need_count = 50 - current_count
        
        for i in range(need_count):
            # 循环复制现有图片
            src_idx = i % current_count
            src_img = os.path.join(role_path, imgs[src_idx])
            ext = os.path.splitext(imgs[src_idx])[1]
            tgt_img = os.path.join(role_path, f'copy_{i}{ext}')
            shutil.copy(src_img, tgt_img)
        
        total_added += need_count
        print(f"✅ {role}: {current_count} → {current_count + need_count} 张")
    
    print("\n" + "=" * 70)
    print(f"已补充 {total_added} 张图片")
    print("=" * 70)
    
    # 最终统计
    total_images = 0
    for role in roles:
        role_path = os.path.join(TARGET_DIR, role)
        count = len([f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])
        total_images += count
    
    print(f"\n📊 最终统计:")
    print(f"角色总数: {len(roles)}")
    print(f"图片总数: {total_images}")
    print(f"平均每角色: {total_images / len(roles):.1f} 张")
    print("🎉 所有角色均已达标！")

if __name__ == '__main__':
    fill_to_50()
