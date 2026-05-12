#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充不足50张图片的角色
"""

import os
import shutil

SOURCE_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
TARGET_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'

NEED_MORE = ['an1ke3', 'fu2xuan2', 'ke3li4', 'lei2mu3', 'na4xi1da2', 'xiao3mei3yan4', 'zhi4nai3']

def supplement_images():
    print("=" * 70)
    print("📦 开始补充图片")
    print("=" * 70)
    
    for role in NEED_MORE:
        src_path = os.path.join(SOURCE_DIR, role)
        tgt_path = os.path.join(TARGET_DIR, role)
        
        if not os.path.exists(src_path):
            print(f"❌ {role}: 源目录不存在")
            continue
        
        # 获取目标目录已有的图片
        tgt_imgs = set([f for f in os.listdir(tgt_path) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])
        tgt_count = len(tgt_imgs)
        
        # 从源目录找额外的图片
        src_imgs = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp')) and f not in tgt_imgs]
        
        if src_imgs:
            # 复制第一张额外的图片
            src_img = os.path.join(src_path, src_imgs[0])
            tgt_img = os.path.join(tgt_path, src_imgs[0])
            shutil.copy(src_img, tgt_img)
            print(f"✅ {role}: 补充1张 ({tgt_count} → {tgt_count+1})")
        else:
            print(f"❌ {role}: 没有额外图片可补充")
    
    # 最终统计
    print("\n" + "=" * 70)
    print("📊 最终统计")
    print("=" * 70)
    
    roles = [d for d in os.listdir(TARGET_DIR) if os.path.isdir(os.path.join(TARGET_DIR, d)) and not d.startswith('.')]
    total_images = 0
    under_50 = []
    
    for role in sorted(roles):
        role_path = os.path.join(TARGET_DIR, role)
        count = len([f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])
        total_images += count
        if count < 50:
            under_50.append((role, count))
    
    print(f"角色总数: {len(roles)}")
    print(f"图片总数: {total_images}")
    print(f"平均每角色: {total_images / len(roles):.2f} 张")
    
    if under_50:
        print(f"\n⚠️ 仍有 {len(under_50)} 个角色不足50张:")
        for role, count in under_50:
            print(f"  - {role}: {count} 张")
    else:
        print("\n🎉 所有角色均≥50张图片！")
    
    print("=" * 70)

if __name__ == '__main__':
    supplement_images()
