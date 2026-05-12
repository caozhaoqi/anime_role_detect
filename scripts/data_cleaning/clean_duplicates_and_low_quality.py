#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理重复和低质量图片 - 保留高质量原始图片，删除复制和低质量图片
"""

import os
import hashlib
from PIL import Image

DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'
MIN_KEEP = 50  # 每个角色至少保留50张

def get_image_hash(img_path):
    """计算图片哈希值"""
    try:
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def is_copy_file(img_name):
    """判断是否为复制文件"""
    return '_copy' in img_name or 'copy_' in img_name

def clean_duplicates():
    print("=" * 70)
    print("🗑️ 清理重复和低质量图片")
    print("=" * 70)
    
    total_deleted = 0
    total_kept = 0
    
    for role in sorted(os.listdir(DATA_DIR)):
        role_dir = os.path.join(DATA_DIR, role)
        
        if not os.path.isdir(role_dir) or role.startswith('.'):
            continue
        
        # 收集所有图片
        imgs = []
        for img_name in os.listdir(role_dir):
            if img_name.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp')):
                img_path = os.path.join(role_dir, img_name)
                imgs.append((img_name, img_path))
        
        # 按是否为复制文件分组
        original_imgs = []
        copy_imgs = []
        
        for img_name, img_path in imgs:
            if is_copy_file(img_name):
                copy_imgs.append((img_name, img_path))
            else:
                original_imgs.append((img_name, img_path))
        
        # 计算需要删除的复制文件数量
        current_total = len(original_imgs) + len(copy_imgs)
        need_delete = max(0, current_total - MIN_KEEP)
        
        # 优先删除复制文件
        deleted_count = 0
        for img_name, img_path in copy_imgs[:need_delete]:
            try:
                os.remove(img_path)
                deleted_count += 1
            except Exception:
                pass
        
        # 如果还需要删除，删除一些原始文件中的重复
        if deleted_count < need_delete:
            remaining_delete = need_delete - deleted_count
            
            # 找出原始文件中的重复
            hashes = {}
            dup_imgs = []
            for img_name, img_path in original_imgs:
                img_hash = get_image_hash(img_path)
                if img_hash:
                    if img_hash in hashes:
                        dup_imgs.append((img_name, img_path))
                    else:
                        hashes[img_hash] = img_name
            
            for img_name, img_path in dup_imgs[:remaining_delete]:
                try:
                    os.remove(img_path)
                    deleted_count += 1
                except Exception:
                    pass
        
        total_deleted += deleted_count
        kept_count = len(original_imgs) + len(copy_imgs) - deleted_count
        total_kept += kept_count
        
        if deleted_count > 0:
            print(f"✅ {role}: 删除 {deleted_count} 张, 保留 {kept_count} 张")
        else:
            print(f"  {role}: 无需清理, 保留 {kept_count} 张")
    
    print("\n" + "=" * 70)
    print(f"已删除: {total_deleted} 张")
    print(f"保留: {total_kept} 张")
    print("=" * 70)
    
    # 最终统计
    roles = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')]
    total_images = 0
    for role in roles:
        role_dir = os.path.join(DATA_DIR, role)
        count = len([f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))])
        total_images += count
    
    print(f"\n📊 最终统计:")
    print(f"角色总数: {len(roles)}")
    print(f"图片总数: {total_images}")
    print(f"平均每角色: {total_images / len(roles):.1f} 张")

if __name__ == '__main__':
    clean_duplicates()
