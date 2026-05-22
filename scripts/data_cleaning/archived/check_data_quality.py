#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量检查脚本 - 检测重复和低质量图片
"""

import os
import hashlib
from PIL import Image

DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'

MIN_WIDTH = 128
MIN_HEIGHT = 128
MIN_SIZE_KB = 5

def get_image_hash(img_path):
    """计算图片的MD5哈希值"""
    try:
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def is_low_quality(img_path):
    """检查图片是否为低质量"""
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            # 检查尺寸
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return True, f"尺寸过小 ({width}x{height})"
            
            # 检查文件大小
            file_size_kb = os.path.getsize(img_path) / 1024
            if file_size_kb < MIN_SIZE_KB:
                return True, f"文件过小 ({file_size_kb:.1f}KB)"
            
            # 检查是否为纯色图片
            if img.mode == 'RGB':
                pixels = list(img.getdata())
                unique_colors = len(set(pixels))
                if unique_colors <= 10:
                    return True, f"纯色/低多样性图片 ({unique_colors}种颜色)"
            
            return False, ""
    except Exception as e:
        return True, f"无法读取: {str(e)}"

def check_duplicates(role_dir):
    """检查角色目录中的重复图片"""
    hashes = {}
    duplicates = []
    
    for img_name in os.listdir(role_dir):
        if not img_name.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp')):
            continue
        
        img_path = os.path.join(role_dir, img_name)
        img_hash = get_image_hash(img_path)
        
        if img_hash:
            if img_hash in hashes:
                duplicates.append((hashes[img_hash], img_name))
            else:
                hashes[img_hash] = img_name
    
    return duplicates

def check_data_quality():
    """检查整个数据集的质量"""
    print("=" * 70)
    print("🔍 数据质量检查")
    print("=" * 70)
    
    total_duplicates = 0
    total_low_quality = 0
    total_corrupted = 0
    duplicate_details = []
    low_quality_details = []
    
    roles = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')])
    
    for role in roles:
        role_dir = os.path.join(DATA_DIR, role)
        imgs = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp'))]
        
        # 检查重复
        duplicates = check_duplicates(role_dir)
        if duplicates:
            total_duplicates += len(duplicates)
            duplicate_details.extend([(role, orig, dup) for orig, dup in duplicates])
        
        # 检查低质量
        for img_name in imgs:
            img_path = os.path.join(role_dir, img_name)
            
            # 检查损坏
            try:
                with Image.open(img_path):
                    pass
            except Exception:
                total_corrupted += 1
                low_quality_details.append((role, img_name, "损坏"))
                continue
            
            # 检查质量
            is_low, reason = is_low_quality(img_path)
            if is_low:
                total_low_quality += 1
                low_quality_details.append((role, img_name, reason))
    
    print(f"\n📊 检查结果:")
    print(f"角色总数: {len(roles)}")
    print(f"重复图片: {total_duplicates} 对")
    print(f"低质量图片: {total_low_quality} 张")
    print(f"损坏图片: {total_corrupted} 张")
    
    if duplicate_details:
        print("\n⚠️ 重复图片详情:")
        for role, orig, dup in duplicate_details[:10]:
            print(f"  {role}: {orig} ↔ {dup}")
        if len(duplicate_details) > 10:
            print(f"  ... (还有 {len(duplicate_details) - 10} 对)")
    
    if low_quality_details:
        print("\n⚠️ 低质量图片详情:")
        for role, img, reason in low_quality_details[:10]:
            print(f"  {role}/{img}: {reason}")
        if len(low_quality_details) > 10:
            print(f"  ... (还有 {len(low_quality_details) - 10} 张)")
    
    print("\n" + "=" * 70)
    if total_duplicates == 0 and total_low_quality == 0 and total_corrupted == 0:
        print("🎉 数据质量检查通过！")
    else:
        print("⚠️ 发现数据质量问题，建议清理")
    
    return total_duplicates, total_low_quality, total_corrupted

if __name__ == '__main__':
    check_data_quality()
