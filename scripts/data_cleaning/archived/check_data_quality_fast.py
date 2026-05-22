#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速数据质量检查脚本
"""

import os
import hashlib

DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/reorganized_dataset'

def fast_check():
    print("=" * 70)
    print("🔍 快速数据质量检查")
    print("=" * 70)
    
    all_hashes = {}
    duplicates = []
    suspicious_files = []
    
    for role in sorted(os.listdir(DATA_DIR)):
        role_dir = os.path.join(DATA_DIR, role)
        
        if not os.path.isdir(role_dir) or role.startswith('.'):
            continue
        
        for img_name in os.listdir(role_dir):
            if not img_name.lower().endswith(('.jpg', '.png', '.webp', '.jpeg', '.bmp')):
                continue
            
            img_path = os.path.join(role_dir, img_name)
            
            # 检查文件大小
            file_size = os.path.getsize(img_path)
            if file_size < 5 * 1024:  # 小于5KB
                suspicious_files.append((role, img_name, f"小文件: {file_size} bytes"))
            
            # 检查是否为复制文件
            if '_copy' in img_name or 'copy_' in img_name:
                suspicious_files.append((role, img_name, "复制文件"))
            
            # 计算哈希值检查重复
            try:
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                key = (role, file_hash)
                if key in all_hashes:
                    duplicates.append((role, all_hashes[key], img_name))
                else:
                    all_hashes[key] = img_name
            except Exception:
                suspicious_files.append((role, img_name, "读取失败"))
    
    print(f"\n📊 检查结果:")
    print(f"重复图片对: {len(duplicates)}")
    print(f"可疑文件: {len(suspicious_files)}")
    
    if duplicates:
        print("\n⚠️ 重复图片:")
        for role, orig, dup in duplicates[:5]:
            print(f"  {role}: {orig} ↔ {dup}")
        if len(duplicates) > 5:
            print(f"  ... 还有 {len(duplicates) - 5} 对")
    
    if suspicious_files:
        print("\n⚠️ 可疑文件:")
        for role, img, reason in suspicious_files[:5]:
            print(f"  {role}/{img}: {reason}")
        if len(suspicious_files) > 5:
            print(f"  ... 还有 {len(suspicious_files) - 5} 个")
    
    print("\n" + "=" * 70)
    if len(duplicates) == 0 and len(suspicious_files) == 0:
        print("🎉 数据质量检查通过！")
    else:
        print("⚠️ 发现潜在问题")

if __name__ == '__main__':
    fast_check()
