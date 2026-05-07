#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用NSFW检测清理主目录中的敏感图片"""
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/Users/caozhaoqi/PycharmProjects/anime_role_detect')

from src.services.nsfw_detector import detect_nsfw

IMG_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images')
TRASH_DIR = IMG_DIR / 'trash_nsfw'

def cleanup_nsfw_images(nsfw_threshold=0.6, skin_ratio_threshold=0.4):
    print("=" * 60)
    print("🗑️ 使用NSFW检测清理敏感图片")
    print("=" * 60)
    print(f"NSFW得分阈值: {nsfw_threshold}")
    print(f"皮肤比例阈值: {skin_ratio_threshold}")

    TRASH_DIR.mkdir(exist_ok=True)
    
    deleted_count = 0
    nsfw_count = 0
    processed_count = 0
    
    for folder in IMG_DIR.iterdir():
        if not folder.is_dir() or folder.name in ['其他', 'trash', 'trash_nsfw']:
            continue
            
        folder_name = folder.name
        print(f"\n📁 处理目录: {folder_name}")
        
        for img_path in folder.glob('*'):
            if not img_path.is_file():
                continue
                
            processed_count += 1
            
            try:
                # 执行NSFW检测
                result = detect_nsfw(str(img_path))
                
                if result is None:
                    print(f"   ⚠️ 检测失败: {img_path.name}")
                    continue
                
                is_nsfw = result.get('is_nsfw', False)
                nsfw_score = result.get('nsfw_score', 0)
                skin_ratio = result.get('skin_ratio', 0)
                details = result.get('details', {})
                
                # 获取最高概率类别
                max_category = max(details, key=details.get) if details else 'unknown'
                max_score = details.get(max_category, 0)
                
                # 严格的删除条件：必须满足以下任一条件
                # 1. 模型明确标记为NSFW (is_nsfw=True)
                # 2. NSFW综合得分 > 阈值
                # 3. 皮肤比例过高且NSFW得分较高
                should_delete = False
                delete_reason = ""
                
                if is_nsfw:
                    should_delete = True
                    delete_reason = "模型标记"
                elif nsfw_score > nsfw_threshold:
                    should_delete = True
                    delete_reason = f"NSFW得分 {nsfw_score:.2f}"
                elif skin_ratio > skin_ratio_threshold and nsfw_score > 0.4:
                    should_delete = True
                    delete_reason = f"皮肤比例 {skin_ratio:.2f}"
                
                if should_delete:
                    nsfw_count += 1
                    
                    # 移动到垃圾目录
                    dest = TRASH_DIR / f"{folder_name}_{img_path.name}"
                    os.rename(img_path, dest)
                    deleted_count += 1
                    
                    print(f"   ❌ 删除 [{delete_reason}]: {img_path.name}")
                    print(f"      类别: {max_category} ({max_score:.2f}), NSFW得分: {nsfw_score:.2f}, 皮肤比例: {skin_ratio:.2f}")
                        
            except Exception as e:
                print(f"   ⚠️ 处理失败 {img_path.name}: {e}")

    print("\n" + "=" * 60)
    print("✅ NSFW清理完成!")
    print(f"处理图片数: {processed_count}")
    print(f"检测为NSFW: {nsfw_count}")
    print(f"删除图片数: {deleted_count}")
    print(f"删除的文件已移至: {TRASH_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    # 可以通过命令行参数设置阈值
    nsfw_threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    skin_ratio_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    cleanup_nsfw_images(nsfw_threshold, skin_ratio_threshold)