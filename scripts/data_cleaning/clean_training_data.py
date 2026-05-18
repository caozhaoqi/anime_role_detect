#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗 training_dataset 数据
1. 删除大文件（> 2MB）
2. 删除小文件（< 10KB）
3. 从 combined_dataset 补充不足100张的角色数据
"""
import os
import shutil
from pathlib import Path

TRAINING_DATASET = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset')
COMBINED_DATASET = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset')

def clean_large_files(max_size_kb=2000):
    """删除大文件"""
    deleted = []
    for img_file in TRAINING_DATASET.rglob('*.jpg'):
        size_kb = img_file.stat().st_size / 1024
        if size_kb > max_size_kb:
            print(f"🗑️ 删除大文件: {img_file.relative_to(TRAINING_DATASET)} ({size_kb:.1f} KB)")
            os.remove(img_file)
            deleted.append(img_file)
    return deleted

def clean_small_files(min_size_kb=10):
    """删除小文件"""
    deleted = []
    for img_file in TRAINING_DATASET.rglob('*.jpg'):
        size_kb = img_file.stat().st_size / 1024
        if size_kb < min_size_kb:
            print(f"🗑️ 删除小文件: {img_file.relative_to(TRAINING_DATASET)} ({size_kb:.1f} KB)")
            os.remove(img_file)
            deleted.append(img_file)
    return deleted

def supplement_data(target_count=100):
    """从 combined_dataset 补充数据"""
    supplemented = {}
    
    for role_dir in TRAINING_DATASET.iterdir():
        if not role_dir.is_dir() or role_dir.name.startswith('.'):
            continue
        
        role_name = role_dir.name
        current_count = len(list(role_dir.glob('*.jpg')))
        
        if current_count >= target_count:
            continue
        
        need_count = target_count - current_count
        print(f"\n🔄 补充 {role_name}: 当前 {current_count} 张，需要补充 {need_count} 张")
        
        # 从 combined_dataset 查找该角色的图片
        source_dir = COMBINED_DATASET / role_name
        if not source_dir.exists():
            print(f"   ⚠️  combined_dataset 中找不到 {role_name}")
            continue
        
        # 获取源目录中的图片
        source_images = list(source_dir.glob('*.jpg'))
        if not source_images:
            print(f"   ⚠️  {role_name} 在 combined_dataset 中也没有图片")
            continue
        
        # 获取目标目录中已有的文件名
        existing_files = set(f.name for f in role_dir.glob('*.jpg'))
        
        # 筛选可用的源图片（排除已存在的）
        available_images = [img for img in source_images if img.name not in existing_files]
        
        if len(available_images) == 0:
            print(f"   ⚠️  没有可用的补充图片")
            continue
        
        # 随机选择需要的数量
        if len(available_images) > need_count:
            import random
            available_images = random.sample(available_images, need_count)
        
        # 复制图片
        copied_count = 0
        for src_img in available_images:
            try:
                shutil.copy2(src_img, role_dir / src_img.name)
                copied_count += 1
            except Exception as e:
                print(f"   ❌ 复制失败 {src_img.name}: {e}")
        
        supplemented[role_name] = copied_count
        print(f"   ✅ 成功补充 {copied_count} 张图片")
    
    return supplemented

def verify_cleanup():
    """验证清洗结果"""
    print('\n' + '=' * 60)
    print('✅ 验证清洗结果')
    print('=' * 60)
    
    total_images = 0
    satisfied_count = 0
    zero_count = 0
    
    for role_dir in sorted(TRAINING_DATASET.iterdir()):
        if role_dir.is_dir() and not role_dir.name.startswith('.'):
            img_count = len(list(role_dir.glob('*.jpg')))
            total_images += img_count
            
            if img_count >= 100:
                satisfied_count += 1
            elif img_count == 0:
                zero_count += 1
    
    print(f"\n总角色数: {len(list(TRAINING_DATASET.iterdir())) - 1}")  # 减去 .DS_Store
    print(f"总图片数: {total_images}")
    print(f"满足100张的角色: {satisfied_count}")
    print(f"0张角色: {zero_count}")
    
    return satisfied_count, zero_count

def main():
    print('🚀 开始数据清洗\n')
    
    # 步骤1: 删除大文件
    print('=' * 60)
    print('步骤1: 删除大文件 (> 2MB)')
    print('=' * 60)
    deleted_large = clean_large_files()
    print(f"已删除 {len(deleted_large)} 个大文件")
    
    # 步骤2: 删除小文件
    print('\n' + '=' * 60)
    print('步骤2: 删除小文件 (< 10KB)')
    print('=' * 60)
    deleted_small = clean_small_files()
    print(f"已删除 {len(deleted_small)} 个小文件")
    
    # 步骤3: 补充数据
    print('\n' + '=' * 60)
    print('步骤3: 补充不足100张的角色数据')
    print('=' * 60)
    supplemented = supplement_data()
    
    # 步骤4: 验证结果
    verify_cleanup()
    
    print(f"\n🎉 清洗完成！")
    print(f"- 删除大文件: {len(deleted_large)} 个")
    print(f"- 删除小文件: {len(deleted_small)} 个")
    print(f"- 补充数据: {sum(supplemented.values())} 张")

if __name__ == '__main__':
    main()