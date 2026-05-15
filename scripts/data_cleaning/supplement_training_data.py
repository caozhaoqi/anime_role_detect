#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充训练数据集中不足的图片
从原数据集(combined_dataset)补充到训练数据集(training_dataset)
确保每个角色至少100张图片
"""
import os
import shutil
import random

# 配置
TRAINING_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset'
SOURCE_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
TARGET_COUNT = 100

def get_image_count(dir_path):
    """获取目录中的jpg图片数量"""
    if not os.path.exists(dir_path):
        return 0
    return len([f for f in os.listdir(dir_path) if f.lower().endswith('.jpg')])

def supplement_images():
    """补充图片"""
    print("🔍 检查并补充训练数据集")
    print("=" * 60)
    
    stats = {
        'total_roles': 0,
        'supplemented_roles': 0,
        'total_added': 0,
        'already_ok': 0,
        'missing_in_source': 0
    }
    
    for role in sorted(os.listdir(TRAINING_PATH)):
        train_dir = os.path.join(TRAINING_PATH, role)
        source_dir = os.path.join(SOURCE_PATH, role)
        
        if not os.path.isdir(train_dir):
            continue
        
        current_count = get_image_count(train_dir)
        source_count = get_image_count(source_dir)
        
        stats['total_roles'] += 1
        
        if current_count >= TARGET_COUNT:
            print(f"✅ {role}: {current_count} 张 (已满足)")
            stats['already_ok'] += 1
            continue
        
        if not os.path.exists(source_dir) or source_count == 0:
            print(f"❌ {role}: {current_count} 张 (原数据集无补充源)")
            stats['missing_in_source'] += 1
            continue
        
        # 需要补充的数量
        needed = TARGET_COUNT - current_count
        
        # 获取源目录中可用的图片（排除已在训练集中的）
        train_images = set(os.listdir(train_dir))
        source_images = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg') and f not in train_images]
        
        if len(source_images) == 0:
            print(f"❌ {role}: {current_count} 张 (源目录无新图片)")
            stats['missing_in_source'] += 1
            continue
        
        # 随机选择需要的数量
        to_copy = random.sample(source_images, min(needed, len(source_images)))
        
        # 复制图片
        for img in to_copy:
            src_path = os.path.join(source_dir, img)
            dst_path = os.path.join(train_dir, img)
            shutil.copy(src_path, dst_path)
        
        stats['supplemented_roles'] += 1
        stats['total_added'] += len(to_copy)
        print(f"🔄 {role}: {current_count} → {current_count + len(to_copy)} 张 (补充 {len(to_copy)} 张)")
    
    print("-" * 60)
    print(f"统计结果:")
    print(f"  总角色数: {stats['total_roles']}")
    print(f"  已满足: {stats['already_ok']}")
    print(f"  已补充: {stats['supplemented_roles']}")
    print(f"  补充图片数: {stats['total_added']}")
    print(f"  无法补充: {stats['missing_in_source']}")
    
    return stats

def verify_final_counts():
    """验证最终图片数量"""
    print("\n🔍 验证最终图片数量")
    print("=" * 60)
    
    total_images = 0
    insufficient_roles = []
    
    for role in sorted(os.listdir(TRAINING_PATH)):
        train_dir = os.path.join(TRAINING_PATH, role)
        if not os.path.isdir(train_dir):
            continue
        
        count = get_image_count(train_dir)
        total_images += count
        
        if count < TARGET_COUNT:
            print(f"⚠️ {role}: {count} 张 (不足)")
            insufficient_roles.append(role)
        else:
            print(f"✅ {role}: {count} 张")
    
    print("-" * 60)
    print(f"总计: {len(os.listdir(TRAINING_PATH))} 个角色, {total_images} 张图片")
    
    if insufficient_roles:
        print(f"\n⚠️ 仍有 {len(insufficient_roles)} 个角色图片不足:")
        print(f"  {', '.join(insufficient_roles)}")
    
    return insufficient_roles

if __name__ == '__main__':
    # 补充图片
    supplement_images()
    
    # 验证结果
    verify_final_counts()
    
    print("\n✅ 补充完成")