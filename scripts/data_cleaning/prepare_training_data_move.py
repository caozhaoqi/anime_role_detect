#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备训练数据：移动100张图片到训练集，原数据集保留剩余图片用于测试
"""
import os
import shutil
import random
from pathlib import Path

# 配置
DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
TRAINING_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset'

# 需要剔除的角色
REMOVE_ROLES = ['Himesaka', 'Hoshino']

# 目标图片数量
TARGET_COUNT = 100

def prepare_training_data():
    """准备训练数据（移动模式）"""
    # 创建训练数据集目录
    os.makedirs(TRAINING_PATH, exist_ok=True)
    
    print("📊 开始准备训练数据（移动模式）")
    print("=" * 70)
    
    stats = {
        'total_roles': 0,
        'total_moved': 0,
        'removed_roles': 0,
        'remaining_images': {}
    }
    
    for role in sorted(os.listdir(DATASET_PATH)):
        src_dir = os.path.join(DATASET_PATH, role)
        
        # 跳过非目录和隐藏文件
        if not os.path.isdir(src_dir) or role.startswith('.') or '.json' in role:
            continue
        
        # 跳过需要剔除的角色
        if role in REMOVE_ROLES:
            print(f"❌ 剔除角色: {role}")
            stats['removed_roles'] += 1
            continue
        
        # 获取所有jpg图片
        images = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpg')]
        total_count = len(images)
        
        if total_count < TARGET_COUNT:
            print(f"⚠️ 跳过角色 {role}: 图片不足({total_count} < {TARGET_COUNT})")
            continue
        
        # 创建目标目录
        dst_dir = os.path.join(TRAINING_PATH, role)
        os.makedirs(dst_dir, exist_ok=True)
        
        # 随机选择100张图片移动
        to_move = random.sample(images, TARGET_COUNT)
        
        # 移动图片
        for img in to_move:
            src_path = os.path.join(src_dir, img)
            dst_path = os.path.join(dst_dir, img)
            shutil.move(src_path, dst_path)
        
        # 统计剩余图片
        remaining = total_count - TARGET_COUNT
        stats['remaining_images'][role] = remaining
        stats['total_roles'] += 1
        stats['total_moved'] += TARGET_COUNT
        
        print(f"✅ {role}: 移动 {TARGET_COUNT} 张, 剩余 {remaining} 张")
    
    print("-" * 70)
    print("📊 训练数据准备完成")
    print(f"  保留角色数: {stats['total_roles']}")
    print(f"  剔除角色数: {stats['removed_roles']}")
    print(f"  移动图片数: {stats['total_moved']}")
    print(f"  原数据集剩余图片可用于测试")
    
    return stats

def verify_data():
    """验证数据分布"""
    print("\n🔍 数据分布验证")
    print("=" * 70)
    
    train_total = 0
    test_total = 0
    
    print("训练数据集:")
    for role in sorted(os.listdir(TRAINING_PATH)):
        dp = os.path.join(TRAINING_PATH, role)
        if os.path.isdir(dp):
            count = len([f for f in os.listdir(dp) if f.lower().endswith('.jpg')])
            train_total += count
            print(f"  {role:<20} {count} 张")
    
    print("\n原数据集（测试用）:")
    for role in sorted(os.listdir(DATASET_PATH)):
        dp = os.path.join(DATASET_PATH, role)
        if os.path.isdir(dp) and not role.startswith('.') and '.json' not in role:
            count = len([f for f in os.listdir(dp) if f.lower().endswith('.jpg')])
            test_total += count
            if count > 0:
                print(f"  {role:<20} {count} 张")
    
    print("-" * 70)
    print(f"训练集: {train_total} 张")
    print(f"测试集(剩余): {test_total} 张")

if __name__ == "__main__":
    # 先清空训练集（如果存在）
    if os.path.exists(TRAINING_PATH):
        shutil.rmtree(TRAINING_PATH)
    
    # 准备训练数据
    prepare_training_data()
    
    # 验证数据
    verify_data()