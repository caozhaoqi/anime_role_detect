#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备训练数据：剔除不足角色，为每个角色抽取100张高质量图片
"""
import os
import shutil
import random
from pathlib import Path

# 配置
DATASET_PATH = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset"
TRAINING_PATH = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset"

# 需要剔除的角色
REMOVE_ROLES = ["Himesaka", "Hoshino"]

# 目标图片数量
TARGET_COUNT = 100


def clean_and_prepare():
    """清理并准备训练数据"""
    # 创建训练数据集目录
    os.makedirs(TRAINING_PATH, exist_ok=True)

    # 获取所有角色目录
    roles = []
    for d in os.listdir(DATASET_PATH):
        dp = os.path.join(DATASET_PATH, d)
        if os.path.isdir(dp) and not d.startswith(".") and ".json" not in d:
            roles.append(d)

    print("📊 开始准备训练数据")
    print("=" * 70)
    print(f"总角色数: {len(roles)}")
    print(f"需要剔除: {REMOVE_ROLES}")
    print("-" * 70)

    # 统计信息
    stats = {"total_roles": 0, "total_images": 0, "removed_roles": 0, "skipped_roles": 0}

    for role in sorted(roles):
        # 跳过需要剔除的角色
        if role in REMOVE_ROLES:
            print(f"❌ 剔除角色: {role}")
            stats["removed_roles"] += 1
            continue

        src_dir = os.path.join(DATASET_PATH, role)
        dst_dir = os.path.join(TRAINING_PATH, role)

        # 获取所有jpg图片
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(".jpg")]

        if len(images) < TARGET_COUNT:
            print(f"⚠️ 跳过角色 {role}: 图片不足({len(images)} < {TARGET_COUNT})")
            stats["skipped_roles"] += 1
            continue

        # 创建目标目录
        os.makedirs(dst_dir, exist_ok=True)

        # 随机抽取100张图片
        selected = random.sample(images, TARGET_COUNT)

        # 复制图片
        for img in selected:
            src_path = os.path.join(src_dir, img)
            dst_path = os.path.join(dst_dir, img)
            shutil.copy(src_path, dst_path)

        print(f"✅ {role}: 抽取 {len(selected)} 张图片")
        stats["total_roles"] += 1
        stats["total_images"] += len(selected)

    print("-" * 70)
    print("📊 训练数据准备完成")
    print(f"  保留角色数: {stats['total_roles']}")
    print(f"  剔除角色数: {stats['removed_roles']}")
    print(f"  跳过角色数: {stats['skipped_roles']}")
    print(f"  总图片数: {stats['total_images']}")

    return stats


def verify_training_data():
    """验证训练数据"""
    print("\n🔍 验证训练数据")
    print("=" * 70)

    roles = []
    total_images = 0

    for d in sorted(os.listdir(TRAINING_PATH)):
        dp = os.path.join(TRAINING_PATH, d)
        if os.path.isdir(dp) and not d.startswith("."):
            count = len([f for f in os.listdir(dp) if f.lower().endswith(".jpg")])
            roles.append((d, count))
            total_images += count
            print(f"{d:<20} {count} 张")

    print("-" * 70)
    print(f"总计: {len(roles)} 个角色, {total_images} 张图片")


if __name__ == "__main__":
    # 准备训练数据
    clean_and_prepare()

    # 验证数据
    verify_training_data()
