#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备训练数据 - 筛选图片数量大于30张的角色
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict

# 配置
SOURCE_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
TRAINING_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
MIN_IMAGES = 30  # 最少图片数量阈值

# 创建训练目录
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# 统计每个角色的图片数
character_stats = defaultdict(int)

for char_dir in SOURCE_DIR.iterdir():
    if char_dir.is_dir():
        char_name = char_dir.name
        # 统计图片数量
        img_count = len(
            [
                f
                for f in char_dir.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
            ]
        )
        character_stats[char_name] = img_count

# 筛选出图片数量大于阈值的角色
qualified_roles = [(name, count) for name, count in character_stats.items() if count >= MIN_IMAGES]
qualified_roles.sort(key=lambda x: x[1], reverse=True)

print("=" * 70)
print("准备训练数据")
print("=" * 70)
print(f"筛选条件: 图片数量 >= {MIN_IMAGES} 张")
print(f"源目录: {SOURCE_DIR}")
print(f"目标目录: {TRAINING_DIR}")
print("=" * 70)

print(f"\n发现 {len(qualified_roles)} 个角色符合条件:")
print(f"{'角色名':<20} {'图片数':<10} {'状态':<10}")
print("-" * 70)

# 移动符合条件的角色目录
moved_count = 0
total_images = 0

for char_name, img_count in qualified_roles:
    source_path = SOURCE_DIR / char_name
    target_path = TRAINING_DIR / char_name

    # 如果目标目录已存在，先删除
    if target_path.exists():
        shutil.rmtree(target_path)

    # 移动整个目录（比复制快）
    shutil.move(str(source_path), str(target_path))

    moved_count += 1
    total_images += img_count

    print(f"{char_name:<20} {img_count:<10} ✅ 已移动")

print("-" * 70)
print(f"\n总结:")
print(f"  已移动角色数: {moved_count}")
print(f"  总图片数: {total_images}")
print(f"  平均每个角色: {total_images/moved_count:.1f} 张")
print("=" * 70)

# 输出训练数据统计
print("\n训练数据统计:")
print(f"  数据集路径: {TRAINING_DIR}")
print(f"  类别数: {moved_count}")
print(f"  图片总数: {total_images}")
print("\n可以开始训练了！")
print("运行命令:")
print("  python3 /Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/model_training/train_model.py")