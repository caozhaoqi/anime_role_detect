#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备训练数据: 剔除小类别 + 分层 train/val 拆分
"""
import os
import shutil
import random
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

random.seed(42)

SOURCE_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/filtered_dataset")
TRAIN_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset_prepared/train")
VAL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset_prepared/val")
MIN_IMAGES = 5     # 剔除图片 ≤ 5 的角色
VAL_RATIO = 0.2    # 验证集比例
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_data():
    """收集角色名 → 图片路径列表的映射"""
    role_images = {}
    for char_dir in sorted(SOURCE_DIR.iterdir()):
        if not char_dir.is_dir() or char_dir.name.startswith("."):
            continue
        images = sorted(
            [str(f) for f in char_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        )
        if len(images) >= MIN_IMAGES:
            role_images[char_dir.name] = images
    return role_images


def stratified_split(role_images):
    """分层拆分 train/val，保证每个角色在两个集合中都有样本"""
    samples = []   # [(path, label_idx), ...]
    labels = []    # label_idx
    label_names = sorted(role_images.keys())
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    for name in label_names:
        for img_path in role_images[name]:
            samples.append((img_path, label_to_idx[name]))
            labels.append(label_to_idx[name])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=42)
    train_idx, val_idx = next(sss.split(samples, labels))

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples, label_names


def copy_samples(samples, target_dir, label_names):
    """按 label 复制图片到目标目录"""
    for img_path, label_idx in samples:
        role_name = label_names[label_idx]
        dst_dir = target_dir / role_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, str(dst_dir))


def print_stats(role_images, train_samples, val_samples, label_names):
    """打印详细的统计信息"""
    train_counter = Counter()
    val_counter = Counter()
    for path, idx in train_samples:
        train_counter[label_names[idx]] += 1
    for path, idx in val_samples:
        val_counter[label_names[idx]] += 1

    print(f"{'角色名':<20} {'总数':>6} {'训练':>6} {'验证':>6} {'验证占比':>8}")
    print("-" * 50)
    total_all = 0
    for name in label_names:
        total = len(role_images[name])
        tr = train_counter.get(name, 0)
        va = val_counter.get(name, 0)
        ratio = va / total * 100 if total > 0 else 0
        print(f"{name:<20} {total:>6} {tr:>6} {va:>6} {ratio:>7.1f}%")
        total_all += total
    print("-" * 50)
    print(f"{'总计':<20} {total_all:>6} {len(train_samples):>6} {len(val_samples):>6}")
    print(f"训练集: {len(train_samples)} 张, 验证集: {len(val_samples)} 张")


def main():
    print("=" * 70)
    print("准备训练数据: 剔除小类别 + 分层拆分")
    print("=" * 70)

    # 1. 收集数据
    role_images = collect_data()
    print(f"原始角色数: {len(list(SOURCE_DIR.iterdir()))} (过滤前)")
    print(f"保留角色数 (≥{MIN_IMAGES}张): {len(role_images)}")

    # 打印各角色图片数
    for name in sorted(role_images.keys()):
        print(f"  {name}: {len(role_images[name])} 张")

    # 2. 分层拆分
    print(f"\n分层拆分 (训练 {1-VAL_RATIO:.0%} / 验证 {VAL_RATIO:.0%})...")
    train_samples, val_samples, label_names = stratified_split(role_images)

    # 3. 统计信息
    print("\n各角色分配详情:")
    print_stats(role_images, train_samples, val_samples, label_names)

    # 4. 复制文件
    print(f"\n复制文件...")
    if TRAIN_DIR.exists():
        shutil.rmtree(str(TRAIN_DIR))
    if VAL_DIR.exists():
        shutil.rmtree(str(VAL_DIR))

    copy_samples(train_samples, TRAIN_DIR, label_names)
    copy_samples(val_samples, VAL_DIR, label_names)

    print(f"\n✅ 完成!")
    print(f"训练集: {TRAIN_DIR} ({len(train_samples)} 张)")
    print(f"验证集: {VAL_DIR} ({len(val_samples)} 张)")
    print(f"类别数: {len(label_names)}")
    print(f"总图片: {len(train_samples) + len(val_samples)} 张")


if __name__ == "__main__":
    main()