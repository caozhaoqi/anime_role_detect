#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据平衡脚本 - 按角色均衡采样
"""

import os
import shutil
import random
from collections import defaultdict


def balance_dataset(single_face_dir, cropped_dir, output_dir, max_per_role=100, min_per_role=20):
    """
    平衡数据集

    Args:
        single_face_dir: 单人脸图片目录
        cropped_dir: 切割后的单人样本目录
        output_dir: 输出目录
        max_per_role: 每个角色最多采样数量
        min_per_role: 最少需要的样本数
    """
    # 创建输出目录
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 收集所有图片
    all_images = defaultdict(list)

    # 收集单人脸图片
    for f in os.listdir(single_face_dir):
        if f.endswith(".jpg"):
            role = f.split("_")[0]
            all_images[role].append(os.path.join(single_face_dir, f))

    # 收集切割后的图片
    for f in os.listdir(cropped_dir):
        if f.endswith(".jpg"):
            role = f.split("_")[0]
            all_images[role].append(os.path.join(cropped_dir, f))

    # 过滤样本数不足的角色
    valid_roles = {role: imgs for role, imgs in all_images.items() if len(imgs) >= min_per_role}

    print(f"🎯 有效角色数: {len(valid_roles)} (总角色数: {len(all_images)})")

    # 统计角色样本数
    stats = []
    for role, imgs in valid_roles.items():
        stats.append((role, len(imgs)))

    # 按样本数排序
    stats.sort(key=lambda x: x[1], reverse=True)

    print("\n📊 角色样本分布:")
    for role, count in stats[:20]:
        print(f"  {role}: {count} 张")

    # 均衡采样
    total_train = 0
    total_val = 0
    role_counts = {}

    for role, imgs in valid_roles.items():
        # 随机打乱
        random.shuffle(imgs)

        # 采样最多 max_per_role 张
        selected = imgs[:max_per_role]

        # 划分训练集和验证集 (80:20)
        split_idx = int(len(selected) * 0.8)
        train_imgs = selected[:split_idx]
        val_imgs = selected[split_idx:]

        # 创建角色目录
        role_train_dir = os.path.join(train_dir, role)
        role_val_dir = os.path.join(val_dir, role)
        os.makedirs(role_train_dir, exist_ok=True)
        os.makedirs(role_val_dir, exist_ok=True)

        # 复制图片
        for img in train_imgs:
            shutil.copy(img, os.path.join(role_train_dir, os.path.basename(img)))

        for img in val_imgs:
            shutil.copy(img, os.path.join(role_val_dir, os.path.basename(img)))

        role_counts[role] = {"train": len(train_imgs), "val": len(val_imgs)}
        total_train += len(train_imgs)
        total_val += len(val_imgs)

    # 输出统计
    print("\n" + "=" * 60)
    print("📊 数据平衡统计")
    print("=" * 60)
    print(f"有效角色数: {len(valid_roles)}")
    print(f"训练集图片数: {total_train}")
    print(f"验证集图片数: {total_val}")
    print(f"总图片数: {total_train + total_val}")
    print(f"平均每角色训练样本: {total_train // len(valid_roles)}")
    print(f"平均每角色验证样本: {total_val // len(valid_roles)}")

    # 保存统计报告
    report_path = os.path.join(output_dir, "balance_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("数据平衡报告\n")
        f.write("=" * 40 + "\n")
        f.write(f"最大每角色样本数: {max_per_role}\n")
        f.write(f"最小每角色样本数: {min_per_role}\n")
        f.write("=" * 40 + "\n")
        f.write(f"有效角色数: {len(valid_roles)}\n")
        f.write(f"训练集图片数: {total_train}\n")
        f.write(f"验证集图片数: {total_val}\n")
        f.write("\n各角色样本数:\n")
        for role, counts in sorted(role_counts.items()):
            f.write(f"  {role}: train={counts['train']}, val={counts['val']}\n")

    print(f"\n✅ 报告已保存: {report_path}")
    return train_dir, val_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据平衡工具")
    parser.add_argument("--single_face", type=str, required=True, help="单人脸图片目录")
    parser.add_argument("--cropped", type=str, required=True, help="切割后的单人样本目录")
    parser.add_argument("--output", type=str, default="./balanced_dataset", help="输出目录")
    parser.add_argument("--max_per_role", type=int, default=100, help="每个角色最多采样数量")
    parser.add_argument("--min_per_role", type=int, default=20, help="最少需要的样本数")

    args = parser.parse_args()

    print("🚀 开始数据平衡")
    balance_dataset(
        args.single_face, args.cropped, args.output, args.max_per_role, args.min_per_role
    )
