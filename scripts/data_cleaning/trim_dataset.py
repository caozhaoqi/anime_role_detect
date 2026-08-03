#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集裁剪脚本

按项目规范裁剪数据集：
- training_dataset: 每角色最多 50 张（训练集标准 30-50）
- final_dataset: 每角色最多 150 张（上限）
- 删除空目录

Usage:
    python3 scripts/data_cleaning/trim_dataset.py --dry-run          # 预览
    python3 scripts/data_cleaning/trim_dataset.py                     # 执行
    python3 scripts/data_cleaning/trim_dataset.py --dataset training  # 只处理训练集
    python3 scripts/data_cleaning/trim_dataset.py --max 100           # 自定义上限
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_DIR = PROJECT_ROOT / "data" / "training_dataset"
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}


def scan_dataset(dataset_dir: Path) -> Dict[str, List[Path]]:
    """扫描数据集，返回 {角色名: [图片路径列表]}"""
    if not dataset_dir.exists():
        return {}

    result = {}
    for role_dir in sorted(dataset_dir.iterdir()):
        if not role_dir.is_dir():
            continue
        images = sorted([
            f for f in role_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ])
        result[role_dir.name] = images
    return result


def trim_role(role_name: str, images: List[Path], max_count: int, dry_run: bool = True) -> int:
    """裁剪单个角色的图片，返回删除数量"""
    if len(images) <= max_count:
        return 0

    # 随机选择要保留的图片，确保均匀分布
    random.shuffle(images)
    to_delete = images[max_count:]
    deleted = 0

    for f in to_delete:
        if dry_run:
            print(f"     [dry-run] 删除: {f}")
        else:
            try:
                f.unlink()
            except OSError as e:
                print(f"     删除失败: {f} - {e}")
                continue
        deleted += 1

    return deleted


def remove_empty_dirs(dataset_dir: Path, dry_run: bool = True) -> int:
    """删除空目录，返回删除数量"""
    if not dataset_dir.exists():
        return 0

    removed = 0
    for role_dir in sorted(dataset_dir.iterdir()):
        if not role_dir.is_dir():
            continue
        images = [f for f in role_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if not images:
            if dry_run:
                print(f"  [dry-run] 删除空目录: {role_dir.name}")
            else:
                try:
                    role_dir.rmdir()
                    print(f"  删除空目录: {role_dir.name}")
                except OSError as e:
                    print(f"  删除目录失败: {role_dir.name} - {e}")
                    continue
            removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(description="数据集裁剪脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际删除")
    parser.add_argument("--dataset", type=str, choices=["training", "final", "all"],
                        default="all", help="处理的数据集")
    parser.add_argument("--training-max", type=int, default=50, help="训练集最大图片数")
    parser.add_argument("--final-max", type=int, default=150, help="最终集最大图片数")
    args = parser.parse_args()

    mode = "预览 (dry-run)" if args.dry_run else "执行"
    print(f"数据集裁剪 — {mode}")
    print(f"训练集上限: {args.training_max} 张/角色")
    print(f"最终集上限: {args.final_max} 张/角色")
    print("=" * 60)

    total_deleted = 0
    total_empty_dirs = 0

    # ── 处理训练集 ──
    if args.dataset in ("training", "all"):
        print(f"\n{'=' * 60}")
        print(f"处理训练集: {TRAINING_DIR}")
        data = scan_dataset(TRAINING_DIR)
        print(f"共 {len(data)} 个角色, {sum(len(v) for v in data.values())} 张图片")

        # 裁剪超限角色
        for role_name, images in sorted(data.items(), key=lambda x: -len(x[1])):
            count = len(images)
            if count > args.training_max:
                print(f"  {role_name}: {count} → {args.training_max} (删除 {count - args.training_max})")
                deleted = trim_role(role_name, images, args.training_max, args.dry_run)
                total_deleted += deleted

        # 删除空目录
        empty = remove_empty_dirs(TRAINING_DIR, args.dry_run)
        total_empty_dirs += empty

    # ── 处理最终集 ──
    if args.dataset in ("final", "all"):
        print(f"\n{'=' * 60}")
        print(f"处理最终集: {FINAL_DIR}")
        data = scan_dataset(FINAL_DIR)
        print(f"共 {len(data)} 个角色, {sum(len(v) for v in data.values())} 张图片")

        # 裁剪超限角色
        for role_name, images in sorted(data.items(), key=lambda x: -len(x[1])):
            count = len(images)
            if count > args.final_max:
                print(f"  {role_name}: {count} → {args.final_max} (删除 {count - args.final_max})")
                deleted = trim_role(role_name, images, args.final_max, args.dry_run)
                total_deleted += deleted

        # 删除空目录
        empty = remove_empty_dirs(FINAL_DIR, args.dry_run)
        total_empty_dirs += empty

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    print(f"汇总 ({mode}):")
    print(f"  删除图片: {total_deleted} 张")
    print(f"  删除空目录: {total_empty_dirs} 个")

    if args.dry_run:
        print(f"\n执行命令: python3 scripts/data_cleaning/trim_dataset.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()