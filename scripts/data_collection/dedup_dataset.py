#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局数据集去重脚本：
扫描 data/final_dataset 下所有图片，按 sha256 内容去重。
可以检测跨角色目录的重复图片（如多人合照存在于多个角色目录下）。

Usage:
    python3 scripts/data_collection/dedup_dataset.py                    # 仅报告
    python3 scripts/data_collection/dedup_dataset.py --delete           # 报告 + 删除重复
    python3 scripts/data_collection/dedup_dataset.py --move-to ./dupes  # 报告 + 移动到目录
"""

import os
import sys
import argparse
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def get_image_files(root_dir: str) -> List[Path]:
    """递归获取所有图片文件"""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = []
    root = Path(root_dir)
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in extensions:
            images.append(f)
    return images


def compute_sha256(file_path: Path) -> str:
    """计算文件 sha256 哈希"""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # 大文件分块读取
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def scan_dataset(root_dir: str) -> Tuple[Dict[str, List[Path]], int, int, int]:
    """
    扫描数据集，返回按 hash 分组的文件列表。
    Returns: (hash_to_files, total_files, unique_files, duplicate_files)
    """
    images = get_image_files(root_dir)
    total = len(images)
    print(f"扫描到 {total} 张图片...")

    hash_to_files: Dict[str, List[Path]] = defaultdict(list)

    for i, img_path in enumerate(images, 1):
        if i % 200 == 0 or i == total:
            print(f"  进度: {i}/{total}")
        try:
            h = compute_sha256(img_path)
            hash_to_files[h].append(img_path)
        except Exception as e:
            print(f"  ⚠️ 跳过 {img_path}: {e}")

    unique = len(hash_to_files)
    duplicate = total - unique
    return hash_to_files, total, unique, duplicate


def print_report(hash_to_files: Dict[str, List[Path]], total: int, unique: int, duplicate: int):
    """打印去重报告"""
    print("\n" + "=" * 60)
    print("  全局去重报告")
    print("=" * 60)
    print(f"  总图片数:    {total}")
    print(f"  唯一图片数:  {unique}")
    print(f"  重复图片数:  {duplicate}")
    print(f"  重复率:      {duplicate/total*100:.2f}%" if total > 0 else "  N/A")
    print("=" * 60)

    # 按角色统计
    role_stats = defaultdict(lambda: {"total": 0, "unique": 0, "dupes": 0})
    for h, files in hash_to_files.items():
        roles = set()
        for f in files:
            role = f.parent.name
            role_stats[role]["total"] += 1
            roles.add(role)
        if len(files) > 1:
            # 每个重复文件对应的角色各计一次
            for f in files:
                role = f.parent.name
                role_stats[role]["dupes"] += 1
        else:
            role = files[0].parent.name
            role_stats[role]["unique"] += 1

    # 打印重复详情
    dupes = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    if dupes:
        print(f"\n发现 {len(dupes)} 组重复:")
        print("-" * 60)
        # 按重复数量排序
        sorted_dupes = sorted(dupes.items(), key=lambda x: -len(x[1]))
        for h, files in sorted_dupes:
            roles = [f.parent.name for f in files]
            print(f"  sha256: {h[:12]}...  (重复 {len(files)} 次)")
            for f in files:
                print(f"    {f}")
            # 跨角色标记
            unique_roles = set(roles)
            if len(unique_roles) > 1:
                print(f"    ⚠️ 跨角色重复: {', '.join(sorted(unique_roles))}")
            print()

    # 按角色的重复统计
    print("\n按角色重复统计:")
    print(f"  {'角色':<20} {'总数':>6} {'唯一':>6} {'重复':>6} {'重复率':>8}")
    print("  " + "-" * 50)
    for role in sorted(role_stats.keys()):
        s = role_stats[role]
        rate = s["dupes"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {role:<20} {s['total']:>6} {s['unique']:>6} {s['dupes']:>6} {rate:>7.1f}%")

    return dupes


def handle_duplicates(dupes: Dict[str, List[Path]], args):
    """处理重复文件"""
    if not args.delete and not args.move_to:
        return

    removed = 0

    for h, files in dupes.items():
        # 按字符数排序文件名，留最长的（最有信息量的）
        sorted_files = sorted(files, key=lambda f: (len(f.name), f.name), reverse=True)
        keeper = sorted_files[0]  # 保留第一个（最长的文件名）
        to_remove = sorted_files[1:]

        for f in to_remove:
            if args.delete:
                os.remove(f)
                print(f"  🗑️ 删除: {f}")
                removed += 1
            elif args.move_to:
                dest_dir = Path(args.move_to)
                dest_dir.mkdir(parents=True, exist_ok=True)
                # 保留目录结构
                rel_path = f.relative_to(Path(args.move_to).parent.parent if args.move_to else "")
                # 简化: 直接按角色目录存放
                role_dest = dest_dir / f.parent.name
                role_dest.mkdir(exist_ok=True)
                shutil.move(str(f), str(role_dest / f.name))
                print(f"  📦 移动: {f} → {role_dest / f.name}")
                removed += 1

    print(f"\n共处理 {removed} 个重复文件")


def main():
    parser = argparse.ArgumentParser(description="全局数据集去重")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="数据集根目录 (默认: data/final_dataset)")
    parser.add_argument("--delete", action="store_true",
                        help="删除重复文件 (保留每个 hash 的第一份)")
    parser.add_argument("--move-to", type=str, default=None,
                        help="将重复文件移动到指定目录 (保留角色子目录结构)")
    args = parser.parse_args()

    # 确定数据集路径
    if args.data_dir:
        root_dir = args.data_dir
    else:
        root_dir = str(Path(__file__).parent.parent.parent / "data" / "final_dataset")

    if not os.path.isdir(root_dir):
        print(f"❌ 目录不存在: {root_dir}")
        sys.exit(1)

    print(f"📂 扫描目录: {root_dir}")

    hash_to_files, total, unique, duplicate = scan_dataset(root_dir)
    dupes = print_report(hash_to_files, total, unique, duplicate)

    handle_duplicates(dupes, args)

    # 保存报告
    report_path = Path(root_dir).parent / "deduplication_report.txt"
    with open(report_path, "w") as f:
        f.write(f"全局去重报告\n")
        f.write(f"{'='*60}\n")
        f.write(f"扫描目录: {root_dir}\n")
        f.write(f"总图片数: {total}\n")
        f.write(f"唯一图片数: {unique}\n")
        f.write(f"重复图片数: {duplicate}\n")
        f.write(f"重复率: {duplicate/total*100:.2f}%\n" if total > 0 else "N/A\n")

    print(f"\n📄 报告已保存: {report_path}")


if __name__ == "__main__":
    main()