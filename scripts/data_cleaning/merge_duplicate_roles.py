#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 final_dataset 中的重复角色目录

问题：旧数据使用短名（Hina, mika, shiroko），新采集使用完整 Danbooru 标签
      （sorasaki_hina, misono_mika, sunaookami_shiroko），导致同一角色两个目录

Usage:
    python3 scripts/data_cleaning/merge_duplicate_roles.py --dry-run   # 预览
    python3 scripts/data_cleaning/merge_duplicate_roles.py             # 执行
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

# 重复映射：短名 → 全名（保留全名目录）
DUPLICATE_MAP = {
    "mika": "misono_mika",
    "Hina": "sorasaki_hina",
    "Hifumi": "ajitani_hifumi",
    "Izuna": "kuda_izuna",
    "shiroko": "sunaookami_shiroko",
    "nodoka": "amami_nodoka",
    "Kayoko": "onikata_kayoko",
    "Serika": "kuromi_serika",
    "nonomi": "izayoi_nonomi",
    "Haruna": "kurodate_haruna",
    "Ayane": "okusora_ayane",
    "Azusa": "shirasu_azusa",
    "hoshino": "takanashi_hoshino",
    "mutsuki": "asagi_mutsuki",
    "wakamo": "kosaka_wakamo",
    "koharu": "shimoe_koharu",
    "Anya": "anya_forger",
}


def count_images(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    return sum(1 for f in dir_path.iterdir()
               if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def merge_roles(short_name: str, full_name: str, dry_run: bool = True) -> Tuple[int, int]:
    """合并两个目录的图片到 full_name 目录"""
    short_dir = FINAL_DIR / short_name
    full_dir = FINAL_DIR / full_name

    if not short_dir.exists():
        return 0, 0

    short_count = count_images(short_dir)
    full_count = count_images(full_dir)

    if short_count == 0:
        if dry_run:
            print(f"  [dry-run] {short_name}(空目录) → 删除")
        else:
            short_dir.rmdir()
        return 0, full_count

    os.makedirs(full_dir, exist_ok=True)

    moved = 0
    for f in short_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            dest = full_dir / f.name
            # 如果目标已存在同名文件，加后缀
            if dest.exists():
                dest = full_dir / f"{f.stem}_dup{f.suffix}"
            if dry_run:
                moved += 1
            else:
                shutil.move(str(f), str(dest))
                moved += 1

    if dry_run:
        print(f"  [dry-run] {short_name}({short_count}张) → {full_name}({full_count}张) = {short_count + full_count} 张")
    else:
        print(f"  {short_name}({short_count}张) → {full_name}({full_count}张) = {short_count + full_count} 张")
        # 删除空目录
        try:
            short_dir.rmdir()
        except OSError:
            # 目录非空（可能有非图片文件），跳过
            pass

    return moved, full_count + moved


def main():
    parser = argparse.ArgumentParser(description="合并重复角色目录")
    parser.add_argument("--dry-run", action="store_true", help="仅预览")
    args = parser.parse_args()

    mode = "预览" if args.dry_run else "执行"
    print(f"合并重复角色目录 — {mode}")
    print(f"数据集: {FINAL_DIR}")
    print("=" * 60)

    total_moved = 0
    total_merged = 0

    for short_name, full_name in DUPLICATE_MAP.items():
        short_dir = FINAL_DIR / short_name
        if not short_dir.exists():
            continue

        short_count = count_images(short_dir)
        full_count = count_images(FINAL_DIR / full_name)

        if short_count == 0 and full_count == 0:
            continue

        moved, new_total = merge_roles(short_name, full_name, args.dry_run)
        total_moved += moved
        total_merged += 1

    print(f"\n{'=' * 60}")
    print(f"汇总 ({mode}):")
    print(f"  合并组数: {total_merged}")
    print(f"  移动图片: {total_moved} 张")

    if args.dry_run:
        print(f"\n执行: python3 scripts/data_cleaning/merge_duplicate_roles.py")


if __name__ == "__main__":
    main()