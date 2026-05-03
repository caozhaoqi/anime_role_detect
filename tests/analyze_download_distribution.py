#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计已下载图片分布"""

import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "organized_images"


def main():
    print("=" * 70)
    print("📊 已下载图片分布统计")
    print("=" * 70)

    role_stats = defaultdict(int)
    total = 0

    for role_dir in OUTPUT_DIR.iterdir():
        if role_dir.is_dir():
            count = 0
            for img in role_dir.glob("*"):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    count += 1
            if count > 0:
                role_stats[role_dir.name] = count
                total += count

    sorted_roles = sorted(role_stats.items(), key=lambda x: x[1], reverse=True)

    print(f"\n📁 总计: {total} 张图片, {len(sorted_roles)} 个角色")
    print("\n" + "=" * 70)
    print(f"{'排名':<6} {'角色':<25} {'图片数':<10} {'占比':<10}")
    print("-" * 70)

    for i, (role, count) in enumerate(sorted_roles, 1):
        percent = (count / total * 100) if total > 0 else 0
        print(f"{i:<6} {role:<25} {count:<10} {percent:.1f}%")

    print("\n" + "=" * 70)

    ranges = [
        (">= 300", lambda x: x >= 300),
        ("200-299", lambda x: 200 <= x < 300),
        ("100-199", lambda x: 100 <= x < 200),
        ("50-99", lambda x: 50 <= x < 100),
        ("< 50", lambda x: x < 50)
    ]

    print("\n📈 图片数区间分布:")
    print("-" * 50)
    for label, cond in ranges:
        count = sum(1 for c in role_stats.values() if cond(c))
        print(f"{label:<12} {count} 个角色")

    print("\n" + "=" * 70)
    print("\n⚠️  图片数不足的角色:")
    low_roles = sorted([(r, c) for r, c in sorted_roles if c < 100], key=lambda x: x[1])
    if low_roles:
        print("-" * 50)
        for role, count in low_roles:
            print(f"{role:<25} {count} 张")
    else:
        print("所有角色图片数均 >= 100")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
