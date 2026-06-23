#!/usr/bin/env python3
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINAL_DATASET_DIR = PROJECT_ROOT / "data" / "final_dataset"
TRAIN_DATASET_DIR = PROJECT_ROOT / "data" / "train_dataset"

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}


def count_images_per_role(base_dir: Path) -> Tuple[Dict[str, int], int, int]:
    role_counts: Dict[str, int] = {}
    total_images = 0
    total_roles = 0
    
    if not base_dir.exists():
        return role_counts, 0, 0
    
    for role_dir in sorted(base_dir.iterdir()):
        if role_dir.is_dir() and not role_dir.name.startswith('.'):
            img_count = sum(1 for f in role_dir.iterdir() 
                           if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            if img_count > 0:
                role_counts[role_dir.name] = img_count
                total_images += img_count
                total_roles += 1
    
    return role_counts, total_images, total_roles


def generate_report() -> None:
    print("=" * 70)
    print("  数据集角色分布报告")
    print("=" * 70)
    
    final_counts, final_total, final_roles = count_images_per_role(FINAL_DATASET_DIR)
    train_counts, train_total, train_roles = count_images_per_role(TRAIN_DATASET_DIR)
    
    all_roles = sorted(set(final_counts.keys()) | set(train_counts.keys()))
    
    print(f"\n📊 总体统计")
    print("-" * 70)
    print(f"{'项目':<25} {'final_dataset':>15} {'train_dataset':>15}")
    print(f"{'角色总数':<25} {final_roles:>15} {train_roles:>15}")
    print(f"{'图片总数':<25} {final_total:>15} {train_total:>15}")
    
    print(f"\n📈 角色分布对比")
    print("-" * 70)
    print(f"{'角色名':<25} {'final_dataset':>15} {'train_dataset':>15} {'差异':>10}")
    print("-" * 70)
    
    for role in all_roles:
        f_count = final_counts.get(role, 0)
        t_count = train_counts.get(role, 0)
        diff = t_count - f_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{role:<25} {f_count:>15} {t_count:>15} {diff_str:>10}")
    
    print(f"\n📋 final_dataset Top 20 角色")
    print("-" * 70)
    for i, (role, count) in enumerate(sorted(final_counts.items(), key=lambda x: -x[1])[:20], 1):
        print(f"  {i:2d}. {role:<20} {count:>8} 张")
    
    print(f"\n📋 train_dataset Top 20 角色")
    print("-" * 70)
    for i, (role, count) in enumerate(sorted(train_counts.items(), key=lambda x: -x[1])[:20], 1):
        print(f"  {i:2d}. {role:<20} {count:>8} 张")
    
    print(f"\n📊 图片数量区间分布")
    print("-" * 70)
    
    def get_bucket_distribution(counts: Dict[str, int]) -> Dict[str, int]:
        buckets = {'1-10': 0, '11-30': 0, '31-50': 0, '51-80': 0, '80+': 0}
        for count in counts.values():
            if count <= 10:
                buckets['1-10'] += 1
            elif count <= 30:
                buckets['11-30'] += 1
            elif count <= 50:
                buckets['31-50'] += 1
            elif count <= 80:
                buckets['51-80'] += 1
            else:
                buckets['80+'] += 1
        return buckets
    
    final_buckets = get_bucket_distribution(final_counts)
    train_buckets = get_bucket_distribution(train_counts)
    
    print(f"{'区间':<10} {'final_dataset':>15} {'train_dataset':>15}")
    for bucket in ['1-10', '11-30', '31-50', '51-80', '80+']:
        print(f"{bucket:<10} {final_buckets[bucket]:>15} {train_buckets[bucket]:>15}")
    
    print(f"\n📊 统计摘要")
    print("-" * 70)
    
    if final_counts:
        avg_final = sum(final_counts.values()) / len(final_counts)
        min_final = min(final_counts.values())
        max_final = max(final_counts.values())
        print(f"final_dataset: 平均 {avg_final:.1f} 张/角色, 最小 {min_final} 张, 最大 {max_final} 张")
    
    if train_counts:
        avg_train = sum(train_counts.values()) / len(train_counts)
        min_train = min(train_counts.values())
        max_train = max(train_counts.values())
        print(f"train_dataset: 平均 {avg_train:.1f} 张/角色, 最小 {min_train} 张, 最大 {max_train} 张")
    
    only_final = [r for r in all_roles if r in final_counts and r not in train_counts]
    only_train = [r for r in all_roles if r in train_counts and r not in final_counts]
    
    if only_final:
        print(f"\n⚠️ 仅存在于 final_dataset 的角色 ({len(only_final)} 个):")
        print("  " + ", ".join(only_final))
    
    if only_train:
        print(f"\n⚠️ 仅存在于 train_dataset 的角色 ({len(only_train)} 个):")
        print("  " + ", ".join(only_train))
    
    same_count = sum(1 for r in all_roles if final_counts.get(r, 0) == train_counts.get(r, 0))
    print(f"\n✅ 两个数据集数量一致的角色: {same_count} 个")


if __name__ == "__main__":
    generate_report()