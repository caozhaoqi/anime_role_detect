#!/usr/bin/env python3
"""统计URL数量分布"""
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.data_collection.database.database_functions import DatabaseManager
from spider_image_system.src.run.constants import PINYIN_MAPPING

def get_role_name_from_pinyin(pinyin: str) -> str:
    """从拼音获取角色名"""
    for name, py in PINYIN_MAPPING.items():
        if py == pinyin:
            return name
    return pinyin

def count_urls_from_files():
    """从img_url目录统计URL数量"""
    img_url_dir = project_root / "spider_image_system" / "data" / "img_url"
    if not img_url_dir.exists():
        return {}

    role_counts = {}
    for file in img_url_dir.glob("*_img.txt"):
        pinyin_name = file.stem.replace("_img", "")
        display_name = get_role_name_from_pinyin(pinyin_name)
        try:
            with open(file, 'r', encoding='utf-8') as f:
                count = len([line for line in f if line.strip()])
            role_counts[display_name] = count
        except Exception:
            pass
    return role_counts

def print_distribution(title, counts_dict):
    """打印分布统计"""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")

    if not counts_dict:
        print("  无数据")
        return

    total = len(counts_dict)
    total_urls = sum(counts_dict.values())

    # 分组统计
    groups = {
        '🔴 紧急 (<20)': [],
        '🟠 不足 (20-49)': [],
        '🟡 较少 (50-99)': [],
        '🔵 达标 (100-199)': [],
        '🟢 充足 (>=200)': [],
    }

    for role, count in counts_dict.items():
        if count < 20:
            groups['🔴 紧急 (<20)'].append((role, count))
        elif count < 50:
            groups['🟠 不足 (20-49)'].append((role, count))
        elif count < 100:
            groups['🟡 较少 (50-99)'].append((role, count))
        elif count < 200:
            groups['🔵 达标 (100-199)'].append((role, count))
        else:
            groups['🟢 充足 (>=200)'].append((role, count))

    print(f"\n📊 总体统计:")
    print(f"  角色总数: {total}")
    print(f"  URL总数:  {total_urls:,}")
    print(f"  平均:     {total_urls/total:.1f} URL/角色")

    print(f"\n📈 分布情况:")
    for group_name, items in groups.items():
        pct = len(items) / total * 100 if total > 0 else 0
        print(f"  {group_name}: {len(items):>3} 个角色 ({pct:>5.1f}%)")

    # 打印各组详情
    for group_name, items in groups.items():
        if items:
            print(f"\n{group_name}:")
            for role, count in sorted(items, key=lambda x: x[1], reverse=True):
                bar = "█" * min(count // 10, 20)
                print(f"  {role:<20} {count:>5} │{bar}")

    return groups

print("=" * 80)
print(f" URL数量分布统计报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# 1. 从数据库统计
print("\n📂 数据源1: SQLite数据库")
db = DatabaseManager(db_type='sqlite')
if db.connect():
    stats = db.get_collection_statistics()
    db_counts = {r: c for r, c in stats.get('role_stats', [])}
    db_groups = print_distribution("数据库角色URL分布", db_counts)
    db.close()
else:
    db_counts = {}
    print("  数据库连接失败")

# 2. 从img_url文件统计
print("\n\n📂 数据源2: img_url目录文件")
file_counts = count_urls_from_files()
file_groups = print_distribution("文件角色URL分布", file_counts)

# 3. 对比分析
if db_counts and file_counts:
    print("\n" + "=" * 80)
    print(" 对比分析: 数据库 vs 文件")
    print("=" * 80)

    all_roles = set(db_counts.keys()) | set(file_counts.keys())

    db_only = set(db_counts.keys()) - set(file_counts.keys())
    file_only = set(file_counts.keys()) - set(db_counts.keys())
    common = set(db_counts.keys()) & set(file_counts.keys())

    print(f"\n  数据库有但文件没有: {len(db_only)} 个")
    if db_only:
        for r in sorted(db_only)[:5]:
            print(f"    - {r}: DB={db_counts[r]}, File=0")
        if len(db_only) > 5:
            print(f"    ... 还有 {len(db_only)-5} 个")

    print(f"\n  文件有但数据库没有: {len(file_only)} 个")
    if file_only:
        for r in sorted(file_only)[:5]:
            print(f"    - {r}: DB=0, File={file_counts[r]}")
        if len(file_only) > 5:
            print(f"    ... 还有 {len(file_only)-5} 个")

    print(f"\n  两者都有: {len(common)} 个")

    # 数量差异
    diff_count = 0
    for role in common:
        if db_counts[role] != file_counts.get(role, 0):
            diff_count += 1

    print(f"\n  数量不一致的角色: {diff_count} 个")

    if diff_count > 0:
        print("\n  差异较大的角色:")
        for role in common:
            db_val = db_counts[role]
            file_val = file_counts.get(role, 0)
            if abs(db_val - file_val) > 50:
                print(f"    {role}: DB={db_val}, File={file_val}, 差值={db_val-file_val}")

# 4. 总结建议
print("\n" + "=" * 80)
print(" 📋 总结建议")
print("=" * 80)

if file_counts:
    urgent = len([c for c in file_counts.values() if c < 20])
    insufficient = len([c for c in file_counts.values() if 20 <= c < 100])
    sufficient = len([c for c in file_counts.values() if c >= 200])

    print(f"\n  🚨 急需采集 (<20 URL):   {urgent} 个角色")
    if urgent > 0:
        low_roles = [(r, c) for r, c in file_counts.items() if c < 20]
        for r, c in sorted(low_roles, key=lambda x: x[1]):
            print(f"     - {r}: {c} 条")

    print(f"\n  ⚠️ 需要补充 (20-99 URL): {insufficient} 个角色")
    if insufficient > 0:
        mid_roles = [(r, c) for r, c in file_counts.items() if 20 <= c < 100]
        for r, c in sorted(mid_roles, key=lambda x: x[1], reverse=True)[:5]:
            print(f"     - {r}: {c} 条")
        if insufficient > 5:
            print(f"     ... 还有 {insufficient - 5} 个角色")

    print(f"\n  ✅ 采集充足 (>=200 URL): {sufficient} 个角色")

    if urgent > 0 or insufficient > 10:
        print("\n💡 建议: 优先补充URL数量不足的角色")

print("\n" + "=" * 80)
