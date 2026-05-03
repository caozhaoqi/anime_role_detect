#!/usr/bin/env python3
"""统计已下载图片构成"""
import os
import sys
from pathlib import Path
from collections import defaultdict

# 配置
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "organized_images"
DB_PATH = PROJECT_ROOT / "data" / "role_images.db"

def main():
    print("=" * 80)
    print(" 📊 已下载图片构成统计")
    print("=" * 80)

    # 统计本地文件
    role_counts = defaultdict(int)
    ext_counts = defaultdict(int)
    total_files = 0
    total_size = 0

    if OUTPUT_DIR.exists():
        for role_dir in OUTPUT_DIR.iterdir():
            if role_dir.is_dir():
                role_name = role_dir.name
                for file in role_dir.iterdir():
                    if file.is_file() and file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.gif'):
                        role_counts[role_name] += 1
                        ext_counts[file.suffix.lower()] += 1
                        total_files += 1
                        total_size += file.stat().st_size

    print(f"\n📁 本地图片统计:")
    print(f"  总文件数: {total_files:,}")
    print(f"  总大小: {total_size / (1024 * 1024):.2f} MB")
    print(f"  角色数: {len(role_counts)}")

    print("\n📂 文件类型分布:")
    total_ext = sum(ext_counts.values())
    for ext, count in sorted(ext_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_ext * 100
        print(f"  {ext}: {count:,} ({pct:.1f}%)")

    print("\n👤 角色图片分布 (前20名):")
    print(f"  {'角色':<20} {'数量':>8} {'占比':<6}")
    print("  " + "-" * 40)
    
    for i, (role, count) in enumerate(sorted(role_counts.items(), key=lambda x: x[1], reverse=True)[:20], 1):
        pct = count / total_files * 100
        print(f"  {i}. {role:<20} {count:>8} {pct:.1f}%")

    # 统计数据库
    print("\n" + "=" * 80)
    print(" 🗄️ 数据库统计")
    print("=" * 80)

    try:
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # 下载记录统计
        cursor.execute('SELECT COUNT(*) FROM downloaded_images')
        db_total = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM downloaded_images WHERE status = "success"')
        db_success = cursor.fetchone()[0]

        cursor.execute('SELECT role_name, COUNT(*) FROM downloaded_images WHERE status = "success" GROUP BY role_name')
        db_role_counts = {r[0]: r[1] for r in cursor.fetchall()}

        # raw_urls统计
        cursor.execute('SELECT COUNT(*) FROM raw_urls')
        raw_total = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM raw_urls WHERE status = "pending"')
        raw_pending = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM raw_urls WHERE status = "downloaded"')
        raw_downloaded = cursor.fetchone()[0]

        conn.close()

        print(f"\n📊 下载记录:")
        print(f"  总记录数: {db_total:,}")
        print(f"  成功下载: {db_success:,}")

        print(f"\n📸 URL状态:")
        print(f"  总URL数: {raw_total:,}")
        print(f"  待下载: {raw_pending:,}")
        print(f"  已下载: {raw_downloaded:,}")

        print("\n👤 数据库角色分布 (前10名):")
        for i, (role, count) in enumerate(sorted(db_role_counts.items(), key=lambda x: x[1], reverse=True)[:10], 1):
            print(f"  {i}. {role}: {count:,}")

    except Exception as e:
        print(f"❌ 数据库统计失败: {e}")

    # 对比分析
    print("\n" + "=" * 80)
    print(" 📈 对比分析")
    print("=" * 80)

    if role_counts and db_role_counts:
        print("\n本地 vs 数据库差异:")
        print(f"  {'角色':<20} {'本地':>8} {'数据库':>10} {'差异':>8}")
        print("  " + "-" * 50)
        
        all_roles = set(role_counts.keys()) | set(db_role_counts.keys())
        diff_count = 0
        
        for role in sorted(all_roles):
            local = role_counts.get(role, 0)
            db = db_role_counts.get(role, 0)
            diff = local - db
            if diff != 0:
                diff_count += 1
                print(f"  {role:<20} {local:>8} {db:>10} {diff:>8}")
        
        print(f"\n  差异角色数: {diff_count}")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
