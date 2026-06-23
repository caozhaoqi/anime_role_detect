#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同步本地角色数据到远程 RDS 数据库

功能:
    1. 扫描 training_dataset 和 final_dataset 目录
    2. 统计每个角色的图片数量
    3. 同步到 RDS 数据库的 role_stats 表

用法:
    python3 scripts/data_collection/sync_role_stats.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data_collection"))

from db_utils import DB


def count_images_in_dir(dataset_dir: Path) -> dict:
    """统计目录中每个角色的图片数量"""
    role_counts = defaultdict(int)

    if not dataset_dir.exists():
        print(f"  ⚠️ 目录不存在: {dataset_dir}")
        return role_counts

    for role_dir in dataset_dir.iterdir():
        if not role_dir.is_dir():
            continue

        # 统计图片文件
        count = 0
        for f in role_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                count += 1

        if count > 0:
            role_counts[role_dir.name] = count

    return role_counts


def sync_role_stats():
    """同步角色统计数据到 RDS"""
    print("=" * 60)
    print("同步角色数据到 RDS 数据库")
    print("=" * 60)

    # 扫描数据集
    training_dir = PROJECT_ROOT / "data" / "training_dataset"
    final_dir = PROJECT_ROOT / "data" / "final_dataset"

    print(f"\n📂 扫描训练集: {training_dir}")
    training_stats = count_images_in_dir(training_dir)
    print(f"  ✓ 统计完成: {len(training_stats)} 个角色")

    print(f"\n📂 扫描 Final 集: {final_dir}")
    final_stats = count_images_in_dir(final_dir)
    print(f"  ✓ 统计完成: {len(final_stats)} 个角色")

    # 合并数据（training 优先）
    all_roles = {}
    for role, count in final_stats.items():
        if role in training_stats:
            # 两个目录都有，取 training 的数量（更干净的数据）
            all_roles[role] = {
                'training_count': training_stats[role],
                'final_count': count,
                'total': training_stats[role]  # 使用 training 的数量
            }
        else:
            all_roles[role] = {
                'training_count': 0,
                'final_count': count,
                'total': count
            }

    for role, count in training_stats.items():
        if role not in all_roles:
            all_roles[role] = {
                'training_count': count,
                'final_count': 0,
                'total': count
            }

    print(f"\n📊 合并统计: {len(all_roles)} 个角色")

    # 同步到数据库
    print("\n🔄 同步到 RDS...")

    try:
        # 先清空表（或者使用 REPLACE INTO）
        # DB.execute("TRUNCATE TABLE role_stats")

        synced_count = 0
        skipped_count = 0

        for role_name, stats in sorted(all_roles.items()):
            total = stats['total']
            training_count = stats['training_count']
            final_count = stats['final_count']

            # 检查是否已存在
            existing = DB._fetchone(
                "SELECT id FROM role_stats WHERE role_name = %s",
                (role_name,)
            )

            if existing:
                # 更新
                DB._execute(
                    "UPDATE role_stats SET training_count=%s, final_count=%s, "
                    "total_count=%s, updated_at=NOW() WHERE role_name=%s",
                    (training_count, final_count, total, role_name)
                )
            else:
                # 插入
                DB._execute(
                    "INSERT INTO role_stats (role_name, training_count, final_count, "
                    "total_count, skip_threshold) VALUES (%s, %s, %s, %s, %s)",
                    (role_name, training_count, final_count, total, 100)
                )

            synced_count += 1

        print(f"  ✅ 同步完成: {synced_count} 个角色")

        # 打印大于100的角色（这些会被跳过采集）
        large_roles = [(r, s['total']) for r, s in all_roles.items() if s['total'] >= 100]
        if large_roles:
            print(f"\n⚠️  以下 {len(large_roles)} 个角色 >= 100 张，将跳过采集:")
            for role, count in sorted(large_roles, key=lambda x: x[1], reverse=True)[:20]:
                print(f"    {role}: {count} 张")

        return len(all_roles), synced_count

    except Exception as e:
        print(f"  ❌ 同步失败: {e}")
        raise


def check_db_table():
    """检查并创建 role_stats 表"""
    print("\n🔍 检查数据库表...")

    try:
        # 检查表是否存在
        tables = DB._fetchall("SHOW TABLES")
        table_names = [list(t.values())[0] for t in tables]

        if 'role_stats' not in table_names:
            print("  📦 创建 role_stats 表...")

            DB._execute("""
                CREATE TABLE role_stats (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    role_name VARCHAR(255) NOT NULL UNIQUE,
                    training_count INT DEFAULT 0,
                    final_count INT DEFAULT 0,
                    total_count INT DEFAULT 0,
                    skip_threshold INT DEFAULT 100,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_role_name (role_name),
                    INDEX idx_total_count (total_count)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            print("  ✅ 表创建完成")
        else:
            print("  ✅ 表已存在")

        return True

    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return False


def get_roles_to_skip() -> dict:
    """获取需要跳过的角色列表（>= 100 张）"""
    print("\n🔍 查询需要跳过的角色...")

    try:
        rows = DB._fetchall(
            "SELECT role_name, total_count FROM role_stats WHERE total_count >= %s",
            (100,)
        )

        skip_roles = {row['role_name']: row['total_count'] for row in rows}
        print(f"  ✅ 找到 {len(skip_roles)} 个角色 >= 100 张")

        return skip_roles

    except Exception as e:
        print(f"  ❌ 查询失败: {e}")
        return {}


if __name__ == "__main__":
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 检查并创建表
    if not check_db_table():
        sys.exit(1)

    # 2. 同步数据
    total_roles, synced = sync_role_stats()

    # 3. 获取跳过列表
    skip_roles = get_roles_to_skip()

    print(f"\n⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ 同步完成！")
    print(f"   总角色数: {total_roles}")
    print(f"   同步数量: {synced}")
    print(f"   跳过数量: {len(skip_roles)} (>= 100 张)")
