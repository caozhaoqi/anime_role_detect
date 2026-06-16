#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建全局哈希数据库，用于跨角色去重。

扫描 data/final_dataset 下所有图片，计算 sha256，
存入 SQLite 数据库（image_hashes.db），服务器直接从 db 读取，无需传输图片。

Usage:
    python3 scripts/data_collection/build_hash_db.py                    # 构建/更新 db
    python3 scripts/data_collection/build_hash_db.py --db ./hashes.db   # 指定输出路径
    python3 scripts/data_collection/build_hash_db.py --stats            # 仅查看统计
"""

import os
import sys
import sqlite3
import hashlib
import argparse
import time
from pathlib import Path
from typing import Set, Tuple

# ── 路径 ──
PROJECT_ROOT = Path(__file__).parent.parent.parent
FINAL_DATASET_DIR = PROJECT_ROOT / "data" / "training_dataset"
# data/training_dataset``
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "image_hashes.db"

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def compute_sha256(file_path: Path) -> str:
    """计算文件 sha256"""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def scan_and_build(root_dir: str, db_path: str, progress: bool = True) -> Tuple[int, int, int]:
    """
    扫描 final_dataset，构建/更新 SQLite 哈希库。

    数据库 schema:
        image_hashes (
            hash       TEXT PRIMARY KEY,   -- sha256
            roles      TEXT NOT NULL,      -- 逗号分隔的角色名列表
            file_count INTEGER DEFAULT 1,  -- 总出现次数
        )
    """
    root = Path(root_dir)
    if not root.exists():
        print(f"❌ 目录不存在: {root_dir}")
        sys.exit(1)

    # 收集所有图片
    all_images: list[Path] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                all_images.append(f)

    total = len(all_images)
    print(f"📂 扫描到 {total} 张图片")

    # 计算哈希 → {(hash, role)}
    hash_role_pairs: set[tuple[str, str]] = set()
    for i, img_path in enumerate(all_images, 1):
        if progress and (i % 200 == 0 or i == total):
            print(f"  计算哈希: {i}/{total}")
        try:
            h = compute_sha256(img_path)
            role = img_path.parent.name
            hash_role_pairs.add((h, role))
        except Exception as e:
            print(f"  ⚠️ 跳过 {img_path}: {e}")

    # 聚合: hash → set(roles), count
    hash_map: dict[str, set[str]] = {}
    hash_count: dict[str, int] = {}
    for h, role in hash_role_pairs:
        if h not in hash_map:
            hash_map[h] = set()
        hash_map[h].add(role)
        hash_count[h] = hash_count.get(h, 0) + 1

    unique = len(hash_map)
    duplicate = total - unique
    print(f"  ✅ 唯一哈希: {unique}, 重复: {duplicate} ({duplicate/total*100:.1f}%)" if total else "")

    # 写入 SQLite
    print(f"💾 写入数据库: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS image_hashes (
            hash       TEXT PRIMARY KEY,
            roles      TEXT NOT NULL,
            file_count INTEGER NOT NULL DEFAULT 1
        )
    """)
    # 清空旧数据以全量重建
    cur.execute("DELETE FROM image_hashes")

    batch = []
    for h, roles in hash_map.items():
        roles_str = ",".join(sorted(roles))
        batch.append((h, roles_str, hash_count[h]))

    cur.executemany("INSERT INTO image_hashes (hash, roles, file_count) VALUES (?, ?, ?)", batch)
    conn.commit()
    conn.close()

    print(f"  ✅ 写入完成: {unique} 条记录")
    return total, unique, duplicate


def print_stats(db_path: str):
    """打印数据库统计"""
    if not Path(db_path).exists():
        print(f"❌ 数据库不存在: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM image_hashes").fetchone()[0]
    multi_role = cur.execute("SELECT COUNT(*) FROM image_hashes WHERE roles LIKE '%,%'").fetchone()[0]
    total_files = cur.execute("SELECT SUM(file_count) FROM image_hashes").fetchone()[0]

    print(f"\n📊 哈希数据库统计: {db_path}")
    print(f"  {'='*40}")
    print(f"  唯一哈希数:    {total}")
    print(f"  覆盖图片数:    {total_files}")
    print(f"  跨角色哈希:    {multi_role}")

    if multi_role > 0:
        print(f"\n  跨角色重复 TOP10:")
        rows = cur.execute("""
            SELECT hash, roles, file_count FROM image_hashes
            WHERE roles LIKE '%,%'
            ORDER BY file_count DESC LIMIT 10
        """).fetchall()
        for h, roles, cnt in rows:
            print(f"    {h[:12]}...  → [{roles}]  出现{cnt}次")

    print()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="构建全局哈希数据库（跨角色去重）")
    parser.add_argument("--data-dir", type=str, default=str(FINAL_DATASET_DIR),
                        help="数据集目录 (默认: data/final_dataset)")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH),
                        help="SQLite 数据库路径 (默认: data/image_hashes.db)")
    parser.add_argument("--stats", action="store_true",
                        help="仅查看数据库统计，不重新构建")
    args = parser.parse_args()

    if args.stats:
        print_stats(args.db)
        return

    t0 = time.time()
    total, unique, duplicate = scan_and_build(args.data_dir, args.db)
    elapsed = time.time() - t0
    print(f"\n⏱️ 耗时: {elapsed:.1f}s")
    print(f"📄 数据库: {args.db}  ({os.path.getsize(args.db) / 1024:.1f} KB)")

    # 展示统计
    print_stats(args.db)


if __name__ == "__main__":
    main()