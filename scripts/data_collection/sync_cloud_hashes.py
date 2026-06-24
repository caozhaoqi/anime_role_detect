#!/usr/bin/env python3
"""
云端哈希库同步脚本

逻辑：
1. 扫描本地 data/final_dataset 下所有图片，计算 SHA256
2. 查询 RDS image_hashes 表所有哈希
3. 删除 RDS 中本地不存在的哈希（清理云端脏数据）

用法：
    # 扫描指定目录，报告差异（不删除）
    python3 scripts/data_collection/sync_cloud_hashes.py --data-dir /path/to/images

    # 报告 + 删除云端多余哈希
    python3 scripts/data_collection/sync_cloud_hashes.py --data-dir /path/to/images --delete

    # 使用默认路径 (data/final_dataset)
    python3 scripts/data_collection/sync_cloud_hashes.py
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Set, Tuple

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_collection.db_utils import DB

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "final_dataset"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def compute_sha256(file_path: Path) -> str:
    """计算文件 sha256 哈希"""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def scan_local_hashes(data_dir: Path) -> Tuple[Set[str], int]:
    """扫描本地所有图片，返回 (hash_set, count)"""
    local_hashes: Set[str] = set()
    total_files = 0

    if not data_dir.exists():
        print(f"❌ 目录不存在: {data_dir}")
        return local_hashes, 0

    for role_dir in sorted(data_dir.iterdir()):
        if not role_dir.is_dir():
            continue
        for f in role_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                total_files += 1
                try:
                    h = compute_sha256(f)
                    local_hashes.add(h)
                except Exception as e:
                    print(f"  ⚠️ 计算哈希失败: {f} - {e}")

        if total_files % 500 == 0:
            print(f"  扫描进度: {total_files} 文件...")

    return local_hashes, total_files


def main():
    parser = argparse.ArgumentParser(description="同步云端哈希库：清理云端但本地不存在的冗余哈希")
    parser.add_argument("--delete", action="store_true", help="执行删除（默认只报告）")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help=f"图片数据目录 (默认: {DEFAULT_DATA_DIR})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 1. 扫描本地
    print(f"📂 扫描本地图片: {data_dir}")
    local_hashes, total_files = scan_local_hashes(data_dir)
    print(f"  本地文件数: {total_files}")
    print(f"  本地唯一哈希数: {len(local_hashes)}")

    # 2. 加载云端哈希
    print("\n☁️  加载云端哈希库...")
    cloud_hashes, cloud_total = DB.load_all_hashes()
    print(f"  云端哈希总数: {cloud_total}")

    # 3. 对比
    only_in_cloud = cloud_hashes - local_hashes
    only_in_local = local_hashes - cloud_hashes

    print(f"\n📊 差异分析:")
    print(f"  云端有、本地无 (待删除): {len(only_in_cloud)}")
    print(f"  本地有、云端无 (待补充): {len(only_in_local)}")

    if not only_in_cloud:
        print("\n✅ 云端数据干净，没有冗余哈希")
        return

    # 4. 展示冗余哈希样本
    print(f"\n  冗余哈希样本 (前10):")
    for h in sorted(only_in_cloud)[:10]:
        row = DB._fetchone("SELECT hash, roles, file_count FROM image_hashes WHERE hash = %s", (h,))
        if row:
            print(f"    {row['hash'][:16]}... | roles: {row['roles']} | count: {row['file_count']}")

    # 5. 确认删除
    if not args.delete:
        print(f"\n⚠️  未指定 --delete，仅报告不删除")
        print(f"   确认无误后执行: python3 {__file__} --delete")
        DB.close()
        return

    total_deleted = 0
    batch_size = 100
    redundant_list = sorted(only_in_cloud)

    print(f"\n🗑️  开始删除 {len(redundant_list)} 个冗余哈希...")

    for i in range(0, len(redundant_list), batch_size):
        batch = redundant_list[i:i + batch_size]
        placeholders = ",".join(["%s"] * len(batch))
        deleted = DB._execute(
            f"DELETE FROM image_hashes WHERE hash IN ({placeholders})",
            batch
        )
        total_deleted += deleted
        print(f"  进度: {min(i + batch_size, len(redundant_list))}/{len(redundant_list)} (已删 {total_deleted})")

    DB.close()
    print(f"\n✅ 同步完成！共删除 {total_deleted} 个冗余哈希")


if __name__ == "__main__":
    main()