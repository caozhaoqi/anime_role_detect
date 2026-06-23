#!/usr/bin/env python3
import os
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINAL_DATASET_DIR = PROJECT_ROOT / "data" / "final_dataset"
TRAIN_DATASET_DIR = PROJECT_ROOT / "data" / "train_dataset"
DUP_LOG_FILE = PROJECT_ROOT / "data" / "duplicate_report.txt"

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}


def get_file_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def scan_images(base_dir: Path) -> Dict[str, Tuple[Path, int]]:
    hash_to_file: Dict[str, Tuple[Path, int]] = {}
    total_files = 0
    total_size = 0
    
    for root, dirs, files in os.walk(base_dir):
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                fp = Path(root) / f
                total_files += 1
                total_size += fp.stat().st_size
                f_hash = get_file_hash(fp)
                if f_hash not in hash_to_file:
                    hash_to_file[f_hash] = (fp, 1)
                else:
                    _, count = hash_to_file[f_hash]
                    hash_to_file[f_hash] = (fp, count + 1)
    
    return hash_to_file, total_files, total_size


def find_duplicates(hash_map: Dict[str, Tuple[Path, int]]) -> Tuple[Set[str], Set[str]]:
    unique_hashes = set()
    duplicate_hashes = set()
    
    for f_hash, (_, count) in hash_map.items():
        if count == 1:
            unique_hashes.add(f_hash)
        else:
            duplicate_hashes.add(f_hash)
    
    return unique_hashes, duplicate_hashes


def collect_all_duplicate_files(base_dir: Path) -> Dict[str, list]:
    hash_to_files = defaultdict(list)
    
    for root, dirs, files in os.walk(base_dir):
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                fp = Path(root) / f
                f_hash = get_file_hash(fp)
                hash_to_files[f_hash].append(fp)
    
    return {k: v for k, v in hash_to_files.items() if len(v) > 1}


def deduplicate_and_organize() -> None:
    print("=" * 60)
    print("  数据集去重和归类脚本")
    print("=" * 60)
    
    if not FINAL_DATASET_DIR.exists():
        print(f"❌ final_dataset 目录不存在: {FINAL_DATASET_DIR}")
        return
    
    print("\n📊 正在扫描 final_dataset...")
    hash_map, total_files, total_size = scan_images(FINAL_DATASET_DIR)
    
    print(f"  扫描完成: {total_files} 个文件, {total_size / (1024*1024):.1f} MB")
    
    unique_hashes, duplicate_hashes = find_duplicates(hash_map)
    print(f"  唯一哈希: {len(unique_hashes)}")
    print(f"  重复哈希: {len(duplicate_hashes)}")
    
    duplicate_files_map = collect_all_duplicate_files(FINAL_DATASET_DIR)
    
    dup_count = sum(len(v) - 1 for v in duplicate_files_map.values())
    print(f"  需要删除的重复文件: {dup_count} 个")
    
    TRAIN_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    removed_count = 0
    removed_size = 0
    copied_count = 0
    copied_size = 0
    
    with open(DUP_LOG_FILE, 'w', encoding='utf-8') as log:
        log.write("=" * 60 + "\n")
        log.write("  重复文件报告\n")
        log.write("=" * 60 + "\n")
        log.write(f"  扫描目录: {FINAL_DATASET_DIR}\n")
        log.write(f"  总文件数: {total_files}\n")
        log.write(f"  重复哈希数: {len(duplicate_hashes)}\n")
        log.write(f"  重复文件数: {dup_count}\n")
        log.write("=" * 60 + "\n\n")
        
        for f_hash, files in duplicate_files_map.items():
            log.write(f"\n哈希: {f_hash}\n")
            log.write(f"重复次数: {len(files)}\n")
            log.write("文件列表:\n")
            for i, fp in enumerate(files):
                if i == 0:
                    log.write(f"  [保留] {fp}\n")
                else:
                    log.write(f"  [删除] {fp}\n")
                    try:
                        removed_size += fp.stat().st_size
                        fp.unlink()
                        removed_count += 1
                    except Exception as e:
                        log.write(f"  [删除失败] {fp}: {e}\n")
    
    print(f"\n🗑️ 删除重复文件: {removed_count} 个, 释放 {removed_size / (1024*1024):.1f} MB")
    
    print("\n📦 正在创建硬链接到 train_dataset...")
    for root, dirs, files in os.walk(FINAL_DATASET_DIR):
        rel_path = Path(root).relative_to(FINAL_DATASET_DIR)
        target_dir = TRAIN_DATASET_DIR / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                src = Path(root) / f
                dst = target_dir / f
                
                if not dst.exists():
                    os.link(src, dst)
                    copied_size += src.stat().st_size
                    copied_count += 1
    
    print(f"  硬链接完成: {copied_count} 个文件, 实际占用 0 MB (硬链接共享存储空间)")
    
    final_hash_map, final_total, final_size = scan_images(FINAL_DATASET_DIR)
    _, final_dups = find_duplicates(final_hash_map)
    
    print(f"\n✅ 去重完成!")
    print(f"  final_dataset: {final_total} 个文件, {final_size / (1024*1024):.1f} MB")
    print(f"  train_dataset: {copied_count} 个文件, {copied_size / (1024*1024):.1f} MB")
    print(f"  剩余重复: {len(final_dups)} 个哈希")
    print(f"  重复报告: {DUP_LOG_FILE}")
    
    print("\n📋 角色统计:")
    role_stats = []
    for role_dir in sorted(FINAL_DATASET_DIR.iterdir()):
        if role_dir.is_dir() and not role_dir.name.startswith('.'):
            img_count = sum(1 for f in role_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
            if img_count > 0:
                role_stats.append((role_dir.name, img_count))
    
    for role, count in sorted(role_stats, key=lambda x: -x[1]):
        print(f"  {role}: {count} 张")


if __name__ == "__main__":
    deduplicate_and_organize()