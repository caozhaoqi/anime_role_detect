#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iCloud Drive 同步脚本
- 首次运行: 全量同步
- 后续运行: 增量同步
- 支持定时同步
"""

import os
import shutil
import time
import hashlib
import json
from datetime import datetime

# 配置
CONFIG = {
    "source_dirs": [
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset",
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data",
    ],
    "icloud_base": os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/"),
    "backup_dir_name": "anime_role_detect_backup",
    "sync_info_file": "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/sync_info.json",
}


def get_file_md5(file_path):
    """计算文件MD5值"""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"计算MD5失败 {file_path}: {e}")
        return None


def load_sync_info():
    """加载同步信息"""
    if os.path.exists(CONFIG["sync_info_file"]):
        try:
            with open(CONFIG["sync_info_file"], "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"first_sync_done": False, "last_sync_time": "", "file_hashes": {}}
    return {"first_sync_done": False, "last_sync_time": "", "file_hashes": {}}


def save_sync_info(sync_info):
    """保存同步信息"""
    with open(CONFIG["sync_info_file"], "w", encoding="utf-8") as f:
        json.dump(sync_info, f, indent=2, ensure_ascii=False)


def sync_directory(source_dir, target_dir, sync_info, is_first_sync):
    """同步单个目录"""
    copied_count = 0
    skipped_count = 0
    deleted_count = 0

    # 获取源目录所有文件
    source_files = {}
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            source_path = os.path.join(root, filename)
            rel_path = os.path.relpath(source_path, source_dir)
            source_files[rel_path] = source_path

    # 获取目标目录所有文件
    target_files = {}
    if os.path.exists(target_dir):
        for root, dirs, files in os.walk(target_dir):
            for filename in files:
                target_path = os.path.join(root, filename)
                rel_path = os.path.relpath(target_path, target_dir)
                target_files[rel_path] = target_path

    # 同步文件
    for rel_path, source_path in source_files.items():
        target_path = os.path.join(target_dir, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        if is_first_sync:
            # 首次同步：强制复制
            shutil.copy2(source_path, target_path)
            copied_count += 1
            # 记录文件哈希
            md5 = get_file_md5(source_path)
            if md5:
                sync_info["file_hashes"][rel_path] = md5
        else:
            # 增量同步：只同步新增或修改的文件
            current_md5 = get_file_md5(source_path)
            if not current_md5:
                continue

            # 检查是否已存在且未修改
            if rel_path in sync_info["file_hashes"]:
                if sync_info["file_hashes"][rel_path] == current_md5:
                    skipped_count += 1
                    continue

            # 复制文件
            shutil.copy2(source_path, target_path)
            copied_count += 1
            sync_info["file_hashes"][rel_path] = current_md5

    # 删除目标中多余的文件（保持镜像）
    for rel_path in list(target_files.keys()):
        if rel_path not in source_files:
            try:
                os.remove(target_files[rel_path])
                deleted_count += 1
                if rel_path in sync_info["file_hashes"]:
                    del sync_info["file_hashes"][rel_path]
            except Exception as e:
                print(f"删除文件失败 {target_files[rel_path]}: {e}")

    return copied_count, skipped_count, deleted_count


def main():
    print("=" * 60)
    print(f"iCloud 同步脚本 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 加载同步信息
    sync_info = load_sync_info()
    is_first_sync = not sync_info["first_sync_done"]

    print(f"同步模式: {'首次全量同步' if is_first_sync else '增量同步'}")
    print(f"上次同步时间: {sync_info['last_sync_time'] if sync_info['last_sync_time'] else '从未'}")
    print()

    # 构建目标路径
    icloud_target = os.path.join(CONFIG["icloud_base"], CONFIG["backup_dir_name"])
    print(f"iCloud目标目录: {icloud_target}")
    os.makedirs(icloud_target, exist_ok=True)

    # 同步所有源目录
    total_copied = 0
    total_skipped = 0
    total_deleted = 0

    for source_dir in CONFIG["source_dirs"]:
        if not os.path.exists(source_dir):
            print(f"警告: 源目录不存在 - {source_dir}")
            continue

        rel_name = os.path.basename(source_dir)
        target_dir = os.path.join(icloud_target, rel_name)

        print(f"同步: {os.path.basename(source_dir)}")
        copied, skipped, deleted = sync_directory(source_dir, target_dir, sync_info, is_first_sync)

        print(f"  复制: {copied} | 跳过: {skipped} | 删除: {deleted}")
        total_copied += copied
        total_skipped += skipped
        total_deleted += deleted

    # 更新同步信息
    sync_info["first_sync_done"] = True
    sync_info["last_sync_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_sync_info(sync_info)

    print()
    print("=" * 60)
    print(f"同步完成！")
    print(f"总计 - 复制: {total_copied} | 跳过: {total_skipped} | 删除: {total_deleted}")
    print("=" * 60)
    print("等待 iCloud 后台同步...")
    time.sleep(3)


if __name__ == "__main__":
    main()
