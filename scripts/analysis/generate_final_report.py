#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 final_dataset 最终统计报告
"""

import os
from pathlib import Path
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect")

# 角色列表文件
ROLE_LIST_FILE = PROJECT_ROOT / "auto_spider_img" / "loli-role.txt"

# 目标目录
TARGET_DIR = PROJECT_ROOT / "data" / "final_dataset"


def parse_role_list():
    """解析角色列表文件"""
    role_info = {}

    if not ROLE_LIST_FILE.exists():
        print(f"警告: 角色列表文件不存在: {ROLE_LIST_FILE}")
        return role_info

    with open(ROLE_LIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 3:
                chinese_name = parts[0]
                game = parts[1] if len(parts) > 1 else "未知"
                english_name = parts[2] if len(parts) > 2 else chinese_name
                japanese_name = parts[3] if len(parts) > 3 else ""

                role_info[english_name] = {
                    "chinese_name": chinese_name,
                    "english_name": english_name,
                    "japanese_name": japanese_name,
                    "game": game,
                }

    return role_info


def generate_final_report():
    """生成最终统计报告"""
    if not TARGET_DIR.exists():
        print(f"目标目录不存在: {TARGET_DIR}")
        return

    # 解析角色列表
    role_info = parse_role_list()

    # 统计信息
    stats = {
        "total_roles": 0,
        "total_files": 0,
        "roles_with_images": 0,
        "roles_without_images": 0,
    }

    # 按游戏分组
    game_stats = defaultdict(lambda: {"roles": 0, "files": 0})

    # 收集每个角色的文件数
    role_file_counts = []

    for role_dir in TARGET_DIR.iterdir():
        if not role_dir.is_dir():
            continue

        stats["total_roles"] += 1

        # 统计文件数
        file_count = len([f for f in role_dir.iterdir() if f.is_file()])
        stats["total_files"] += file_count

        if file_count > 0:
            stats["roles_with_images"] += 1
        else:
            stats["roles_without_images"] += 1

        # 获取角色信息
        info = role_info.get(
            role_dir.name,
            {
                "chinese_name": role_dir.name,
                "english_name": role_dir.name,
                "japanese_name": "",
                "game": "未知",
            },
        )

        # 按游戏统计
        game = info["game"]
        game_stats[game]["roles"] += 1
        game_stats[game]["files"] += file_count

        role_file_counts.append(
            {
                "chinese_name": info["chinese_name"],
                "english_name": info["english_name"],
                "japanese_name": info["japanese_name"],
                "game": game,
                "file_count": file_count,
            }
        )

    # 按文件数量排序
    role_file_counts.sort(key=lambda x: x["file_count"], reverse=True)

    # 生成报告
    report_file = PROJECT_ROOT / "docs" / "final_dataset_最终统计报告.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Final Dataset 最终统计报告\n\n")
        f.write(f"**生成时间**: {os.popen('date').read().strip()}\n")
        f.write(f"**数据集位置**: `{TARGET_DIR}`\n\n")

        f.write("## 整体统计\n\n")
        f.write(f"- 角色总数: {stats['total_roles']}\n")
        f.write(f"- 有图片的角色: {stats['roles_with_images']}\n")
        f.write(f"- 无图片的角色: {stats['roles_without_images']}\n")
        f.write(f"- 图片总数: {stats['total_files']}\n")
        f.write(
            f"- 平均每个角色: {stats['total_files'] // stats['roles_with_images'] if stats['roles_with_images'] > 0 else 0} 张\n\n"
        )

        f.write("## 按游戏统计\n\n")
        f.write("| 游戏 | 角色数 | 图片数 |\n")
        f.write("|------|--------|--------|\n")
        for game in sorted(game_stats.keys()):
            f.write(f"| {game} | {game_stats[game]['roles']} | {game_stats[game]['files']} |\n")
        f.write("\n")

        f.write("## 各角色图片数量\n\n")
        f.write("| 序号 | 中文名 | 英文名 | 日文名 | 游戏 | 图片数 |\n")
        f.write("|------|--------|--------|--------|------|--------|\n")
        for idx, role in enumerate(role_file_counts, 1):
            f.write(
                f"| {idx} | {role['chinese_name']} | {role['english_name']} | {role['japanese_name']} | {role['game']} | {role['file_count']} |\n"
            )

        f.write("\n## 图片数量分布\n\n")
        f.write("### 500+ 张\n")
        for role in role_file_counts:
            if role["file_count"] >= 500:
                f.write(
                    f"- {role['chinese_name']} ({role['english_name']}): {role['file_count']} 张\n"
                )

        f.write("\n### 300-499 张\n")
        for role in role_file_counts:
            if 300 <= role["file_count"] < 500:
                f.write(
                    f"- {role['chinese_name']} ({role['english_name']}): {role['file_count']} 张\n"
                )

        f.write("\n### 200-299 张\n")
        for role in role_file_counts:
            if 200 <= role["file_count"] < 300:
                f.write(
                    f"- {role['chinese_name']} ({role['english_name']}): {role['file_count']} 张\n"
                )

        f.write("\n### 100-199 张\n")
        for role in role_file_counts:
            if 100 <= role["file_count"] < 200:
                f.write(
                    f"- {role['chinese_name']} ({role['english_name']}): {role['file_count']} 张\n"
                )

        f.write("\n### 50-99 张\n")
        for role in role_file_counts:
            if 50 <= role["file_count"] < 100:
                f.write(
                    f"- {role['chinese_name']} ({role['english_name']}): {role['file_count']} 张\n"
                )

        f.write("\n### 1-49 张\n")
        for role in role_file_counts:
            if 1 <= role["file_count"] < 50:
                f.write(
                    f"- {role['chinese_name']} ({role['english_name']}): {role['file_count']} 张\n"
                )

        f.write("\n### 0 张\n")
        for role in role_file_counts:
            if role["file_count"] == 0:
                f.write(
                    f"- {role['chinese_name']} ({role['english_name']}): {role['file_count']} 张\n"
                )

    print(f"报告已保存到: {report_file}")
    print("\n最终统计:")
    print(f"  角色总数: {stats['total_roles']}")
    print(f"  有图片的角色: {stats['roles_with_images']}")
    print(f"  无图片的角色: {stats['roles_without_images']}")
    print(f"  图片总数: {stats['total_files']}")


if __name__ == "__main__":
    generate_final_report()
