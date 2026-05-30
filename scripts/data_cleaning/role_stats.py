#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出角色图片数目分布
"""
import os


def main():
    DATASET_PATH = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset"
    ROLE_LIST_FILE = (
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt"
    )

    # 读取角色列表
    roles = {}
    with open(ROLE_LIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                cn_name = parts[0]
                en_name = parts[2]
                roles[en_name] = cn_name

    # 统计各目录图片数量
    stats = []
    for dirname in sorted(os.listdir(DATASET_PATH)):
        dir_path = os.path.join(DATASET_PATH, dirname)
        if not os.path.isdir(dir_path) or dirname.startswith(".") or dirname.endswith(".json"):
            continue

        count = len([f for f in os.listdir(dir_path) if f.lower().endswith(".jpg")])
        cn_name = roles.get(dirname, "未知")
        stats.append({"en_name": dirname, "cn_name": cn_name, "count": count})

    # 按图片数排序
    stats.sort(key=lambda x: x["count"], reverse=True)

    # 输出
    print("=" * 80)
    print("📊 角色图片数目分布")
    print("=" * 80)
    print(f"{'英文名':<20} {'中文名':<12} {'图片数':<6}")
    print("-" * 80)

    total = 0
    for stat in stats:
        print(f"{stat['en_name']:<20} {stat['cn_name']:<12} {stat['count']:<6}")
        total += stat["count"]

    print("-" * 80)
    print(f"总计: {len(stats)} 个角色，共 {total:,} 张图片")

    # 检查 Iselin 和 Iselin LeviSius
    print("\n" + "=" * 80)
    print("🔍 Iselin 和 Iselin LeviSius 的关系")
    print("=" * 80)

    iselin_count = None
    iselin_levisius_count = None

    for stat in stats:
        if stat["en_name"] == "Iselin":
            iselin_count = stat["count"]
        elif stat["en_name"] == "Iselin LeviSius":
            iselin_levisius_count = stat["count"]

    print(f"Iselin: {iselin_count} 张图片")
    print(f"Iselin LeviSius: {iselin_levisius_count} 张图片")

    # 检查角色列表
    print("\n📋 角色列表中相关条目:")
    with open(ROLE_LIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if "伊瑟琳" in line or "Iselin" in line:
                print(f"  {line.strip()}")

    print("\n结论：Iselin 和 Iselin LeviSius 应该是同一个角色（伊瑟琳·利维休斯）")
    print("建议将 Iselin 目录合并到 Iselin LeviSius")


if __name__ == "__main__":
    main()
