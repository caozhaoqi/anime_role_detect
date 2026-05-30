#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计现有角色的URL数目
"""

import os
import sys
from pypinyin import lazy_pinyin, Style

# 配置
ROLE_LIST_FILE = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt"
URL_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url"


def get_all_roles():
    """获取完整角色列表"""
    roles = []
    with open(ROLE_LIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(" ")
                chinese_name = parts[0]
                roles.append(chinese_name)
    return roles


def count_urls():
    """统计每个角色的URL数量"""
    all_roles = get_all_roles()
    stats = []

    for role in all_roles:
        pinyin = "".join(lazy_pinyin(role, style=Style.TONE3))
        url_file = os.path.join(URL_DIR, f"{pinyin}_img.txt")

        if os.path.exists(url_file):
            with open(url_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                count = len([l for l in lines if l.strip()])
            stats.append((role, count))
        else:
            stats.append((role, 0))

    return stats


def print_stats():
    """打印统计结果"""
    stats = count_urls()

    # 按URL数量排序
    stats.sort(key=lambda x: x[1], reverse=True)

    total_roles = len(stats)
    completed_roles = sum(1 for _, count in stats if count > 0)
    missing_roles = sum(1 for _, count in stats if count == 0)
    total_urls = sum(count for _, count in stats)

    print("=" * 70)
    print(f"角色URL统计报告")
    print("=" * 70)
    print(f"总角色数: {total_roles}")
    print(f"已采集角色数: {completed_roles}")
    print(f"缺失角色数: {missing_roles}")
    print(f"总URL数: {total_urls}")
    print("-" * 70)
    print(f"{'角色':<12} {'URL数量':<8} {'状态'}")
    print("-" * 70)

    for role, count in stats:
        status = "✅" if count > 0 else "❌"
        print(f"{role:<12} {count:<8} {status}")

    # 打印缺失角色列表
    if missing_roles > 0:
        print("\n" + "-" * 70)
        print("缺失URL的角色:")
        missing_list = [role for role, count in stats if count == 0]
        print(", ".join(missing_list[:10]) + ("..." if len(missing_list) > 10 else ""))


if __name__ == "__main__":
    print_stats()
