#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计角色图片数量
"""

import os

def count_role_images():
    # 角色列表文件
    role_file = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
    # 图片目录
    image_dirs = [
        '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset',
        '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/merged_english_dataset',
        '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
    ]

    # 读取角色列表（提取英文名称，第三列）
    with open(role_file, 'r', encoding='utf-8') as f:
        roles = []
        role_info = {}  # 存储完整信息
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    english_name = parts[2]
                    roles.append(english_name)
                    role_info[english_name] = line  # 保存完整行

    print(f"角色总数: {len(roles)}")
    print("-" * 60)

    # 统计每个角色的图片数量
    role_stats = []
    for role in roles:
        total_count = 0
        for img_dir in image_dirs:
            role_dir = os.path.join(img_dir, role)
            if os.path.exists(role_dir) and os.path.isdir(role_dir):
                try:
                    files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
                    total_count += len(files)
                except Exception as e:
                    pass
        if total_count > 0:
            role_stats.append((role, total_count))
        else:
            role_stats.append((role, 0))

    # 筛选图片少于50张的角色
    low_count_roles = [(r, c) for r, c in role_stats if c < 50]
    low_count_roles.sort(key=lambda x: x[1])

    # 输出结果
    print(f"图片少于50张的角色: {len(low_count_roles)} 个")
    print("-" * 60)
    for role, count in low_count_roles:
        full_info = role_info.get(role, "")
        print(f"{role:<15} {count:3d} 张  | {full_info}")

    print("-" * 60)
    print(f"无图片的角色: {sum(1 for _, c in low_count_roles if c == 0)} 个")

if __name__ == '__main__':
    count_role_images()
