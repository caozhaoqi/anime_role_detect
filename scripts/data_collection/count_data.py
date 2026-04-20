#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据统计脚本
统计已采集的数据数量
"""

import os

DATA_DIR = "./data/downloaded_images"

# 统计数据
total_images = 0
total_roles = 0
role_stats = {}

for role_dir in os.listdir(DATA_DIR):
    role_path = os.path.join(DATA_DIR, role_dir)
    if not os.path.isdir(role_path):
        continue
    
    # 统计图片数量
    images = [f for f in os.listdir(role_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    count = len(images)
    
    if count > 0:
        role_stats[role_dir] = count
        total_images += count
        total_roles += 1

# 打印统计结果
print("=" * 60)
print("已采集数据统计")
print("=" * 60)
print(f"总图片数量: {total_images} 张")
print(f"总角色数量: {total_roles} 个")
print(f"平均每个角色: {total_images / total_roles:.1f} 张")
print()

# 按数量排序
sorted_roles = sorted(role_stats.items(), key=lambda x: x[1], reverse=True)

print("角色分布 (按图片数量排序):")
print("-" * 60)
for i, (role, count) in enumerate(sorted_roles, start=1):
    print(f"{i:3d}. {role:20s}: {count:4d} 张")

print()
print("=" * 60)
print("统计完成")
print("=" * 60)
