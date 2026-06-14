#!/usr/bin/env python3
"""统计数据目录信息"""
import os
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")

# 统计每个角色的图片数
character_stats = defaultdict(lambda: {"jpg": 0, "png": 0, "other": 0})

for char_dir in DATA_DIR.iterdir():
    if char_dir.is_dir():
        char_name = char_dir.name
        for img_file in char_dir.iterdir():
            if img_file.is_file():
                ext = img_file.suffix.lower()
                if ext == ".jpg" or ext == ".jpeg":
                    character_stats[char_name]["jpg"] += 1
                elif ext == ".png":
                    character_stats[char_name]["png"] += 1
                else:
                    character_stats[char_name]["other"] += 1

# 计算总数
total_chars = len(character_stats)
total_images = sum(s["jpg"] + s["png"] + s["other"] for s in character_stats.values())
total_jpg = sum(s["jpg"] for s in character_stats.values())
total_png = sum(s["png"] for s in character_stats.values())

# 按图片数排序
sorted_chars = sorted(character_stats.items(), key=lambda x: x[1]["jpg"] + x[1]["png"], reverse=True)

# 图片数分布
distribution = defaultdict(int)
for char_name, stats in character_stats.items():
    count = stats["jpg"] + stats["png"]
    if count >= 100:
        distribution["100+"] += 1
    elif count >= 50:
        distribution["50-99"] += 1
    elif count >= 30:
        distribution["30-49"] += 1
    elif count >= 20:
        distribution["20-29"] += 1
    elif count >= 10:
        distribution["10-19"] += 1
    else:
        distribution["0-9"] += 1

print("=" * 60)
print("数据采集进展分析")
print("=" * 60)
print(f"\n总体统计:")
print(f"  角色目录数: {total_chars}")
print(f"  图片总数: {total_images}")
print(f"  JPG: {total_jpg} ({total_jpg/total_images*100:.1f}%)")
print(f"  PNG: {total_png} ({total_png/total_images*100:.1f}%)")

print(f"\n图片数分布:")
for range_name in ["100+", "50-99", "30-49", "20-29", "10-19", "0-9"]:
    count = distribution[range_name]
    print(f"  {range_name}张: {count} 个角色 ({count/total_chars*100:.1f}%)")

print(f"\n图片数最多的前20个角色:")
for i, (char_name, stats) in enumerate(sorted_chars[:20], 1):
    total = stats["jpg"] + stats["png"]
    print(f"  {i}. {char_name}: {total}张 (JPG:{stats['jpg']}, PNG:{stats['png']})")

print(f"\n未达到100张目标的角色数: {sum(1 for s in character_stats.values() if s['jpg']+s['png'] < 100)}")
print(f"已达到100张目标的角色数: {sum(1 for s in character_stats.values() if s['jpg']+s['png'] >= 100)}")