#!/usr/bin/env python3
"""使用项目现有代码对final_dataset进行NSFW检测"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import json

# 设置路径
DATASET_PATH = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset')
OUTPUT_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/nsfw_results')


def detect_skin_percentage(image):
    """检测皮肤占比（使用项目现有代码）"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 30, 80], dtype=np.uint8)
    upper_skin = np.array([30, 180, 230], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)

    lower_skin2 = np.array([160, 30, 80], dtype=np.uint8)
    upper_skin2 = np.array([180, 180, 230], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

    mask = cv2.bitwise_or(mask1, mask2)
    skin_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]

    return (skin_pixels / total_pixels) * 100


def detect_sensitive_areas(image):
    """检测敏感区域特征（使用项目现有代码）"""
    height, width = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50], dtype=np.uint8)
    upper_red = np.array([10, 255, 255], dtype=np.uint8)
    mask_red1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red2 = np.array([170, 50, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    mid_region = image[int(height * 0.3) : int(height * 0.7), :]
    mid_red_pixels = cv2.countNonZero(
        cv2.bitwise_and(red_mask, red_mask)[int(height * 0.3) : int(height * 0.7), :]
    )
    mid_total = mid_region.shape[0] * mid_region.shape[1]

    red_percentage = (mid_red_pixels / mid_total) * 100 if mid_total > 0 else 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edge_density = (cv2.countNonZero(edges) / (height * width)) * 100

    return red_percentage, edge_density


def analyze_nsfw(image_path):
    """基于规则的NSFW分析（使用项目现有代码）"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return str(image_path), 0.0, "无法加载", 0.0, 0.0

        skin_percent = detect_skin_percentage(image)
        red_percent, edge_density = detect_sensitive_areas(image)

        score = 0.0

        if skin_percent > 35:
            score += (skin_percent - 35) * 0.3
        if skin_percent > 55:
            score += (skin_percent - 55) * 0.5

        if red_percent > 3:
            score += (red_percent - 3) * 2
        if red_percent > 10:
            score += (red_percent - 10) * 3

        if edge_density < 1.5:
            score += 15

        score = min(score, 100.0)

        if score < 25:
            label = "Safe"
        elif score < 50:
            label = "Suggestive"
        else:
            label = "NSFW"

        return str(image_path), score, label, skin_percent, edge_density

    except Exception as e:
        print(f"处理失败 {image_path}: {e}")
        return str(image_path), 0.0, "错误", 0.0, 0.0


def process_dataset():
    """处理final_dataset"""
    output_path = OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集所有图片
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = []
    
    for char_dir in DATASET_PATH.iterdir():
        if not char_dir.is_dir():
            continue
        
        for ext in image_extensions:
            image_files.extend(char_dir.glob(f"*{ext}"))

    print(f"📁 找到 {len(image_files)} 张图片")
    print(f"📍 使用基于规则的检测方式")

    nsfw_count = 0
    suggestive_count = 0
    safe_count = 0
    error_count = 0
    results = []

    total = len(image_files)
    for i, img_path in enumerate(image_files):
        path, score, label, skin_percent, edge_density = analyze_nsfw(img_path)

        results.append({
            "path": str(img_path),
            "character": img_path.parent.name,
            "score": score,
            "label": label,
            "skin_percent": skin_percent,
            "edge_density": edge_density
        })

        if label == "NSFW":
            nsfw_count += 1
        elif label == "Suggestive":
            suggestive_count += 1
        elif label == "Safe":
            safe_count += 1
        else:
            error_count += 1

        if (i + 1) % 50 == 0:
            print(
                f"   已处理: {i + 1}/{total} | NSFW: {nsfw_count} | Suggestive: {suggestive_count} | Safe: {safe_count}"
            )

    print(f"\n✅ 处理完成!")
    print(f"=" * 60)
    print(f"总图片数: {len(image_files)}")
    print(f"NSFW: {nsfw_count} ({nsfw_count/len(image_files)*100:.1f}%)")
    print(f"Suggestive: {suggestive_count} ({suggestive_count/len(image_files)*100:.1f}%)")
    print(f"Safe: {safe_count} ({safe_count/len(image_files)*100:.1f}%)")
    print(f"错误: {error_count}")

    # 保存结果
    with open(output_path / "nsfw_detection_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_path / "nsfw_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"NSFW检测结果\n")
        f.write(f"=============\n")
        f.write(f"总图片数: {len(image_files)}\n")
        f.write(f"NSFW: {nsfw_count} ({nsfw_count/len(image_files)*100:.1f}%)\n")
        f.write(f"Suggestive: {suggestive_count} ({suggestive_count/len(image_files)*100:.1f}%)\n")
        f.write(f"Safe: {safe_count} ({safe_count/len(image_files)*100:.1f}%)\n")

    # 按角色统计
    char_stats = {}
    for result in results:
        char = result["character"]
        if char not in char_stats:
            char_stats[char] = {"total": 0, "nsfw": 0, "suggestive": 0, "safe": 0}
        char_stats[char]["total"] += 1
        if result["label"] == "NSFW":
            char_stats[char]["nsfw"] += 1
        elif result["label"] == "Suggestive":
            char_stats[char]["suggestive"] += 1
        else:
            char_stats[char]["safe"] += 1

    print(f"\n📊 各角色NSFW分布:")
    print(f"{'角色名称':<30} {'总数':>6} {'NSFW':>6} {'Suggestive':>12} {'Safe':>6}")
    print(f"-" * 70)
    for char, stats in sorted(char_stats.items(), key=lambda x: x[1]["nsfw"], reverse=True):
        print(f"{char:<30} {stats['total']:>6} {stats['nsfw']:>6} {stats['suggestive']:>12} {stats['safe']:>6}")

    print(f"\n📁 检测结果已保存到 {output_path}")

    return {
        "total_images": len(image_files),
        "nsfw_count": nsfw_count,
        "suggestive_count": suggestive_count,
        "safe_count": safe_count,
        "character_stats": char_stats
    }


if __name__ == "__main__":
    process_dataset()