#!/usr/bin/env python3
"""皮肤检测工具 - 检测图片中的皮肤区域"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import shutil

DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "combined_dataset"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "skin_detection"


def detect_skin(image):
    """检测图片中的皮肤区域"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)

    lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)

    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def detect_anime_skin(image):
    """检测动漫风格的皮肤区域 - 使用更严格的阈值"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 35, 80], dtype=np.uint8)
    upper_skin = np.array([25, 170, 230], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)

    lower_skin2 = np.array([165, 35, 80], dtype=np.uint8)
    upper_skin2 = np.array([180, 170, 230], dtype=np.uint8)

    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 500
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            cv2.drawContours(mask, [contour], 0, 0, -1)

    return mask


def analyze_skin(image_path, use_anime_mode=True):
    """分析图片的皮肤区域"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None, 0, 0, 0, False

        height, width = image.shape[:2]
        total_pixels = height * width

        if use_anime_mode:
            mask = detect_anime_skin(image)
        else:
            mask = detect_skin(image)

        skin_pixels = cv2.countNonZero(mask)
        skin_percentage = (skin_pixels / total_pixels) * 100

        has_large_skin_area = skin_percentage > 15

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_skin_regions = len(contours)

        return str(image_path), skin_percentage, num_skin_regions, skin_pixels, has_large_skin_area

    except Exception as e:
        print(f"处理失败 {image_path}: {e}")
        return str(image_path), 0, 0, 0, False


def process_dataset(dataset_path, output_path, use_anime_mode=True, sample_limit=None):
    """处理整个数据集"""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    skin_dir = output_path / "has_skin"
    no_skin_dir = output_path / "no_skin"
    skin_dir.mkdir(exist_ok=True)
    no_skin_dir.mkdir(exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(dataset_path.rglob(f"*{ext}"))

    if sample_limit:
        image_files = image_files[:sample_limit]

    print(f"找到 {len(image_files)} 张图片")
    print(f"使用{'动漫模式' if use_anime_mode else '通用模式'}进行皮肤检测...")

    skin_images = []
    no_skin_images = []
    results = []

    total = len(image_files)
    for i, img_path in enumerate(image_files):
        path, skin_percent, num_regions, skin_pixels, has_skin = analyze_skin(
            img_path, use_anime_mode
        )

        results.append(
            {
                "path": path,
                "skin_percentage": skin_percent,
                "num_skin_regions": num_regions,
                "has_skin": has_skin,
            }
        )

        if has_skin:
            skin_images.append((path, skin_percent, num_regions))
        else:
            no_skin_images.append((path, skin_percent))

        if (i + 1) % 500 == 0:
            print(
                f"已处理: {i + 1}/{total} | 有皮肤: {len(skin_images)} | 无皮肤: {len(no_skin_images)}"
            )

    print(f"\n处理完成!")
    print(f"=" * 60)
    print(f"总图片数: {len(image_files)}")
    print(f"包含皮肤区域: {len(skin_images)} ({len(skin_images)/len(image_files)*100:.1f}%)")
    print(f"无明显皮肤: {len(no_skin_images)} ({len(no_skin_images)/len(image_files)*100:.1f}%)")

    avg_skin_percent = sum(sp for _, sp, _ in skin_images) / len(skin_images) if skin_images else 0
    print(f"\n皮肤区域统计:")
    print(f"  平均皮肤占比: {avg_skin_percent:.1f}%")
    print(f"  最多皮肤区域: {max((n for _, _, n in skin_images), default=0)} 个")

    if skin_images:
        print(f"\n正在复制带皮肤的图片到 {skin_dir}...")
        for path, skin_percent, num_regions in sorted(
            skin_images, key=lambda x: x[1], reverse=True
        ):
            src = Path(path)
            dst = skin_dir / f"skin_{skin_percent:.1f}_{src.name}"
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"复制失败 {src}: {e}")

        print(f"已保存 {len(skin_images)} 张带皮肤的图片")

    with open(output_path / "skin_detection_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_path / "skin_images.txt", "w", encoding="utf-8") as f:
        for path, skin_percent, num_regions in sorted(
            skin_images, key=lambda x: x[1], reverse=True
        ):
            f.write(f"{skin_percent:.1f}%\t{num_regions}\t{path}\n")

    print(f"\n检测结果已保存到 {output_path / 'skin_detection_results.json'}")

    summary = {
        "total_images": len(image_files),
        "skin_images_count": len(skin_images),
        "no_skin_images_count": len(no_skin_images),
        "avg_skin_percentage": avg_skin_percent,
        "mode": "anime" if use_anime_mode else "general",
    }

    with open(output_path / "detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="皮肤检测工具")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="数据集路径")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="输出路径")
    parser.add_argument("--anime", action="store_true", default=True, help="使用动漫模式")
    parser.add_argument("--general", action="store_true", help="使用通用模式")
    parser.add_argument("--sample", type=int, default=None, help="仅处理前N张图片")

    args = parser.parse_args()

    use_anime = args.anime and not args.general

    if not Path(args.dataset).exists():
        print(f"错误: 数据集路径不存在: {args.dataset}")
        sys.exit(1)

    process_dataset(args.dataset, args.output, use_anime, args.sample)
