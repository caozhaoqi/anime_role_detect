#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速数据清洗脚本 - 高效执行核心清洗步骤
"""

import os
import hashlib
import json
import argparse
from PIL import Image
from tqdm import tqdm

# 配置参数
MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_FILE_SIZE_KB = 5
TARGET_SIZE = (512, 512)

CHARACTER_FEATURES = {
    "Tsukiyo": ["blue hair", "long hair", "blue eyes", "school uniform", "serafuku", "calm"],
    "Hina": ["pink hair", "long hair", "pink eyes", "school uniform", "gentle", "smile"],
    "Madoka": ["pink hair", "twintails", "pink eyes", "magical girl", "pink dress"],
    "Homura": ["black hair", "long hair", "purple eyes", "school uniform", "serious"],
    "Sayaka": ["blue hair", "ponytail", "blue eyes", "magical girl", "sword"],
    "Mami": ["blonde hair", "twin drills", "yellow eyes", "magical girl", "rifle"],
    "Kyoko": ["red hair", "ponytail", "orange eyes", "magical girl", "spear"],
    "Arona": ["blue hair", "short hair", "blue eyes", "school uniform", "robot", "halo"],
    "Shiroko": ["white hair", "short hair", "blue eyes", "school uniform", "gun"],
    "Default": ["anime", "character", "portrait"],
}


def get_image_hash(img_path):
    """计算图片的MD5哈希值"""
    try:
        with open(img_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def is_low_quality_fast(img_path):
    """快速检查图片是否为低质量"""
    try:
        file_size_kb = os.path.getsize(img_path) / 1024
        if file_size_kb < MIN_FILE_SIZE_KB:
            return True, f"文件过小 ({file_size_kb:.1f}KB)"

        with Image.open(img_path) as img:
            width, height = img.size
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return True, f"尺寸过小 ({width}x{height})"

            if img.format not in ["JPEG", "PNG", "WEBP"]:
                return True, f"格式不支持 ({img.format})"

            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 5:
                return True, f"宽高比异常"

        return False, ""
    except Exception as e:
        return True, f"无法读取: {str(e)}"


def resize_and_convert_fast(img_path, output_path):
    """快速调整图片尺寸"""
    try:
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)

            new_img = Image.new("RGB", TARGET_SIZE, (255, 255, 255))
            x = (TARGET_SIZE[0] - img.size[0]) // 2
            y = (TARGET_SIZE[1] - img.size[1]) // 2
            new_img.paste(img, (x, y))

            new_img.save(output_path, "JPEG", quality=95)
        return True
    except Exception:
        return False


def generate_tags(role_name):
    """生成标签"""
    if role_name in CHARACTER_FEATURES:
        return list(set(CHARACTER_FEATURES[role_name]))
    return list(set(CHARACTER_FEATURES["Default"]))


def main():
    parser = argparse.ArgumentParser(description="快速数据清洗")
    parser.add_argument("--data-dir", type=str, default="./data/merged_dataset")
    parser.add_argument("--output-dir", type=str, default="./data_cleaned")
    parser.add_argument("--report-file", type=str, default="cleaning_report.json")
    args = parser.parse_args()

    print("🚀 开始快速数据清洗")
    report = {}

    # 步骤1: 删除重复图片
    print("\n📦 步骤1: 删除重复图片")
    hashes = {}
    duplicates = []
    for root, dirs, files in os.walk(args.data_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".webp", ".jpeg")):
                path = os.path.join(root, f)
                h = get_image_hash(path)
                if h:
                    if h in hashes:
                        duplicates.append(path)
                    else:
                        hashes[h] = path

    for p in duplicates:
        os.remove(p)
    report["step1"] = {"duplicates_removed": len(duplicates), "unique_images": len(hashes)}
    print(f"  ✅ 删除 {len(duplicates)} 张重复图片")

    # 步骤2: 过滤低质量图片
    print("\n🔍 步骤2: 过滤低质量图片")
    low_quality = []
    for root, dirs, files in os.walk(args.data_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".webp", ".jpeg")):
                path = os.path.join(root, f)
                is_bad, _ = is_low_quality_fast(path)
                if is_bad:
                    low_quality.append(path)

    for p in low_quality:
        os.remove(p)
    report["step2"] = {"low_quality_removed": len(low_quality)}
    print(f"  ✅ 删除 {len(low_quality)} 张低质量图片")

    # 步骤3: 标准化和标签标注
    print("\n📐 步骤3: 标准化图片和标签标注")
    os.makedirs(args.output_dir, exist_ok=True)
    all_tags = {}
    processed = 0

    for role_name in os.listdir(args.data_dir):
        role_dir = os.path.join(args.data_dir, role_name)
        if not os.path.isdir(role_dir):
            continue

        out_role_dir = os.path.join(args.output_dir, role_name)
        os.makedirs(out_role_dir, exist_ok=True)
        role_tags = {}

        for f in os.listdir(role_dir):
            if f.lower().endswith((".jpg", ".png", ".webp", ".jpeg")):
                in_path = os.path.join(role_dir, f)
                out_path = os.path.join(out_role_dir, os.path.splitext(f)[0] + ".jpg")

                if resize_and_convert_fast(in_path, out_path):
                    processed += 1
                    role_tags[os.path.basename(out_path)] = generate_tags(role_name)

        all_tags[role_name] = role_tags

    # 保存标签
    with open(os.path.join(args.output_dir, "image_tags.json"), "w", encoding="utf-8") as f:
        json.dump(all_tags, f, ensure_ascii=False, indent=2)

    report["step3"] = {"processed_images": processed, "output_dir": args.output_dir}
    print(f"  ✅ 处理 {processed} 张图片")

    # 步骤4: 数据均衡性检查
    print("\n⚖️ 步骤4: 数据均衡性检查")
    role_counts = {}
    for role_name in os.listdir(args.output_dir):
        role_dir = os.path.join(args.output_dir, role_name)
        if os.path.isdir(role_dir):
            cnt = len([f for f in os.listdir(role_dir) if f.endswith(".jpg")])
            role_counts[role_name] = cnt

    report["step4"] = {
        "total_roles": len(role_counts),
        "min_images": min(role_counts.values()),
        "max_images": max(role_counts.values()),
        "avg_images": sum(role_counts.values()) // len(role_counts),
        "role_details": role_counts,
    }
    print(
        f"  📊 {len(role_counts)} 个角色, 平均 {sum(role_counts.values()) // len(role_counts)} 张/角色"
    )

    # 保存报告
    with open(args.report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 清洗完成! 报告: {args.report_file}, 输出: {args.output_dir}")


if __name__ == "__main__":
    main()
