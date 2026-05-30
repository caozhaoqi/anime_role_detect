#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 training_dataset 的数据质量和清洗情况
"""
import os
from pathlib import Path
from PIL import Image
import hashlib
from collections import defaultdict

TRAINING_DATASET = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")


def analyze_image_quality(img_path):
    """分析图片质量"""
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            size_kb = img_path.stat().st_size / 1024

            return {
                "valid": True,
                "width": width,
                "height": height,
                "size_kb": size_kb,
                "format": img.format,
                "mode": img.mode,
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def check_duplicates():
    """检查重复文件"""
    hash_to_files = defaultdict(list)

    for img_file in TRAINING_DATASET.rglob("*.jpg"):
        if img_file.is_file():
            try:
                with open(img_file, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                hash_to_files[file_hash].append(img_file)
            except:
                pass

    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    return duplicates


def analyze_dataset():
    """分析整个数据集"""
    print("=" * 80)
    print("📊 training_dataset 数据质量分析")
    print("=" * 80)

    results = {}
    total_images = 0
    total_size = 0
    invalid_images = []
    small_images = []
    large_images = []

    for role_dir in sorted(TRAINING_DATASET.iterdir()):
        if role_dir.is_dir() and not role_dir.name.startswith("."):
            role_name = role_dir.name
            img_files = list(role_dir.glob("*.jpg"))
            img_count = len(img_files)

            if img_count == 0:
                results[role_name] = {
                    "count": 0,
                    "valid": 0,
                    "invalid": 0,
                    "avg_size": 0,
                    "issues": ["无图片"],
                }
                continue

            valid_count = 0
            invalid_count = 0
            role_size = 0
            role_issues = []

            for img_file in img_files:
                quality = analyze_image_quality(img_file)
                total_images += 1

                if quality["valid"]:
                    valid_count += 1
                    size_kb = quality["size_kb"]
                    total_size += size_kb
                    role_size += size_kb

                    # 检查图片尺寸
                    if quality["width"] < 100 or quality["height"] < 100:
                        small_images.append((img_file, quality))
                        role_issues.append(f'小尺寸: {quality["width"]}x{quality["height"]}')

                    # 检查文件大小
                    if size_kb < 10:
                        role_issues.append(f"文件过小: {size_kb:.1f}KB")
                    elif size_kb > 2000:
                        large_images.append((img_file, quality))
                        role_issues.append(f"文件过大: {size_kb:.1f}KB")
                else:
                    invalid_count += 1
                    invalid_images.append((img_file, quality["error"]))
                    role_issues.append(f'损坏: {quality["error"]}')

            results[role_name] = {
                "count": img_count,
                "valid": valid_count,
                "invalid": invalid_count,
                "avg_size": role_size / valid_count if valid_count > 0 else 0,
                "issues": role_issues,
            }

    # 检查重复文件
    print("\n🔍 检查重复文件...")
    duplicates = check_duplicates()

    # 输出结果
    print(f"\n📈 总体统计")
    print("-" * 80)
    print(f"总角色数: {len(results)}")
    print(f"总图片数: {total_images}")
    print(f"总大小: {total_size / 1024:.1f} MB")
    print(f"平均每角色: {total_images / len(results):.1f} 张")
    print(f"平均每张图片: {total_size / total_images:.1f} KB")

    print(f"\n⚠️  问题统计")
    print("-" * 80)
    print(f"损坏图片: {len(invalid_images)} 张")
    print(f"小尺寸图片: {len(small_images)} 张")
    print(f"大文件图片: {len(large_images)} 张")
    print(f"重复文件组: {len(duplicates)} 组")

    # 详细问题列表
    if invalid_images:
        print(f"\n❌ 损坏图片列表:")
        for img_file, error in invalid_images[:10]:
            print(f"  {img_file.relative_to(TRAINING_DATASET)}: {error}")
        if len(invalid_images) > 10:
            print(f"  ... 还有 {len(invalid_images) - 10} 张")

    if small_images:
        print(f"\n📏 小尺寸图片列表:")
        for img_file, quality in small_images[:10]:
            print(
                f'  {img_file.relative_to(TRAINING_DATASET)}: {quality["width"]}x{quality["height"]}'
            )
        if len(small_images) > 10:
            print(f"  ... 还有 {len(small_images) - 10} 张")

    if duplicates:
        print(f"\n🔄 重复文件组:")
        for h, files in list(duplicates.items())[:5]:
            print(f"  MD5: {h[:8]}...")
            for f in files:
                print(f"    {f.relative_to(TRAINING_DATASET)}")
        if len(duplicates) > 5:
            print(f"  ... 还有 {len(duplicates) - 5} 组")

    # 角色详细统计
    print(f"\n📊 角色详细统计")
    print("-" * 80)
    print(f"{'角色名':<20} {'总数':<6} {'有效':<6} {'无效':<6} {'平均大小':<10} {'问题':<20}")
    print("-" * 80)

    for role_name in sorted(results.keys()):
        data = results[role_name]
        issues_str = ", ".join(data["issues"][:2])
        if len(data["issues"]) > 2:
            issues_str += f'...({len(data["issues"])}个)'

        print(
            f"{role_name:<20} {data['count']:<6} {data['valid']:<6} {data['invalid']:<6} {data['avg_size']:<10.1f} {issues_str:<20}"
        )

    # 统计满足100张的角色
    satisfied = sum(1 for r in results.values() if r["count"] >= 100)
    print(f"\n✅ 满足100张的角色: {satisfied}/{len(results)}")
    print(f'❌ 0张角色: {sum(1 for r in results.values() if r["count"] == 0)}')

    return results


if __name__ == "__main__":
    analyze_dataset()
