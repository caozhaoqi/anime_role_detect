#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析并补充角色数据到每个角色至少100张
"""
import os
import shutil
from pathlib import Path
import random

COMBINED_DATASET = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset")
SOURCE_DIRS = [
    Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset"),
    Path(
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/detection/multi_face_detected_anime/multi_face"
    ),
    Path(
        "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/detection/multi_face_detected/multi_face"
    ),
    Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/nsfw/skin_detection/has_skin"),
]


def analyze_current_data():
    """分析当前数据集"""
    print("=== 分析当前数据集 ===\n")

    results = []
    for role_dir in sorted(COMBINED_DATASET.iterdir()):
        if role_dir.is_dir() and not role_dir.name.startswith("."):
            img_count = len(list(role_dir.glob("*.jpg")))
            results.append((role_dir.name, img_count))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"{'角色名':<25} {'图片数':<8} {'状态':<10}")
    print("-" * 50)

    insufficient = []
    for role_name, count in results:
        status = "✅ 满足" if count >= 100 else "❌ 不足"
        print(f"{role_name:<25} {count:<8} {status:<10}")
        if count < 100:
            insufficient.append((role_name, count, 100 - count))

    print("\n" + "=" * 50)
    print(f"总角色数: {len(results)}")
    print(f"满足100张的角色: {len(results) - len(insufficient)}")
    print(f"不足100张的角色: {len(insufficient)}")
    print(f"总图片数: {sum(r[1] for r in results)}")

    return results, insufficient


def find_source_images(role_name):
    """从源目录查找角色的图片"""
    found_images = []

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue

        # 查找所有jpg文件
        for img_file in source_dir.rglob("*.jpg"):
            # 检查文件名是否包含角色名（不区分大小写）
            if role_name.lower() in img_file.name.lower():
                found_images.append(img_file)

    return found_images


def supplement_data(insufficient_roles):
    """补充不足的角色数据"""
    print("\n=== 开始补充数据 ===\n")

    for role_name, current_count, need_count in insufficient_roles:
        print(f"\n处理角色: {role_name} (当前: {current_count}, 需要: {need_count})")

        # 查找源图片
        source_images = find_source_images(role_name)

        if not source_images:
            print(f"  ⚠️  未找到 {role_name} 的源图片")
            continue

        print(f"  找到 {len(source_images)} 张源图片")

        # 随机选择需要的数量
        if len(source_images) > need_count:
            source_images = random.sample(source_images, need_count)

        # 复制图片
        target_dir = COMBINED_DATASET / role_name
        target_dir.mkdir(exist_ok=True)

        existing_images = set([f.name for f in target_dir.glob("*.jpg")])
        copied_count = 0

        for src_img in source_images:
            # 生成新文件名避免冲突
            new_name = src_img.name
            counter = 1
            while new_name in existing_images:
                name_part = src_img.stem
                ext_part = src_img.suffix
                new_name = f"{name_part}_{counter}{ext_part}"
                counter += 1

            try:
                shutil.copy2(src_img, target_dir / new_name)
                copied_count += 1
            except Exception as e:
                print(f"  ❌ 复制失败 {src_img.name}: {e}")

        print(f"  ✅ 成功复制 {copied_count} 张图片")

        # 更新统计
        new_count = len(list(target_dir.glob("*.jpg")))
        print(f"  📊 更新后数量: {new_count} 张")


def verify_data():
    """验证最终数据集"""
    print("\n=== 验证最终数据集 ===\n")

    results = []
    for role_dir in sorted(COMBINED_DATASET.iterdir()):
        if role_dir.is_dir() and not role_dir.name.startswith("."):
            img_count = len(list(role_dir.glob("*.jpg")))
            results.append((role_dir.name, img_count))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"{'角色名':<25} {'图片数':<8} {'状态':<10}")
    print("-" * 50)

    insufficient = []
    for role_name, count in results:
        status = "✅ 满足" if count >= 100 else "❌ 不足"
        print(f"{role_name:<25} {count:<8} {status:<10}")
        if count < 100:
            insufficient.append((role_name, count, 100 - count))

    print("\n" + "=" * 50)
    print(f"总角色数: {len(results)}")
    print(f"满足100张的角色: {len(results) - len(insufficient)}")
    print(f"不足100张的角色: {len(insufficient)}")
    print(f"总图片数: {sum(r[1] for r in results)}")

    if insufficient:
        print("\n⚠️  以下角色仍需补充:")
        for role_name, count, need in insufficient:
            print(f"  {role_name}: {count}张 → 需要补充{need}张")
    else:
        print("\n✅ 所有角色均已满足100张图片要求！")

    return results, insufficient


if __name__ == "__main__":
    print("🚀 开始数据补充任务\n")

    # 分析当前数据
    results, insufficient = analyze_current_data()

    if not insufficient:
        print("\n✅ 所有角色均已满足100张图片要求，无需补充！")
    else:
        # 补充数据
        supplement_data(insufficient)

        # 验证结果
        verify_data()

    print("\n🎉 任务完成！")
