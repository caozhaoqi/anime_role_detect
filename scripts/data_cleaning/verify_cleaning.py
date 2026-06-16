#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗结果抽样验证脚本

从以下类别各随机抽 N 张，复制到验证目录供人工检查：
  1. good/        → final_dataset 保留的好图
  2. bad_noface/  → 被判定无人脸的图
  3. bad_multiface/ → 被判定多人脸的图
  4. bad_farface/ → 被判定远景的图

输出:
  - data/cleaning_verify/{category}_role_filename.jpg
  - data/cleaning_verify/summary.txt
"""

import random
import shutil
from pathlib import Path

# ====== 配置 ======
SAMPLE_SIZE = 100  # 每类抽样数
RANDOM_SEED = 42

FD_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset")
CLEANED_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/cleaned_fd")
VERIFY_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/cleaning_verify")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

random.seed(RANDOM_SEED)


def collect_images(source_dir: Path) -> list:
    """递归收集目录下所有图片，返回 (路径, 角色名) 列表"""
    images = []
    for role_dir in source_dir.iterdir():
        if not role_dir.is_dir() or role_dir.name.startswith("."):
            continue
        for f in role_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                images.append((f, role_dir.name))
    return images


def sample_and_copy(category: str, images: list, verify_dir: Path):
    """从 images 中随机抽 SAMPLE_SIZE 张，复制到验证目录"""
    if len(images) <= SAMPLE_SIZE:
        sampled = images
        print(f"  ⚠️ {category}: 仅 {len(images)} 张，不足 {SAMPLE_SIZE}")
    else:
        sampled = random.sample(images, SAMPLE_SIZE)

    cat_dir = verify_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    for img_path, role in sampled:
        # 文件名: 角色_原文件名 (避免重名)
        dst_name = f"{role}_{img_path.name}"
        dst = cat_dir / dst_name
        shutil.copy2(str(img_path), str(dst))

    print(f"  ✅ {category}: 已复制 {len(sampled)} 张到 {cat_dir}")
    return sampled


def main():
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("清洗结果抽样验证")
    print(f"每类抽样: {SAMPLE_SIZE} 张")
    print("=" * 60)

    # 1. good — final_dataset 原地保留的
    good_images = collect_images(FD_DIR)
    print(f"\n📂 good (final_dataset): {len(good_images)} 张")
    good_sampled = sample_and_copy("good", good_images, VERIFY_DIR)

    # 2. bad_noface
    noface_dir = CLEANED_DIR / "bad_noface"
    noface_images = collect_images(noface_dir) if noface_dir.exists() else []
    print(f"\n📂 bad_noface: {len(noface_images)} 张")
    noface_sampled = sample_and_copy("bad_noface", noface_images, VERIFY_DIR)

    # 3. bad_multiface
    multiface_dir = CLEANED_DIR / "bad_multiface"
    multiface_images = collect_images(multiface_dir) if multiface_dir.exists() else []
    print(f"\n📂 bad_multiface: {len(multiface_images)} 张")
    multiface_sampled = sample_and_copy("bad_multiface", multiface_images, VERIFY_DIR)

    # 4. bad_farface
    farface_dir = CLEANED_DIR / "bad_farface"
    farface_images = collect_images(farface_dir) if farface_dir.exists() else []
    print(f"\n📂 bad_farface: {len(farface_images)} 张")
    farface_sampled = sample_and_copy("bad_farface", farface_images, VERIFY_DIR)

    # 汇总
    print("\n" + "=" * 60)
    print("抽样完成！验证目录：")
    print(f"  {VERIFY_DIR}/")
    print()
    print(f"  good/         — {len(good_sampled)} 张 (保留的好图)")
    print(f"  bad_noface/   — {len(noface_sampled)} 张 (被判定无人脸)")
    print(f"  bad_multiface/ — {len(multiface_sampled)} 张 (被判定多人脸)")
    print(f"  bad_farface/  — {len(farface_sampled)} 张 (被判定远景)")
    print()
    print("请人工查看后判断检测器准确率。")

    # 保存摘要
    summary = f"""Cleaning Verification - {SAMPLE_SIZE} samples per category
{'='*50}

good (kept):         {len(good_sampled)}/{len(good_images)}
bad_noface:          {len(noface_sampled)}/{len(noface_images)}
bad_multiface:       {len(multiface_sampled)}/{len(multiface_images)}
bad_farface:         {len(farface_sampled)}/{len(farface_images)}
"""
    with open(VERIFY_DIR / "summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()