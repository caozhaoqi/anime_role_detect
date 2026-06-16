#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_dataset 基础清洗（不做人脸过滤）

只删除：
  1. 损坏/无法打开的图片
  2. 分辨率 < 256×256 的图片
  3. GIF/动图

保留人脸过滤不做（LBPCascade 漏检严重已弃用）。
"""

import imghdr
import logging
import time
import traceback
from pathlib import Path

from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TRAIN_DIR = Path(
    "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset"
)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MIN_SIZE = 256


def cleanup_training_dataset():
    stats = {"total": 0, "removed_corrupt": 0, "removed_small": 0, "removed_gif": 0, "kept": 0}

    for role_dir in sorted(TRAIN_DIR.iterdir()):
        if not role_dir.is_dir() or role_dir.name.startswith("."):
            continue

        role_name = role_dir.name
        role_images = [f for f in role_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
        role_gifs = [f for f in role_dir.iterdir() if f.is_file() and f.suffix.lower() == ".gif"]
        role_other = [
            f for f in role_dir.iterdir()
            if f.is_file() and f.suffix.lower() not in IMG_EXTS and f.suffix.lower() != ".gif"
        ]

        # 删除 GIF
        for g in role_gifs:
            g.unlink()
            stats["removed_gif"] += 1

        # 删除非标准后缀
        for o in role_other:
            o.unlink()

        role_removed_corrupt = 0
        role_removed_small = 0

        for img_path in role_images:
            stats["total"] += 1

            # 检查是否能打开
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                img_path.unlink()
                role_removed_corrupt += 1
                stats["removed_corrupt"] += 1
                logger.debug(f"  损坏: {role_name}/{img_path.name}")
                continue

            # 检查分辨率
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                img_path.unlink()
                role_removed_corrupt += 1
                stats["removed_corrupt"] += 1
                continue

            min_dim = min(w, h)
            if min_dim < MIN_SIZE:
                img_path.unlink()
                role_removed_small += 1
                stats["removed_small"] += 1
                continue

            stats["kept"] += 1

        if role_removed_corrupt or role_removed_small:
            after = sum(
                1
                for f in role_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTS
            )
            logger.info(
                f"  {role_name}: del_corrupt={role_removed_corrupt} "
                f"del_small={role_removed_small} "
                f"remaining={after}"
            )

    logger.info("\n" + "=" * 50)
    logger.info("training_dataset 基础清洗完成")
    logger.info(f"  总处理: {stats['total']}")
    logger.info(f"  删除损坏: {stats['removed_corrupt']}")
    logger.info(f"  删除 <{MIN_SIZE}px: {stats['removed_small']}")
    logger.info(f"  删除 GIF: {stats['removed_gif']}")
    logger.info(f"  保留: {stats['kept']}")
    logger.info("=" * 50)

    # 统计各类别保留数
    logger.info("\n各类别图片数:")
    for role_dir in sorted(TRAIN_DIR.iterdir()):
        if not role_dir.is_dir():
            continue
        count = sum(1 for f in role_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
        logger.info(f"  {role_dir.name}: {count}")

    return stats


if __name__ == "__main__":
    t0 = time.time()
    cleanup_training_dataset()
    logger.info(f"耗时: {time.time() - t0:.1f}s")