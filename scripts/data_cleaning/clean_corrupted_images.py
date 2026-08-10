#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理损坏的图片文件
"""
from pathlib import Path
from PIL import Image
import logging
import sys

# 解码策略统一收口到 src/common/preprocess 唯一真源（导入即继承截断图/像素上限策略）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# 数据清洗需逐张扫描原始大图，个别图可能超过模块级 2 亿像素上限——
# 走显式例外 API，而非裸写 Image.MAX_IMAGE_PIXELS = None 全局关闭防护。
from src.common.preprocess import allow_unlimited_pixels  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")

# 统计
total_checked = 0
corrupted_files = []
valid_files = []

logger.info("=" * 70)
logger.info("开始检查图片文件完整性")
logger.info("=" * 70)
logger.info(f"数据目录: {DATA_DIR}")

# 检查所有图片（在显式例外通道内：临时关闭解压炸弹上限以扫描原始超大图，
# 退出上下文后自动恢复模块级上限，不会污染后续任何解码）
with allow_unlimited_pixels():
    for char_dir in DATA_DIR.iterdir():
        if char_dir.is_dir():
            char_name = char_dir.name
            logger.info(f"检查角色: {char_name}")

            for img_file in char_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    total_checked += 1

                    try:
                        # 尝试打开图片
                        with Image.open(img_file) as img:
                            # 尝试加载图片数据
                            img.load()
                        valid_files.append(img_file)
                    except (OSError, IOError, Image.DecompressionBombError) as e:
                        corrupted_files.append(img_file)
                        logger.warning(f"  发现损坏文件: {img_file.name} - {e}")

logger.info("=" * 70)
logger.info(f"检查完成:")
logger.info(f"  总检查文件数: {total_checked}")
logger.info(f"  有效文件数: {len(valid_files)}")
logger.info(f"  损坏文件数: {len(corrupted_files)}")

if corrupted_files:
    logger.info("\n损坏文件列表:")
    for file in corrupted_files:
        logger.info(f"  {file}")

    # 删除损坏文件
    logger.info("\n开始删除损坏文件...")
    deleted_count = 0
    for file in corrupted_files:
        try:
            file.unlink()
            deleted_count += 1
            logger.info(f"  ✅ 已删除: {file}")
        except Exception as e:
            logger.error(f"  ❌ 删除失败: {file} - {e}")

    logger.info(f"\n删除完成: {deleted_count} 个损坏文件已删除")
else:
    logger.info("\n✅ 没有发现损坏文件")

logger.info("=" * 70)