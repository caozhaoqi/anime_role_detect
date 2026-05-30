#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整理蜘蛛图像系统采集的数据到训练数据目录
"""

import os
import sys
import shutil
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger

    logger = get_logger("data_organizer")
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("data_organizer")

# 配置参数
SPIDER_OUTPUT_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/src/data/auto_spider_img"
TRAINING_DATA_DIR = "./data/downloaded_images"

# 角色映射（搜索关键词 -> 英文目录名）
KEYWORD_TO_DIRECTORY = {
    "琮玉": "cong2yu3",
    "迪奥娜": "di2ao4na4",
    "菲谢尔": "fei1xie4er3",
    "符玄": "fu2xuan2",
    "孤明德莲": "gu3ming2di4lian4",
    "黑塔": "hei1ta3",
    "可莉": "ke3li2",
    "科林·维克斯": "ke3lin2_wei1ke4si1",
    "莉莉娅·艾琳": "li4li4ya3_a1lin2",
    "罗莎莉娅·艾琳": "luo2sha1li4ya3_a1lin2",
    "梅比乌斯": "mei2bi3wu3si1",
    "纳西妲": "na4xi1da4",
    "西格温": "xi1ge2wen2",
    "瑶瑶": "yao2yao2",
}


def get_image_count(directory):
    """获取目录中的图像数量"""
    if not os.path.exists(directory):
        return 0
    count = 0
    for file in os.listdir(directory):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            count += 1
    return count


def organize_data():
    """整理数据"""
    # 遍历蜘蛛图像系统的输出目录
    total_moved = 0

    for keyword, target_dir in KEYWORD_TO_DIRECTORY.items():
        # 查找包含关键字的目录
        target_directory = os.path.join(TRAINING_DATA_DIR, target_dir)
        os.makedirs(target_directory, exist_ok=True)

        # 查找蜘蛛系统中包含该关键字的目录
        spider_dirs = []
        for item in os.listdir(SPIDER_OUTPUT_DIR):
            item_path = os.path.join(SPIDER_OUTPUT_DIR, item)
            if os.path.isdir(item_path) and keyword in item:
                spider_dirs.append(item_path)

        if not spider_dirs:
            logger.info(f"未找到包含 '{keyword}' 的目录")
            continue

        # 统计当前目标目录的图像数量
        current_count = get_image_count(target_directory)
        logger.info(f"处理 {keyword}: 当前 {current_count} 张图像")

        # 移动图像
        moved_count = 0
        for spider_dir in spider_dirs:
            for file in os.listdir(spider_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    src_path = os.path.join(spider_dir, file)
                    dest_filename = f"spider_{current_count + moved_count:04d}.jpg"
                    dest_path = os.path.join(target_directory, dest_filename)

                    try:
                        shutil.copy2(src_path, dest_path)
                        moved_count += 1
                        total_moved += 1
                    except Exception as e:
                        logger.error(f"移动失败 {src_path}: {e}")

        if moved_count > 0:
            logger.info(f"  成功移动 {moved_count} 张图像到 {target_dir}")
        else:
            logger.info(f"  没有找到可移动的图像")

    # 最终统计
    logger.info(f"\n总计移动了 {total_moved} 张图像")

    # 显示每个目录的最终图像数量
    logger.info("\n最终数据状态:")
    total_images = 0
    for keyword, target_dir in KEYWORD_TO_DIRECTORY.items():
        target_directory = os.path.join(TRAINING_DATA_DIR, target_dir)
        count = get_image_count(target_directory)
        total_images += count
        logger.info(f"  {keyword}: {count} 张图像")

    logger.info(f"\n总计: {total_images} 张图像")
    logger.info(f"平均每个角色: {total_images / len(KEYWORD_TO_DIRECTORY):.1f} 张图像")


def main():
    logger.info("=" * 60)
    logger.info("开始整理蜘蛛图像系统数据")
    logger.info("=" * 60)

    organize_data()

    logger.info("\n" + "=" * 60)
    logger.info("整理完成")
    logger.info("=" * 60)
    logger.info("接下来的步骤:")
    logger.info(
        "1. 运行数据增强脚本: python scripts/data_collection/enhance_and_balance_dataset.py"
    )
    logger.info("2. 开始训练模型: python scripts/model_training/train_loli_models.py")


if __name__ == "__main__":
    main()
