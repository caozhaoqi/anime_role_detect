#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一重命名角色目录中的图片文件为：角色名_序号.jpg
"""

import os
import logging

# 配置
DATASET_PATH = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("rename_images")


def rename_images_in_role_dir(role_name):
    """重命名单个角色目录中的图片"""
    role_dir = os.path.join(DATASET_PATH, role_name)
    if not os.path.isdir(role_dir):
        logger.warning(f"目录不存在: {role_dir}")
        return 0

    # 获取所有图片文件
    img_files = sorted([f for f in os.listdir(role_dir) if f.lower().endswith(".jpg")])

    if not img_files:
        logger.info(f"{role_name}: 无图片文件")
        return 0

    renamed_count = 0
    for idx, old_name in enumerate(img_files, 1):
        # 检查是否已经是正确的格式
        expected_name = f"{role_name}_{idx:03d}.jpg"
        if old_name == expected_name:
            continue

        old_path = os.path.join(role_dir, old_name)
        new_path = os.path.join(role_dir, expected_name)

        # 如果新文件名已存在，添加后缀
        counter = 1
        while os.path.exists(new_path):
            new_path = os.path.join(role_dir, f"{role_name}_{idx:03d}_{counter}.jpg")
            counter += 1

        try:
            os.rename(old_path, new_path)
            renamed_count += 1
        except Exception as e:
            logger.error(f"重命名失败 {old_name} -> {os.path.basename(new_path)}: {e}")

    logger.info(f"{role_name}: {len(img_files)} 张图片，重命名 {renamed_count} 张")
    return renamed_count


def main():
    logger.info("=" * 60)
    logger.info("开始统一重命名图片文件")
    logger.info("=" * 60)

    # 获取所有角色目录
    role_dirs = sorted(
        [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    )

    total_renamed = 0
    total_images = 0

    for role_name in role_dirs:
        role_dir = os.path.join(DATASET_PATH, role_name)
        img_count = len([f for f in os.listdir(role_dir) if f.lower().endswith(".jpg")])
        total_images += img_count
        total_renamed += rename_images_in_role_dir(role_name)

    logger.info("=" * 60)
    logger.info(f"完成！")
    logger.info(f"总角色数: {len(role_dirs)}")
    logger.info(f"总图片数: {total_images}")
    logger.info(f"重命名数量: {total_renamed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
