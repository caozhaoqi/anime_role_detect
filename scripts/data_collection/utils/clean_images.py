#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洗漫画风格图片和非单个角色图片
实用版：结合简单规则和手动检查
"""

import os
import sys
import shutil
import logging

# 配置
DATASET_PATH = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset"
TRASH_PATH = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/trash_images"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("clean_images")


def should_delete_image(image_path):
    """
    判断图片是否应该删除
    使用更严格的规则，减少误删
    """
    try:
        # 获取文件大小
        file_size = os.path.getsize(image_path)

        # 规则1: 文件过小（小于5KB，可能是缩略图或图标）
        if file_size < 5 * 1024:
            return True, f"文件过小 ({file_size} bytes)"

        # 规则2: 文件过大（大于10MB，可能是漫画章节）
        if file_size > 10 * 1024 * 1024:
            return True, f"文件过大 ({file_size} bytes)"

        # 使用PIL检查图片
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size
                ratio = width / height

                # 规则3: 宽高比异常（漫画分镜通常很宽或很窄）
                if ratio > 3.0 or ratio < 0.3:
                    return True, f"宽高比异常 {width}x{height} ({ratio:.2f})"

                # 规则4: 图片尺寸过小
                if width < 300 or height < 300:
                    return True, f"图片尺寸过小 {width}x{height}"

                # 规则5: 检查是否为灰度图（某些漫画扫描件是灰度的）
                if img.mode == "L" or img.mode == "P":
                    # 检查是否真的是灰度漫画（颜色单调）
                    if len(img.getcolors(maxcolors=256)) < 64:
                        return True, f"灰度图片，可能是漫画扫描件"

        except Exception as e:
            logger.warning(f"无法打开图片 {image_path}: {e}")
            return True, f"无法读取图片: {e}"

    except Exception as e:
        logger.warning(f"无法检查文件 {image_path}: {e}")
        return True, f"文件访问错误: {e}"

    return False, ""


def clean_all_roles(dry_run=False, min_images=30):
    """
    清洗所有角色的图片
    :param dry_run: 预览模式
    :param min_images: 保留的最小图片数
    """
    logger.info("=" * 60)
    logger.info("开始清洗漫画风格图片")
    logger.info(f"模式: {'预览' if dry_run else '实际删除'}")
    logger.info(f"每个角色最少保留: {min_images} 张")
    logger.info("=" * 60)

    os.makedirs(TRASH_PATH, exist_ok=True)

    role_dirs = sorted(
        [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    )

    total_kept = 0
    total_deleted = 0

    for role_name in role_dirs:
        role_dir = os.path.join(DATASET_PATH, role_name)
        img_files = sorted([f for f in os.listdir(role_dir) if f.lower().endswith(".jpg")])

        if not img_files:
            continue

        # 找出需要删除的图片
        to_delete = []
        for img_file in img_files:
            img_path = os.path.join(role_dir, img_file)
            should_del, reason = should_delete_image(img_path)
            if should_del:
                to_delete.append((img_file, reason))

        # 确保至少保留min_images张图片
        max_deletable = len(img_files) - min_images
        if len(to_delete) > max_deletable:
            # 只删除前max_deletable个
            to_delete = to_delete[:max_deletable]

        # 执行删除
        kept = len(img_files) - len(to_delete)
        deleted = len(to_delete)

        if deleted > 0:
            logger.info(f"{role_name}: 保留 {kept} 张, 删除 {deleted} 张")
            for img_file, reason in to_delete[:3]:
                logger.info(f"  - {img_file}: {reason}")
            if len(to_delete) > 3:
                logger.info(f"  ... 还有 {len(to_delete) - 3} 张")

            if not dry_run:
                for img_file, _ in to_delete:
                    img_path = os.path.join(role_dir, img_file)
                    trash_role_dir = os.path.join(TRASH_PATH, role_name)
                    os.makedirs(trash_role_dir, exist_ok=True)
                    trash_path = os.path.join(trash_role_dir, img_file)
                    shutil.move(img_path, trash_path)

        total_kept += kept
        total_deleted += deleted

    logger.info("=" * 60)
    logger.info(f"清洗完成！")
    logger.info(f"总保留: {total_kept} 张")
    logger.info(f"总删除: {total_deleted} 张")
    logger.info("=" * 60)


def create_cleaning_report():
    """生成清洗报告，显示需要人工检查的图片"""
    logger.info("=" * 60)
    logger.info("生成清洗报告")
    logger.info("=" * 60)

    report_lines = []

    for role_name in sorted(os.listdir(DATASET_PATH)):
        role_dir = os.path.join(DATASET_PATH, role_name)
        if not os.path.isdir(role_dir):
            continue

        img_files = sorted([f for f in os.listdir(role_dir) if f.lower().endswith(".jpg")])
        if not img_files:
            continue

        # 找出可能有问题的图片
        suspicious = []
        for img_file in img_files:
            img_path = os.path.join(role_dir, img_file)
            should_del, reason = should_delete_image(img_path)
            if should_del:
                suspicious.append((img_file, reason))

        if suspicious:
            report_lines.append(f"\n角色: {role_name}")
            report_lines.append(f"  图片总数: {len(img_files)}")
            report_lines.append(f"  可疑图片: {len(suspicious)}")
            for img_file, reason in suspicious[:5]:
                report_lines.append(f"    - {img_file}: {reason}")
            if len(suspicious) > 5:
                report_lines.append(f"    ... 还有 {len(suspicious) - 5} 张")

    # 保存报告
    report_path = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/cleaning_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"报告已保存到: {report_path}")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--dry-run":
            clean_all_roles(dry_run=True)
        elif sys.argv[1] == "--report":
            create_cleaning_report()
        elif sys.argv[1] == "--help":
            print("用法:")
            print("  python clean_images.py          # 执行清洗")
            print("  python clean_images.py --dry-run # 预览模式")
            print("  python clean_images.py --report  # 生成报告")
            print("  python clean_images.py --help    # 显示帮助")
        else:
            print(f"未知参数: {sys.argv[1]}")
    else:
        # 默认执行清洗
        clean_all_roles(dry_run=False)


if __name__ == "__main__":
    main()
