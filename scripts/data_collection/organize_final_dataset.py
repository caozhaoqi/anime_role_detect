#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计并整理数据集到 final_dataset 目录：
1. 统计当前下载的图片数量
2. 根据角色列表匹配中英文名称
3. 将有效角色图片整理到 final_dataset
4. 统一图片格式为 JPG
"""

import os
import sys
import shutil
import logging
from PIL import Image
from pypinyin import lazy_pinyin, Style

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("organize_final_dataset")

# 路径配置
DOWNLOADED_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/downloaded_images"
FINAL_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset"
ROLE_LIST_FILE = (
    "/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/auto_spider_img/loli-role.txt"
)


def parse_role_list(role_list_path):
    """解析角色列表，建立中英文映射"""
    role_mapping = {}  # 拼音 -> 英文名称
    english_to_chinese = {}  # 英文名称 -> 中文名称
    all_roles = []

    with open(role_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                chinese_name = parts[0]
                game_name = parts[1]
                english_name = parts[2]
                # 生成拼音
                pinyin = "".join(lazy_pinyin(chinese_name, style=Style.TONE3))
                role_mapping[pinyin] = english_name
                english_to_chinese[english_name] = chinese_name
                all_roles.append(
                    {
                        "chinese": chinese_name,
                        "english": english_name,
                        "game": game_name,
                        "pinyin": pinyin,
                    }
                )

    logger.info(f"从角色列表解析出 {len(all_roles)} 个角色")
    return role_mapping, english_to_chinese, all_roles


def convert_to_jpg(image_path, output_path):
    """将图片转换为 JPG 格式"""
    try:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode == "P":
                img = img.convert("RGB")

            img.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        logger.warning(f"转换图片失败 {image_path}: {e}")
        return False


def is_valid_image(file_path):
    """检查是否为有效图片文件"""
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
    name, ext = os.path.splitext(file_path)
    return ext.lower() in valid_extensions and os.path.isfile(file_path)


def organize_dataset():
    """整理数据集到 final_dataset"""
    # 解析角色列表
    role_mapping, english_to_chinese, all_roles = parse_role_list(ROLE_LIST_FILE)

    # 创建最终数据集目录
    os.makedirs(FINAL_DIR, exist_ok=True)

    # 统计变量
    total_images_found = 0
    total_images_copied = 0
    total_images_converted = 0
    total_images_skipped = 0
    roles_found = 0
    roles_not_found = []

    # 遍历所有下载批次
    for batch_dir in os.listdir(DOWNLOADED_DIR):
        batch_path = os.path.join(DOWNLOADED_DIR, batch_dir)
        if not os.path.isdir(batch_path):
            continue

        logger.info(f"处理批次目录: {batch_dir}")

        # 遍历角色目录
        for role_pinyin in os.listdir(batch_path):
            role_dir = os.path.join(batch_path, role_pinyin)
            if not os.path.isdir(role_dir):
                continue

            # 查找匹配的英文名称
            if role_pinyin in role_mapping:
                english_name = role_mapping[role_pinyin]
                chinese_name = english_to_chinese.get(english_name, role_pinyin)

                # 创建目标目录
                target_dir = os.path.join(FINAL_DIR, english_name)
                os.makedirs(target_dir, exist_ok=True)

                # 复制图片
                image_count = 0
                for filename in os.listdir(role_dir):
                    src_path = os.path.join(role_dir, filename)

                    if not is_valid_image(src_path):
                        total_images_skipped += 1
                        continue

                    total_images_found += 1

                    # 生成目标文件名
                    name, ext = os.path.splitext(filename)
                    ext = ext.lower()

                    if ext in (".jpg", ".jpeg"):
                        # 直接复制 JPG
                        dst_path = os.path.join(target_dir, f"{name}.jpg")
                        if not os.path.exists(dst_path):
                            shutil.copy2(src_path, dst_path)
                            image_count += 1
                            total_images_copied += 1
                    elif ext == ".png":
                        # 转换 PNG 到 JPG
                        dst_path = os.path.join(target_dir, f"{name}.jpg")
                        if not os.path.exists(dst_path):
                            if convert_to_jpg(src_path, dst_path):
                                image_count += 1
                                total_images_converted += 1
                    else:
                        # 其他格式尝试转换
                        dst_path = os.path.join(target_dir, f"{name}.jpg")
                        if not os.path.exists(dst_path):
                            if convert_to_jpg(src_path, dst_path):
                                image_count += 1
                                total_images_converted += 1
                            else:
                                total_images_skipped += 1

                if image_count > 0:
                    roles_found += 1
                    logger.info(f"  ✓ [{chinese_name} / {english_name}]: {image_count} 张图片")
            else:
                roles_not_found.append(role_pinyin)

    # 输出统计结果
    logger.info("\n" + "=" * 60)
    logger.info("📊 数据集整理完成")
    logger.info("=" * 60)
    logger.info(f"处理角色数: {roles_found} 个")
    logger.info(f"未匹配角色: {len(roles_not_found)} 个")
    if roles_not_found:
        logger.info(
            f"  未匹配列表: {', '.join(roles_not_found[:5])}{'...' if len(roles_not_found) > 5 else ''}"
        )
    logger.info("-" * 60)
    logger.info(f"发现图片: {total_images_found} 张")
    logger.info(f"直接复制: {total_images_copied} 张")
    logger.info(f"格式转换: {total_images_converted} 张")
    logger.info(f"跳过无效: {total_images_skipped} 张")
    logger.info("-" * 60)

    # 统计最终数据集
    final_stats = []
    total_final = 0
    for role_dir in sorted(os.listdir(FINAL_DIR)):
        role_path = os.path.join(FINAL_DIR, role_dir)
        if os.path.isdir(role_path):
            count = len(
                [f for f in os.listdir(role_path) if os.path.isfile(os.path.join(role_path, f))]
            )
            final_stats.append((role_dir, count))
            total_final += count

    logger.info(f"最终数据集: {total_final} 张图片，分布在 {len(final_stats)} 个角色目录")
    logger.info("=" * 60)

    return {
        "roles_found": roles_found,
        "roles_not_found": roles_not_found,
        "total_images": total_final,
        "per_role": final_stats,
    }


if __name__ == "__main__":
    organize_dataset()
