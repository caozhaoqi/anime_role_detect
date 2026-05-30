#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色图片采集脚本
使用多种数据源采集角色图片
"""

import os
import requests
from PIL import Image
import io
import time
import random
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("role_image_collection.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# 配置参数
ROLE_NAMES = [
    "原神 可莉",
    "原神 纳西妲",
    "崩坏星穹铁道 三月七",
    "崩坏星穹铁道 白露",
    "蔚蓝档案 阿罗娜",
    "明日方舟 阿米娅",
    "绝区零 白月魁",
    "轻音少女 平泽唯",
    "魔法少女小圆 鹿目圆",
    "间谍过家家 阿尼亚",
]

DATA_DIR = "./data/role_training_data"
MAX_IMAGES_PER_ROLE = 20  # 每个角色最多采集的图片数量
TIMEOUT = 15  # 请求超时时间（秒）
DELAY = 2  # 请求延迟时间（秒）
MAX_WORKERS = 3  # 最大并发数

# 创建数据目录
os.makedirs(DATA_DIR, exist_ok=True)


def is_valid_image(content):
    """检查是否为有效图片"""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except:
        return False


def download_image(url, save_dir, role_name, timeout=15):
    """下载单张图片"""
    try:
        headers = {
            "User-Agent": random.choice(
                [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                ]
            ),
            "Referer": "https://www.google.com/",
        }

        response = requests.get(url, headers=headers, timeout=timeout)

        if response.status_code == 200:
            if is_valid_image(response.content):
                # 生成文件名
                url_hash = abs(hash(url)) % 1000000
                filename = f"{url_hash:06d}.jpg"
                filepath = os.path.join(save_dir, filename)

                # 避免重复下载
                if os.path.exists(filepath):
                    return False, "文件已存在"

                # 保存图片
                with open(filepath, "wb") as f:
                    f.write(response.content)

                return True, f"{filename}"
            else:
                return False, "无效图片"
        else:
            return False, f"HTTP {response.status_code}"

    except Exception as e:
        return False, str(e)


def search_images_google(role_name, num=20):
    """使用Google搜索图片"""
    try:
        search_url = f"https://www.google.com/search?q={quote(role_name)}&tbm=isch"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        response = requests.get(search_url, headers=headers, timeout=30)

        if response.status_code == 200:
            # 简单解析HTML获取图片链接
            import re

            # 查找图片链接
            img_links = re.findall(
                r'"(https?://[^"\s]+\.(?:jpg|jpeg|png|gif|webp))"', response.text
            )
            # 去重
            img_links = list(set(img_links))[:num]
            return img_links
        else:
            logger.error(f"Google搜索失败: {response.status_code}")
            return []

    except Exception as e:
        logger.error(f"Google搜索错误: {e}")
        return []


def collect_role_images(role_name):
    """采集单个角色的图片"""
    logger.info(f"开始采集角色: {role_name}")

    # 创建角色目录
    role_dir = os.path.join(DATA_DIR, role_name)
    os.makedirs(role_dir, exist_ok=True)

    # 统计现有图片数量
    existing_images = len(
        [f for f in os.listdir(role_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    )
    if existing_images >= MAX_IMAGES_PER_ROLE:
        logger.info(f"角色 {role_name} 已有 {existing_images} 张图片，跳过采集")
        return []

    # 需要采集的图片数量
    need_images = MAX_IMAGES_PER_ROLE - existing_images

    # 从Google搜索采集图片
    logger.info(f"从Google搜索采集 {role_name} 的图片...")
    img_links = search_images_google(role_name, need_images)

    if not img_links:
        logger.warning(f"未找到 {role_name} 的图片链接")
        return []

    # 下载图片
    success_count = 0
    fail_count = 0
    downloaded_images = []

    for i, url in enumerate(img_links):
        success, message = download_image(url, role_dir, role_name)
        if success:
            success_count += 1
            downloaded_images.append(os.path.join(role_dir, message))
            logger.info(
                f"角色 {role_name}: 下载成功 ({success_count}/{len(img_links)}) - {message}"
            )
        else:
            fail_count += 1
            logger.warning(
                f"角色 {role_name}: 下载失败 ({fail_count}/{len(img_links)}) - {message}"
            )

        # 延迟，避免请求过于频繁
        time.sleep(DELAY)

    logger.info(f"角色 {role_name}: 采集完成，成功 {success_count} 张，失败 {fail_count} 张")
    return downloaded_images


def main():
    """主函数"""
    print("=" * 60)
    print("角色图片采集脚本")
    print("=" * 60)

    print(f"共采集 {len(ROLE_NAMES)} 个角色")
    print(f"数据目录: {DATA_DIR}")
    print(f"每个角色最多采集 {MAX_IMAGES_PER_ROLE} 张图片")
    print()

    # 采集数据
    results = {}

    for role_name in ROLE_NAMES:
        images = collect_role_images(role_name)
        if images:
            results[role_name] = images

        # 角色间延迟
        time.sleep(5)

    # 输出结果
    print("\n" + "=" * 60)
    print("数据采集完成")
    print("=" * 60)
    print(f"成功采集 {len(results)} 个角色的图片")

    total_images = sum(len(images) for images in results.values())
    print(f"共采集 {total_images} 张图片")

    if results:
        print("\n角色图片统计:")
        for character, images in results.items():
            print(f"  {character}: {len(images)} 张")

    print(f"\n数据已保存到: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
