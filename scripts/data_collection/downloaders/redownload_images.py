#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新下载数据不足角色的图片
"""
import os
import sys
import time
import requests
from pathlib import Path

# 需要重新下载的角色
ROLES_TO_DOWNLOAD = [
    {"cn_name": "姬坂乃爱", "en_name": "Himesaka", "pinyin": "ji1ban3nai3ai4", "needed": 79},
    {"cn_name": "科谢尼娅", "en_name": "Koshenia", "pinyin": "ke1xie4ni2ya4", "needed": 28},
    {"cn_name": "克拉拉", "en_name": "Clara", "pinyin": "ke4la1la1", "needed": 13},
    {
        "cn_name": "小鸟游星野",
        "en_name": "Hoshino",
        "pinyin": "xiao3niao3you2xing1ye3",
        "needed": 7,
    },
]

# 配置
SPIDER_DATA_DIR = (
    "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url"
)
DATASET_PATH = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset"
TIMEOUT = 10
RETRIES = 3


def download_image(url, save_path):
    """下载单个图片"""
    for _ in range(RETRIES):
        try:
            response = requests.get(url, timeout=TIMEOUT, stream=True)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return True
        except Exception as e:
            time.sleep(1)
    return False


def download_role_images(role):
    """下载某个角色的图片"""
    pinyin = role["pinyin"]
    en_name = role["en_name"]
    cn_name = role["cn_name"]
    needed = role["needed"]

    # 读取URL文件
    url_file = Path(SPIDER_DATA_DIR) / f"{pinyin}_img.txt"
    if not url_file.exists():
        print(f"❌ {cn_name}: 未找到URL文件")
        return 0

    with open(url_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print(f"❌ {cn_name}: URL列表为空")
        return 0

    # 创建保存目录
    save_dir = Path(DATASET_PATH) / en_name
    save_dir.mkdir(exist_ok=True)

    # 获取已有的图片数量
    existing_imgs = [f for f in os.listdir(save_dir) if f.lower().endswith(".jpg")]
    current_count = len(existing_imgs)
    print(f"📥 {cn_name}: 当前 {current_count} 张, 需要补充 {needed} 张")

    # 下载图片
    downloaded = 0
    skipped = 0
    failed = 0

    for i, url in enumerate(urls):
        if downloaded >= needed:
            break

        # 生成文件名
        filename = f"{en_name}_{i:04d}.jpg"
        save_path = save_dir / filename

        # 跳过已存在的文件
        if save_path.exists():
            skipped += 1
            continue

        # 下载图片
        if download_image(url, save_path):
            downloaded += 1
            print(f"   ✅ [{downloaded}/{needed}] {filename}")
        else:
            failed += 1

    print(f"📊 {cn_name}: 成功 {downloaded}, 跳过 {skipped}, 失败 {failed}")
    return downloaded


def main():
    print("🚀 开始重新下载角色图片")
    print("=" * 60)

    total_downloaded = 0

    for role in ROLES_TO_DOWNLOAD:
        print(f"\n------ {role['cn_name']} ({role['en_name']}) ------")
        downloaded = download_role_images(role)
        total_downloaded += downloaded
        time.sleep(2)  # 避免请求过快

    print("\n" + "=" * 60)
    print(f"✅ 下载完成，共下载 {total_downloaded} 张图片")

    # 更新统计
    print("\n📊 更新后统计:")
    print("-" * 50)
    for role in ROLES_TO_DOWNLOAD:
        save_dir = Path(DATASET_PATH) / role["en_name"]
        count = len([f for f in os.listdir(save_dir) if f.lower().endswith(".jpg")])
        print(f"{role['cn_name']:<12} {count} 张")


if __name__ == "__main__":
    main()
