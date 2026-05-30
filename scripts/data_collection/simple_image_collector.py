#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单图片采集器 - 使用搜索API采集角色图片
"""

import os
import sys
import json
import urllib.parse
from urllib.request import urlopen, Request
import ssl

# 配置
DATASET_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset"
OUTPUT_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/supplement_images"
TARGET_COUNT = 100  # 目标每个角色的图片数量

# 角色名称映射
ROLE_MAPPING = {
    "Himesaka": "姬坂乃爱",
    "Koshenia": "科谢尼娅",
    "Clara": "克拉拉",
    "Hoshino": "小鸟游星野",
    "Tsukiyo": "月咏",
    "Ichinose": "一之濑",
    "Shirosaki": "白咲",
    "Nagan": "那颜",
    "Paimon": "派蒙",
    "Tanemura": "种村",
    "March": "三月七",
    "Dream!": "梦想",
    "Shakri": "夏克里",
    "Hinaatsu": "阳夏",
    "Fu": "符",
    "Konomori": "小森",
    "Mika": "美嘉",
    "Sayu": "早柚",
    "Suzuran": "铃兰",
}


def get_insufficient_roles(target_count=TARGET_COUNT):
    """获取样本数不足的角色列表"""
    insufficient = []

    if not os.path.exists(DATASET_DIR):
        print(f"❌ 数据集目录不存在: {DATASET_DIR}")
        return insufficient

    for role_name in sorted(os.listdir(DATASET_DIR)):
        role_dir = os.path.join(DATASET_DIR, role_name)
        if not os.path.isdir(role_dir) or role_name.startswith("."):
            continue

        # 统计图片数量
        img_count = len(
            [
                f
                for f in os.listdir(role_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
        )

        if img_count < target_count:
            needed = target_count - img_count
            insufficient.append(
                {
                    "name": role_name,
                    "current": img_count,
                    "needed": needed,
                    "chinese_name": ROLE_MAPPING.get(role_name, role_name),
                }
            )

    insufficient.sort(key=lambda x: x["needed"], reverse=True)
    return insufficient


def search_images(query, count=50):
    """搜索图片URL"""
    urls = []

    try:
        # 使用 DuckDuckGo 搜索API
        encoded_query = urllib.parse.quote(f"{query} 动漫 角色 图片")
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&pretty=1"

        context = ssl._create_unverified_context()
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urlopen(req, context=context, timeout=30)
        data = json.loads(response.read().decode("utf-8"))

        # 提取图片URL
        if "results" in data:
            for result in data["results"][:count]:
                if "image" in result:
                    urls.append(result["image"])

        print(f"🔍 搜索 '{query}' 找到 {len(urls)} 个图片")
        return urls

    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        return urls


def download_image(url, save_path):
    """下载单张图片"""
    try:
        context = ssl._create_unverified_context()
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, context=context, timeout=30) as response:
            with open(save_path, "wb") as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"  ⚠️ 下载失败: {url} - {e}")
        return False


def download_role_images(role_name, chinese_name, needed_count):
    """下载角色图片"""
    print(f"\n📥 开始下载 {role_name} ({chinese_name}) 的图片...")

    # 创建输出目录
    role_dir = os.path.join(OUTPUT_DIR, role_name)
    os.makedirs(role_dir, exist_ok=True)

    # 搜索图片
    urls = search_images(chinese_name, needed_count * 2)  # 获取双倍数量以应对下载失败

    downloaded_count = 0
    for i, url in enumerate(urls):
        if downloaded_count >= needed_count:
            break

        # 生成文件名
        ext = url.split(".")[-1].lower()
        if ext not in ["jpg", "jpeg", "png", "webp"]:
            ext = "jpg"
        filename = f"{role_name}_{i:04d}.{ext}"
        save_path = os.path.join(role_dir, filename)

        if download_image(url, save_path):
            downloaded_count += 1
            print(f"  ✅ 已下载: {downloaded_count}/{needed_count}", end="\r")

    print(f"\n✅ 完成下载 {downloaded_count} 张图片")
    return downloaded_count


def main():
    print("=" * 70)
    print("📊 简单图片采集器")
    print("=" * 70)

    # 获取不足的角色
    insufficient_roles = get_insufficient_roles()

    if not insufficient_roles:
        print("🎉 所有角色样本数都已达到目标！")
        return

    print(f"\n发现 {len(insufficient_roles)} 个角色样本数不足")
    print("-" * 70)

    # 只处理前3个最需要补充的角色
    for role in insufficient_roles[:3]:
        print(f"\n{'='*70}")
        print(f"处理: {role['name']} ({role['chinese_name']})")
        print(f"当前: {role['current']} 张, 需要补充: {role['needed']} 张")
        print("=" * 70)

        download_role_images(role["name"], role["chinese_name"], role["needed"])

    print("\n" + "=" * 70)
    print("✅ 采集任务完成")
    print(f"📁 图片已保存到: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
