#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用接口采集URL - 采集前先检查已有URL是否足够
"""

import os
import sys
import requests
import time
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 配置
DATASET_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset"
URL_DIR_CN = (
    "/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/spider_image_system/data/img_url"
)
URL_DIR_EN = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/spider_image_system/data/img_url_english"
TARGET_COUNT = 100  # 目标每个角色的图片数量
API_BASE_URL = "http://localhost:33333/api/v1.2.5.260305/sis"

# 角色名称映射
ROLE_MAPPING = {
    "Himesaka": "姬坂乃爱",
    "Hoshino": "小鸟游星野",
    "Tsukiyo": "月咏",
    "Koshenia": "科谢尼娅",
    "Clara": "克拉拉",
    "Shirosaki": "白咲",
    "Nagan": "那颜",
    "March": "三月七",
    "Shakri": "夏克里",
    "Fu": "符",
    "Sayu": "早柚",
}


def get_existing_url_count(role_name):
    """获取角色已有的URL数量"""
    total_urls = 0

    # 检查中文URL目录
    cn_patterns = [f"{role_name}_img.txt", f"*{role_name}*_img.txt"]
    for pattern in cn_patterns:
        files = glob.glob(os.path.join(URL_DIR_CN, pattern))
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                total_urls += len([line for line in fp if line.strip()])

    # 检查英文URL目录
    en_file = os.path.join(URL_DIR_EN, f"{role_name}_img.txt")
    if os.path.exists(en_file):
        with open(en_file, "r", encoding="utf-8") as fp:
            total_urls += len([line for line in fp if line.strip()])

    return total_urls


def get_insufficient_roles():
    """获取样本数不足且URL也不足的角色"""
    insufficient = []

    if not os.path.exists(DATASET_DIR):
        print(f"❌ 数据集目录不存在: {DATASET_DIR}")
        return insufficient

    for role_name in sorted(os.listdir(DATASET_DIR)):
        role_dir = os.path.join(DATASET_DIR, role_name)
        if not os.path.isdir(role_dir) or role_name.startswith("."):
            continue

        img_count = len(
            [
                f
                for f in os.listdir(role_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
        )

        url_count = get_existing_url_count(role_name)
        needed_images = max(0, TARGET_COUNT - img_count)
        needed_urls = max(0, needed_images * 2)  # URL应该是需要的2倍

        if url_count < needed_urls:
            insufficient.append(
                {
                    "name": role_name,
                    "current_images": img_count,
                    "current_urls": url_count,
                    "needed_images": needed_images,
                    "needed_urls": needed_urls,
                    "chinese_name": ROLE_MAPPING.get(role_name, role_name),
                }
            )

    insufficient.sort(key=lambda x: x["needed_urls"], reverse=True)
    return insufficient


def get_spider_status():
    """获取爬虫状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/spider/status")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"❌ 获取状态失败: {e}")
        return None


def start_spider(keyword):
    """启动单个关键字爬取"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/spider_start/single", params={"key_word": keyword}
        )
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                print(f"✅ 开始爬取: {keyword}")
                return True
            else:
                print(f"❌ 爬取失败: {result.get('msg', '未知错误')}")
                return False
        return False
    except Exception as e:
        print(f"❌ 启动爬虫失败: {e}")
        return False


def wait_for_completion(timeout=300):
    """等待爬虫完成"""
    start_time = time.time()
    last_count = 0
    while time.time() - start_time < timeout:
        status = get_spider_status()
        if status:
            data = status.get("data", {})
            is_running = data.get("is_running", False)
            current_keyword = data.get("current_keyword", "")
            current_count = data.get("current_count", 0)

            if current_keyword and current_count != last_count:
                print(f"⏳ 爬取中: {current_keyword} ({current_count} 个URL)", end="\r")
                last_count = current_count

            if not is_running and current_keyword:
                print(f"\n✅ {current_keyword} 爬取完成，获得 {current_count} 个URL")
                return True, current_count
            elif not is_running:
                return True, 0

        time.sleep(3)

    print("\n⏰ 超时")
    return False, 0


def collect_urls(role_name, chinese_name, needed_urls):
    """采集角色URL"""
    print(f"\n🔍 开始采集 {role_name} ({chinese_name}) 的URL...")
    print(f"   需要: {needed_urls} 个URL")

    # 启动爬虫
    if start_spider(chinese_name):
        success, count = wait_for_completion()
        if success:
            print(f"✅ {role_name} 采集完成，获得 {count} 个URL")
            return count > 0
    else:
        print(f"❌ 无法启动爬虫")

    return False


def main():
    print("=" * 70)
    print("🔍 URL采集检查与采集工具")
    print("=" * 70)

    # 检查需要采集的角色
    insufficient_roles = get_insufficient_roles()

    if not insufficient_roles:
        print("\n🎉 所有角色的URL都已足够！")
        return

    print(f"\n发现 {len(insufficient_roles)} 个角色需要采集URL:")
    print("-" * 70)
    print(f"{'角色':<15} {'中文名':<12} {'当前图片':<10} {'当前URL':<10} {'需要URL':<10}")
    print("-" * 70)
    for role in insufficient_roles:
        print(
            f"{role['name']:<15} {role['chinese_name']:<12} {role['current_images']:<10} {role['current_urls']:<10} {role['needed_urls']:<10}"
        )
    print("-" * 70)

    # 确认是否开始采集
    print("\n是否开始采集？ (y/n): ", end="")
    response = input().strip().lower()
    if response != "y":
        print("已取消采集")
        return

    # 开始采集
    for role in insufficient_roles[:5]:  # 先处理前5个
        print(f"\n{'='*70}")
        print(f"处理: {role['name']} ({role['chinese_name']})")
        print(f"当前URL: {role['current_urls']}, 需要: {role['needed_urls']}")
        print("=" * 70)

        success = collect_urls(role["name"], role["chinese_name"], role["needed_urls"])
        if not success:
            print(f"⚠️ {role['name']} 采集失败，继续下一个...")

        time.sleep(2)

    print("\n" + "=" * 70)
    print("✅ 采集任务完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
