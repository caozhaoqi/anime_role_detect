#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量采集所有角色图片脚本
为模型支持的所有74个类别采集测试图片
"""

import os
import sys
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 类别名称到中文的映射（常用角色）
role_chinese_mapping = {
    "Anya": "阿尼亚",
    "Aris": "爱丽丝",
    "Arona": "阿罗娜",
    "Bronya": "布洛妮娅",
    "Chino": "香风智乃",
    "Clara": "克拉拉",
    "Diona": "迪奥娜",
    "Dori": "多莉",
    "Fran": "芙兰",
    "Fu": "芙萱",
    "Herta": "黑塔",
    "Hina": "日奈",
    "Hoshino": "小鸟游星野",
    "Illya": "伊莉雅",
    "Kanna": "康娜",
    "Klee": "可莉",
    "Madoka": "鹿目圆",
    "Nahida": "纳西妲",
    "Nezuko": "祢豆子",
    "Paimon": "派蒙",
    "Ram": "拉姆",
    "Rem": "雷姆",
    "Sayu": "早柚",
    "Seele": "希儿",
    "Shiroko": "白子",
    "Suzuran": "铃兰",
    "Theresa": "德丽莎",
    "Umaru": "小埋",
    "Yaoyao": "瑶瑶",
    "Yoshino": "五河琴里",
}

# 爬虫配置（使用正确的API路径）
SPIDER_API_URL = "http://localhost:33333/api/v1.2.5.260305/sis"
MAX_WORKERS = 1  # 串行采集，避免冲突
TIMEOUT = 60


def get_spider_status():
    """获取爬虫状态"""
    try:
        response = requests.get(f"{SPIDER_API_URL}/spider/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def start_spider(keyword):
    """启动单个关键字爬取"""
    try:
        response = requests.post(
            f"{SPIDER_API_URL}/spider_start/single", params={"key_word": keyword}, timeout=TIMEOUT
        )
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                return True
        return False
    except Exception:
        return False


def wait_for_completion(timeout=120):
    """等待爬虫完成"""
    start_time = time.time()
    last_keyword = ""

    while time.time() - start_time < timeout:
        status = get_spider_status()
        if status:
            data = status.get("data", {})
            is_running = data.get("is_running", False)
            current_keyword = data.get("current_keyword", "")

            if current_keyword and current_keyword != last_keyword:
                print(f"⏳ 爬取中: {current_keyword}", end="\r")
                last_keyword = current_keyword

            if not is_running:
                if current_keyword:
                    print(f"\n✅ {current_keyword} 爬取完成")
                return True

        time.sleep(3)

    print("\n⏰ 超时")
    return False


def get_spider_result(keyword):
    """获取爬取结果"""
    try:
        response = requests.get(
            f"{SPIDER_API_URL}/spider/result", params={"keyword": keyword}, timeout=TIMEOUT
        )
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                return result.get("data", {}).get("urls", [])
        return []
    except Exception:
        return []


def collect_urls_for_role(role_name):
    """为单个角色采集URL"""
    chinese_name = role_chinese_mapping.get(role_name, role_name)
    print(f"📡 正在采集: {role_name} ({chinese_name})")

    # 启动爬虫
    if not start_spider(chinese_name):
        print(f"❌ {role_name} 无法启动爬虫")
        return []

    # 等待完成
    if not wait_for_completion():
        print(f"❌ {role_name} 爬取超时")
        return []

    # 获取结果
    urls = get_spider_result(chinese_name)
    print(f"✅ {role_name} 采集完成: {len(urls)} 个URL")

    return urls


def save_urls_to_file(role_name, urls):
    """保存URL到文件"""
    url_dir = os.path.join(project_root, "data", "img_url")
    os.makedirs(url_dir, exist_ok=True)

    filename = f"{role_name.lower()}_img.txt"
    filepath = os.path.join(url_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for i, url in enumerate(urls, 1):
            f.write(f"{i}→{url}\n")

    return filepath


def download_image(url, save_path):
    """下载单张图片"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, timeout=30, stream=True, headers=headers)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if os.path.getsize(save_path) < 100:
            os.remove(save_path)
            return False

        return True
    except Exception:
        return False


def download_images_for_role(role_name, urls, max_images=10):
    """下载角色图片"""
    download_dir = os.path.join(project_root, "data", "downloaded_images", role_name)
    os.makedirs(download_dir, exist_ok=True)

    # 过滤有效URL
    valid_urls = [u for u in urls if u.lower().endswith(".jpg") or u.lower().endswith(".png")]
    valid_urls = valid_urls[:max_images]

    success = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, url in enumerate(valid_urls):
            filename = f"{role_name.lower()}_{i:03d}.jpg"
            save_path = os.path.join(download_dir, filename)
            futures.append(executor.submit(download_image, url, save_path))

        for future in as_completed(futures):
            if future.result():
                success += 1

    print(f"📥 {role_name} 下载完成: {success}/{len(valid_urls)}")
    return success


def main():
    print("=" * 60)
    print("🚀 开始批量采集所有角色图片")
    print("=" * 60)

    # 加载模型类别列表
    model_dir = os.path.join(
        project_root, "models", "efficientnet_b3_loli_optimized_v2_20260529_133654"
    )
    config_path = os.path.join(model_dir, "training_results.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    class_names = config["class_names"]

    # 只采集有中文映射的角色（提高成功率）
    valid_roles = [r for r in class_names if r in role_chinese_mapping]
    print(f"📋 待采集类别: {len(valid_roles)} 个（有中文映射）")
    print(f"角色列表: {valid_roles}")

    total_collected = 0
    total_failed = 0

    # 串行采集每个角色
    for role_name in valid_roles:
        urls = collect_urls_for_role(role_name)

        if urls:
            save_urls_to_file(role_name, urls)
            success = download_images_for_role(role_name, urls, max_images=10)
            total_collected += success
        else:
            total_failed += 1

        time.sleep(1)  # 间隔

    print("\n" + "=" * 60)
    print("📊 批量采集完成")
    print("=" * 60)
    print(f"总采集图片: {total_collected} 张")
    print(f"失败角色: {total_failed} 个")
    print("=" * 60)

    # 运行基准测试
    print("\n🚀 运行基准测试...")
    benchmark_script = os.path.join(
        project_root, "scripts", "model_evaluation", "benchmark_new_model.py"
    )
    os.system(f"python3 {benchmark_script}")


if __name__ == "__main__":
    main()
