#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调用爬虫API采集数据不足角色的URL
"""
import requests
import time
import json

API_BASE_URL = "http://localhost:33333/sis"

# 需要采集URL的角色
MISSING_ROLES = [
    {"cn_name": "姬坂乃爱", "en_name": "Himesaka", "needed": 79},
    {"cn_name": "科谢尼娅", "en_name": "Koshenia", "needed": 28},
    {"cn_name": "克拉拉", "en_name": "Clara", "needed": 13},
    {"cn_name": "小鸟游星野", "en_name": "Hoshino", "needed": 7},
]


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
            if result.get("code") == 200:
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
    while time.time() - start_time < timeout:
        status = get_spider_status()
        if status:
            is_running = status.get("data", {}).get("is_running", False)
            current_keyword = status.get("data", {}).get("current_keyword", "")
            current_count = status.get("data", {}).get("current_count", 0)

            print(f"⏳ 爬取中: {current_keyword} ({current_count} 个URL)", end="\r")

            if not is_running and current_keyword:
                print(f"\n✅ {current_keyword} 爬取完成")
                return True
            elif not is_running:
                return True

        time.sleep(5)

    print("\n⏰ 超时")
    return False


def main():
    print("📡 开始通过API采集角色URL")
    print("=" * 60)

    # 重置爬虫状态
    try:
        requests.get(f"{API_BASE_URL}/spider/reset")
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ 重置状态失败: {e}")

    # 逐个采集角色
    for role in MISSING_ROLES:
        cn_name = role["cn_name"]
        en_name = role["en_name"]
        needed = role["needed"]

        print(f"\n📥 准备采集: {cn_name} ({en_name})")
        print(f"   需要补充: {needed} 张图片")

        # 检查爬虫状态
        status = get_spider_status()
        if status and not status.get("data", {}).get("is_running", False):
            # 启动爬虫
            if start_spider(cn_name):
                # 等待完成
                wait_for_completion()
            else:
                print(f"❌ 无法启动 {cn_name} 的爬虫")
        else:
            print(f"⚠️ 爬虫正在运行中，跳过 {cn_name}")

        # 等待2秒再处理下一个
        time.sleep(2)

    print("\n" + "=" * 60)
    print("✅ URL采集任务完成")


if __name__ == "__main__":
    main()
