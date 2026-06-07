#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采集单个角色的URL（支持命令行参数）
"""

import requests
import time
import argparse
import os

API_BASE_URL = "http://localhost:33333/api/v1.2.5.260305/sis"
URL_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/img_url"


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
    while time.time() - start_time < timeout:
        status = get_spider_status()
        if status:
            data = status.get("data", {})
            is_running = data.get("is_running", False)
            current_keyword = data.get("current_keyword", "")
            current_count = data.get("current_count", 0)

            if current_keyword:
                print(f"⏳ 爬取中: {current_keyword} ({current_count} 个URL)", end="\r")

            if not is_running and current_keyword:
                print(f"\n✅ {current_keyword} 爬取完成")
                return True, current_count
            elif not is_running:
                return True, 0

        time.sleep(3)

    print("\n⏰ 超时")
    return False, 0


def save_urls_to_file(role_name, keyword):
    """从API获取URL并保存到文件"""
    try:
        response = requests.get(f"{API_BASE_URL}/spider/result", params={"keyword": keyword})
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                urls = result.get("data", {}).get("urls", [])
                if urls:
                    os.makedirs(URL_DIR, exist_ok=True)
                    url_file = os.path.join(URL_DIR, f"{role_name}_img.txt")
                    with open(url_file, "w", encoding="utf-8") as f:
                        for url in urls:
                            f.write(url + "\n")
                    print(f"✅ URL已保存到: {url_file}")
                    print(f"   共 {len(urls)} 个URL")
                    return True
                else:
                    print(f"⚠️ 未采集到URL")
                    return False
            else:
                print(f"❌ 获取结果失败: {result.get('msg')}")
                return False
        return False
    except Exception as e:
        print(f"❌ 获取URL失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="采集单个角色的URL")
    parser.add_argument("--role", required=True, help="角色英文名")
    parser.add_argument("--chinese", required=True, help="角色中文名（用于搜索）")

    args = parser.parse_args()

    print(f"📡 准备采集: {args.role} ({args.chinese})")

    # 启动爬虫
    if start_spider(args.chinese):
        # 等待完成
        success, count = wait_for_completion()
        if success:
            # 保存URL到文件
            save_urls_to_file(args.role, args.chinese)
    else:
        print(f"❌ 无法启动爬虫")


if __name__ == "__main__":
    main()
