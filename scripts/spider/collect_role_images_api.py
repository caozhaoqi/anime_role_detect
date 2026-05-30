#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色图片采集脚本 - 使用API接口版本
基于 loli-role.txt 名单，通过 spider_image_system API 采集图片
下载到新目录 data/spider_images_v2
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

# 配置
ROLE_LIST_PATH = (
    "/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/auto_spider_img/loli-role.txt"
)
OUTPUT_BASE_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/spider_images_v2"
API_BASE_URL = "http://localhost:33333/api/v1.2.5.260305/sis"
TIMEOUT = 30


def load_role_list(file_path):
    """加载角色名单"""
    roles = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) >= 4:
                roles.append({"cn": parts[0], "anime": parts[1], "en": parts[2], "jp": parts[3]})
            else:
                print(f"⚠️ 格式错误: {line}")
    return roles


def api_get(endpoint):
    """GET 请求"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=TIMEOUT)
        return response.json()
    except Exception as e:
        print(f"❌ API请求失败: {e}")
        return None


def api_post(endpoint, params=None):
    """POST 请求"""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", params=params, timeout=TIMEOUT)
        return response.json()
    except Exception as e:
        print(f"❌ API请求失败: {e}")
        return None


def get_spider_status():
    """获取爬虫状态"""
    return api_get("/spider/status")


def start_spider_single(keyword):
    """开始爬取单个关键字"""
    return api_post("/spider_start/single", {"key_word": keyword})


def stop_spider():
    """停止爬虫"""
    return api_get("/spider_image/stop")


def reset_spider():
    """重置爬虫状态"""
    return api_post("/spider/reset")


def wait_for_spider_complete(timeout=300, check_interval=5):
    """等待爬虫完成"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = get_spider_status()
        if status and status.get("code") == 0:
            data = status.get("data", {})
            if not data.get("is_running", True):
                return True
        time.sleep(check_interval)
    return False


def collect_role_images(role):
    """采集单个角色的图片"""
    keyword = f"{role['cn']} {role['anime']}"

    print(f"\n🔍 搜索关键词: {keyword}")

    # 重置爬虫状态
    reset_spider()
    time.sleep(1)

    # 启动爬虫
    result = start_spider_single(keyword)
    if result and result.get("code") == 0:
        print(f"✅ 爬虫已启动")
    else:
        print(f"❌ 启动爬虫失败: {result}")
        return False

    # 等待爬虫完成（最多等待5分钟）
    if wait_for_spider_complete(timeout=300):
        print(f"✅ 爬取完成")
        return True
    else:
        print(f"⚠️ 爬取超时，停止爬虫")
        stop_spider()
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 角色图片采集脚本 - API接口版本")
    print("=" * 60)

    # 检查API服务是否运行
    print(f"\n🔌 检查API服务: {API_BASE_URL}")
    status = get_spider_status()
    if not status:
        print(f"❌ 无法连接到API服务，请确保服务运行在 {API_BASE_URL}")
        return
    print(f"✅ API服务已连接")

    # 加载角色名单
    print(f"\n📥 加载角色名单: {ROLE_LIST_PATH}")
    roles = load_role_list(ROLE_LIST_PATH)
    print(f"✅ 加载到 {len(roles)} 个角色")

    # 创建输出目录
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    print(f"\n📁 输出目录: {OUTPUT_BASE_DIR}")

    # 统计信息
    total_collected = 0
    failed_roles = []
    success_roles = []

    # 逐个采集角色图片
    for idx, role in enumerate(roles, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(roles)}] 正在采集: {role['cn']}")
        print(f"   英文名: {role['en']}")
        print(f"   所属动漫: {role['anime']}")
        print(f"   日文名: {role['jp']}")
        print("-" * 60)

        try:
            success = collect_role_images(role)

            if success:
                success_roles.append(f"{role['cn']} ({role['anime']})")
                total_collected += 1
                print(f"\n✅ 采集成功")
            else:
                failed_roles.append(role["cn"])
                print(f"\n❌ 采集失败")

        except Exception as e:
            print(f"\n❌ 处理角色 {role['cn']} 时出错: {e}")
            failed_roles.append(role["cn"])

        # 每个角色采集后停止爬虫
        stop_spider()
        reset_spider()

        # 避免请求过快
        time.sleep(2)

    # 输出统计
    print("\n" + "=" * 60)
    print("📊 采集完成!")
    print("=" * 60)
    print(f"总角色数: {len(roles)}")
    print(f"成功采集: {total_collected} 个角色")

    if success_roles:
        print("\n✅ 成功采集的角色:")
        for role in success_roles:
            print(f"   - {role}")

    if failed_roles:
        print(f"\n❌ 采集失败的角色 ({len(failed_roles)} 个):")
        for role in failed_roles:
            print(f"   - {role}")

    print("\n🎉 采集任务完成!")
    print(f"📁 图片保存在: {OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    main()
