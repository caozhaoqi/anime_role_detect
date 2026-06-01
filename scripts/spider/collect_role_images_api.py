#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色图片采集脚本 - 使用API接口版本
基于 loli-role.txt 名单，通过 spider_image_system API 采集图片
下载到新目录 data/spider_images_v2
支持飞书通知推送采集进度
"""

import os
import sys
import json
import time
import requests
import logging
from pathlib import Path
from loguru import logger

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from pypinyin import lazy_pinyin, Style

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "notification_config.json")

ROLE_LIST_PATH = (
    "/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/auto_spider_img/loli-role.txt"
)
OUTPUT_BASE_DIR = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/spider_images_v2"
API_BASE_URL = "http://localhost:33333/api/v1.2.5.260305/sis"
TIMEOUT = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NOTIFICATION_AVAILABLE = False
notification_manager = None


def init_notification():
    """初始化通知服务"""
    global NOTIFICATION_AVAILABLE, notification_manager
    try:
        from src.services.notification_service import get_notification_manager

        notification_manager = get_notification_manager()
        NOTIFICATION_AVAILABLE = True
        logger.info("通知服务初始化成功")
        return True
    except ImportError as e:
        logger.warning(f"通知服务未找到: {e}")
        return False


def send_notification(message, title=None, level="info"):
    """发送飞书通知"""
    if not NOTIFICATION_AVAILABLE or not notification_manager:
        logger.debug(f"通知不可用: {message}")
        return False
    try:
        return notification_manager.send(message, title, level)
    except Exception as e:
        logger.warning(f"发送通知失败: {e}")
        return False


def send_progress(role, status, idx, total, message=""):
    """发送采集进度通知"""
    status_emoji = {"running": "🔄", "completed": "✅", "error": "❌", "skipped": "⏭️", "starting": "🚀"}
    status_text = {
        "running": "采集中",
        "completed": "采集完成",
        "error": "采集失败",
        "skipped": "已跳过",
        "starting": "开始采集",
    }

    emoji = status_emoji.get(status, "📦")
    text = status_text.get(status, "未知状态")

    title = f"{emoji} 角色图片采集进度"
    content = f"**角色**: {role['cn']} ({role['anime']})\n"
    content += f"**状态**: {text}\n"
    content += f"**进度**: [{idx}/{total}]\n"
    content += f"**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    if message:
        content += f"**消息**: {message}"

    level = "success" if status == "completed" else "error" if status == "error" else "info"
    return send_notification(content, title, level)


def send_summary(success_count, failed_count, total, success_roles, failed_roles):
    """发送采集完成汇总通知"""
    title = "📊 角色图片采集完成"
    content = f"""**采集任务完成**

**统计信息**:
- 总角色数: {total}
- ✅ 成功: {success_count} 个
- ❌ 失败: {failed_count} 个

**成功采集的角色**:
"""
    for role in success_roles[:10]:
        content += f"- {role}\n"
    if len(success_roles) > 10:
        content += f"... 还有 {len(success_roles) - 10} 个\n"

    if failed_roles:
        content += f"\n**失败的角色** ({len(failed_roles)} 个):\n"
        for role in failed_roles[:5]:
            content += f"- {role}\n"
        if len(failed_roles) > 5:
            content += f"... 还有 {len(failed_roles) - 5} 个\n"

    content += f"\n**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    content += f"\n**保存位置**: {OUTPUT_BASE_DIR}"

    return send_notification(content, title, level="success")


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

    # 初始化通知服务
    init_notification()

    # 发送开始通知
    send_notification(
        f"🚀 角色图片采集任务已启动\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "采集任务开始",
        level="info",
    )

    # 检查API服务是否运行
    print(f"\n🔌 检查API服务: {API_BASE_URL}")
    status = get_spider_status()
    if not status:
        print(f"❌ 无法连接到API服务，请确保服务运行在 {API_BASE_URL}")
        send_notification(f"❌ 无法连接到API服务: {API_BASE_URL}", "采集任务失败", level="error")
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

        # 发送开始采集通知
        send_progress(role, "starting", idx, len(roles))

        try:
            success = collect_role_images(role)

            if success:
                success_roles.append(f"{role['cn']} ({role['anime']})")
                total_collected += 1
                print(f"\n✅ 采集成功")
                # 发送成功通知
                send_progress(role, "completed", idx, len(roles))
            else:
                failed_roles.append(role["cn"])
                print(f"\n❌ 采集失败")
                # 发送失败通知
                send_progress(role, "error", idx, len(roles), "采集失败")

        except Exception as e:
            print(f"\n❌ 处理角色 {role['cn']} 时出错: {e}")
            failed_roles.append(role["cn"])
            # 发送错误通知
            send_progress(role, "error", idx, len(roles), str(e))

        # 每个角色采集后停止爬虫
        stop_spider()
        reset_spider()

        # 避免请求过快
        time.sleep(2)

    # 发送汇总通知
    send_summary(total_collected, len(failed_roles), len(roles), success_roles, failed_roles)

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
