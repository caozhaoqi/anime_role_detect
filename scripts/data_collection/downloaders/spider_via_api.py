#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过API接口爬取缺失角色的URL - 控制采集数量
支持WebSocket实时进度推送
支持飞书通知推送采集进度（使用统一通知服务）
"""

import os
import sys
import time
import requests
import logging
import json
import threading
from loguru import logger

# 添加项目根目录到Python路径
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

# 先定义 WEBSOCKET_AVAILABLE
try:
    import websocket

    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("websocket-client 未安装，将使用轮询方式获取进度")

from pypinyin import lazy_pinyin, Style

# 加载飞书配置（与成功发送通知的脚本相同方式）
config_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "notification_config.json",
)
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        notification_config = json.load(f)

    # 设置飞书通知环境变量
    os.environ["NOTIFICATION_ENABLED"] = "true"
    os.environ["NOTIFICATION_PLATFORM"] = notification_config["platform"]
    os.environ["FEISHU_APP_ID"] = notification_config["feishu"]["app_id"]
    os.environ["FEISHU_APP_SECRET"] = notification_config["feishu"]["app_secret"]
    os.environ["FEISHU_RECEIVE_ID"] = notification_config["feishu"]["receive_id"]
    os.environ["FEISHU_RECEIVE_ID_TYPE"] = notification_config["feishu"]["receive_id_type"]
    logger.info(f"已加载通知配置: {config_path}")
else:
    logger.warning(f"未找到通知配置文件: {config_path}")

# 导入统一通知服务
try:
    from src.services.notification_service import get_notification_manager, send_notification

    NOTIFICATION_AVAILABLE = True
except ImportError as e:
    NOTIFICATION_AVAILABLE = False
    logging.warning(f"通知服务未找到，将使用内置的飞书通知功能: {e}")

# 配置
API_BASE_URL = "http://localhost:33333/api/v1.2.5.260305"
WS_BASE_URL = "ws://localhost:33333/api/v1.2.5.260305"
ROLE_LIST_FILE = (
    "/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/auto_spider_img/loli-role-new.txt"
)
URL_DIR = (
    "/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/spider_image_system/data/img_url"
)

# 爬取配置
MAX_URLS_PER_ROLE = 200  # 每个角色最多采集200个URL（加快采集速度）
MIN_URLS_PER_ROLE = 100  # 每个角色至少采集100个URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 进度状态
current_progress = {"keyword": "", "current_count": 0, "status": "idle", "page": 0, "message": ""}
progress_event = threading.Event()

# 通知管理器实例
notification_manager = None


def init_notification():
    """初始化通知服务"""
    global notification_manager
    if NOTIFICATION_AVAILABLE:
        try:
            notification_manager = get_notification_manager()
            logger.info("通知服务初始化成功")
            return True
        except Exception as e:
            logger.warning(f"通知服务初始化失败: {e}")
            return False
    return False


def send_spider_notification(message, title=None, level="info"):
    """
    发送爬虫通知（统一接口）
    :param message: 消息内容
    :param title: 消息标题
    :param level: 消息级别 (info, warning, error, success)
    """
    if notification_manager:
        try:
            return notification_manager.send(message, title, level)
        except Exception as e:
            logger.warning(f"发送通知失败: {e}")
            return False
    return False


def send_spider_progress(role, status, count, total, message=""):
    """
    发送采集进度通知
    :param role: 角色名称
    :param status: 状态 (running/completed/error/skipped)
    :param count: 当前数量
    :param total: 总数
    :param message: 附加消息
    """
    status_emoji = {"running": "🔄", "completed": "✅", "error": "❌", "skipped": "⏭️"}

    status_text = {
        "running": "采集进行中",
        "completed": "采集完成",
        "error": "采集失败",
        "skipped": "已跳过",
    }

    emoji = status_emoji.get(status, "📦")
    text = status_text.get(status, "未知状态")

    title = f"{emoji} 角色采集状态更新"
    content = f"**角色**: {role}\n**状态**: {text}\n**进度**: {count}/{total}\n**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"

    if message:
        content += f"\n**消息**: {message}"

    send_spider_notification(
        content,
        title,
        level="success" if status == "completed" else "error" if status == "error" else "info",
    )


def get_all_roles():
    """获取完整角色列表"""
    roles = []
    with open(ROLE_LIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(" ")
                chinese_name = parts[0]
                roles.append(chinese_name)
    return roles


def get_missing_roles():
    """获取需要爬取的角色列表"""
    all_roles = get_all_roles()
    existing_files = set()

    if os.path.exists(URL_DIR):
        for filename in os.listdir(URL_DIR):
            if filename.endswith("_img.txt"):
                role_pinyin = filename.replace("_img.txt", "")
                existing_files.add(role_pinyin)

    missing = []
    for role in all_roles:
        pinyin = "".join(lazy_pinyin(role, style=Style.TONE3))
        if f"{pinyin}_img.txt" not in existing_files:
            missing.append(role)

    return missing, len(all_roles)


def check_url_count(role):
    """检查角色已有的URL数量"""
    pinyin = "".join(lazy_pinyin(role, style=Style.TONE3))
    url_file = os.path.join(URL_DIR, f"{pinyin}_img.txt")

    if os.path.exists(url_file):
        with open(url_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return len([l for l in lines if l.strip()])
    return 0


def start_spider_via_api(keyword):
    """通过API接口启动爬虫"""
    try:
        url = f"{API_BASE_URL}/sis/spider_start/single"
        params = {"key_word": keyword}
        response = requests.post(url, params=params, timeout=60)
        if response.status_code == 200:
            result = response.json()
            # 检查成功条件（支持多种成功响应格式）
            if result.get("code") == 0 or (
                result.get("msg") and "success" in result.get("msg").lower()
            ):
                return True, "success"
            else:
                return False, result.get("msg", "unknown error")
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)


def check_api_status():
    """检查API服务是否可用"""
    try:
        response = requests.get(f"{API_BASE_URL}/sis/spider_image/config", timeout=5)
        return response.status_code == 200
    except Exception as e:
        return False


def get_spider_status():
    """获取当前爬虫状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/sis/spider/status", timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                return result.get("data", {})
        return {}
    except Exception as e:
        logger.debug(f"获取状态失败: {e}")
        return {}


def is_spider_busy():
    """检查爬虫是否正在运行"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/sis/spider_start/single", params={"key_word": ""}, timeout=10
        )
        result = response.json()

        msg = result.get("msg", "").strip()
        data = result.get("data", "").strip()
        code = result.get("code", -1)

        # 如果返回"操作进行中"，说明爬虫正在运行
        if "操作进行中" in msg or "操作进行中" in data:
            return True
        # 如果返回"关键词不能为空"，说明服务空闲（拒绝空关键词）
        elif "关键词不能为空" in data:
            return False
        # 如果code=0，说明服务空闲（可以接受请求）
        elif code == 0:
            return False

        # 默认认为忙碌
        logger.debug(f"未知响应: {result}")
        return True
    except Exception as e:
        logger.debug(f"检查状态失败: {e}")
        return True  # 出错时默认认为忙碌


def on_ws_message(ws, message):
    """WebSocket消息处理"""
    global current_progress
    try:
        data = json.loads(message)
        if data.get("type") == "spider_progress":
            current_progress = {
                "keyword": data.get("keyword", ""),
                "current_count": data.get("current_count", 0),
                "status": data.get("status", "idle"),
                "page": data.get("page", 0),
                "message": data.get("message", ""),
            }
            actual_count = (
                check_url_count(current_progress["keyword"]) if current_progress["keyword"] else 0
            )
            logger.info(
                f"进度更新 [{current_progress['keyword']}]: {actual_count} URLs - {current_progress['message']}"
            )

            # 如果状态变为完成或错误，触发事件
            if current_progress["status"] in ["completed", "error"]:
                progress_event.set()
    except Exception as e:
        logger.warning(f"WebSocket消息解析失败: {e}")


def on_ws_error(ws, error):
    """WebSocket错误处理"""
    logger.warning(f"WebSocket错误: {error}")


def on_ws_close(ws, close_status_code, close_msg):
    """WebSocket关闭处理"""
    logger.debug("WebSocket连接已关闭")


def on_ws_open(ws):
    """WebSocket连接建立"""
    logger.debug("WebSocket连接已建立")


def start_websocket_listener():
    """启动WebSocket监听器"""
    if not WEBSOCKET_AVAILABLE:
        return None

    ws_url = f"{WS_BASE_URL}/sis/spider/progress/ws"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_ws_open,
        on_message=on_ws_message,
        on_error=on_ws_error,
        on_close=on_ws_close,
    )

    # 在后台线程运行
    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()

    # 等待连接建立
    time.sleep(2)
    return ws


def wait_for_spider_completion(timeout=300):
    """等待当前爬虫任务完成，显示详细状态"""
    global current_progress

    start_time = time.time()
    logger.debug("等待爬虫任务完成...")

    # 先检查是否已经空闲
    if not is_spider_busy():
        logger.debug("服务已经空闲")
        return True

    # 重置进度状态
    current_progress["status"] = "running"
    progress_event.clear()

    if WEBSOCKET_AVAILABLE:
        # 使用WebSocket等待（同时轮询获取状态）
        try:
            while time.time() - start_time < timeout:
                if progress_event.wait(timeout=15):
                    logger.debug("通过WebSocket检测到任务完成")
                    return True

                # 获取并显示当前状态（使用实际文件中的URL数量）
                file_count = (
                    check_url_count(current_progress["keyword"])
                    if current_progress["keyword"]
                    else 0
                )
                status = get_spider_status()
                if status:
                    keyword = status.get("current_keyword", "未知")
                    max_urls = status.get("max_urls", MAX_URLS_PER_ROLE)
                    logger.info(f"等待中... [{keyword}] 进度: {file_count}/{max_urls} URLs")

            logger.warning("等待超时")
            return False
        except Exception as e:
            logger.warning(f"WebSocket等待失败: {e}")
            return False
    else:
        # 使用轮询方式
        while time.time() - start_time < timeout:
            if not is_spider_busy():
                logger.debug("通过轮询检测到任务完成")
                return True
            # 输出当前进度
            current_count = (
                check_url_count(current_progress["keyword"]) if current_progress["keyword"] else 0
            )
            logger.info(f"等待中... 当前进度: {current_count} URLs")
            time.sleep(15)

        logger.warning("等待超时")
        return False


def spider_via_api():
    """主函数：通过API爬取缺失角色"""
    global current_progress

    # 初始化通知服务
    init_notification()

    logger.info("检查API服务状态...")
    if not check_api_status():
        logger.error("API服务不可用，请先启动爬虫服务！")
        send_spider_notification(
            "❌ API服务不可用，请先启动爬虫服务！", "爬虫服务异常", level="error"
        )
        return

    # 启动WebSocket监听器
    ws = None
    if WEBSOCKET_AVAILABLE:
        logger.info("启动WebSocket进度监听...")
        ws = start_websocket_listener()

    logger.info("获取缺失角色列表...")
    missing_roles, total = get_missing_roles()

    if not missing_roles:
        logger.info("所有角色的URL都已采集完成！")
        send_spider_notification("✅ 所有角色的URL都已采集完成！", "采集任务完成", level="success")
        return

    logger.info(f"总角色数: {total}")
    logger.info(f"已采集: {total - len(missing_roles)}")
    logger.info(f"缺少URL: {len(missing_roles)}")
    logger.info(
        f"缺失角色: {', '.join(missing_roles[:10])}{'...' if len(missing_roles) > 10 else ''}"
    )
    logger.info(f"采集配置: 每个角色最多 {MAX_URLS_PER_ROLE} 个URL")

    # 发送开始采集通知
    send_spider_notification(
        f"**🚀 角色URL采集任务开始**\n\n总角色数: {total}\n待采集: {len(missing_roles)}\n配置: 每个角色最多 {MAX_URLS_PER_ROLE} 个URL\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "角色URL采集任务开始",
        level="info",
    )

    # 逐个爬取
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for i, role in enumerate(missing_roles, 1):
        # 检查是否已有足够的URL
        current_count = check_url_count(role)
        if current_count >= MIN_URLS_PER_ROLE:
            logger.info(f"[{i}/{len(missing_roles)}] ⏭️ 跳过 {role} (已有 {current_count} 个URL)")
            skipped_count += 1
            # 发送跳过通知
            send_spider_progress(
                role, "skipped", current_count, MAX_URLS_PER_ROLE, f"已有足够URL，跳过采集"
            )
            continue

        # 等待爬虫服务空闲
        logger.info(f"[{i}/{len(missing_roles)}] 等待服务空闲...")
        wait_for_spider_completion(300)

        # 设置当前角色
        current_progress["keyword"] = role

        logger.info(f"[{i}/{len(missing_roles)}] 开始爬取角色: {role}")

        # 发送开始采集通知
        send_spider_progress(
            role, "running", 0, MAX_URLS_PER_ROLE, f"开始第 {i}/{len(missing_roles)} 个角色采集"
        )

        success, msg = start_spider_via_api(role)
        if success:
            logger.info(f"✓ 成功启动爬虫: {role}")

            # 等待爬取完成（最多等待3分钟）
            logger.info(f"  等待爬取完成...")
            wait_for_spider_completion(180)

            # 检查采集结果
            final_count = check_url_count(role)
            status = current_progress.get("status", "completed")

            if status == "completed":
                logger.info(f"  ✅ 采集完成: 获取 {final_count} 个URL")
                success_count += 1
                # 发送完成通知
                send_spider_progress(role, "completed", final_count, MAX_URLS_PER_ROLE)
            elif status == "error":
                error_msg = current_progress.get("message", "未知错误")
                logger.error(f"  ❌ 采集错误: {error_msg}")
                failed_count += 1
                # 发送错误通知
                send_spider_progress(role, "error", final_count, MAX_URLS_PER_ROLE, error_msg)
            else:
                logger.warning(f"  ⚠️ 采集超时: 获取 {final_count} 个URL")
                success_count += 1
                # 发送超时通知
                send_spider_progress(
                    role, "completed", final_count, MAX_URLS_PER_ROLE, "采集超时，已获取部分URL"
                )
        else:
            logger.error(f"✗ 启动爬虫失败: {role} - {msg}")
            failed_count += 1
            # 发送失败通知
            send_spider_progress(role, "error", 0, MAX_URLS_PER_ROLE, f"启动爬虫失败: {msg}")

        # 间隔30秒再爬取下一个（加快采集速度）
        time.sleep(30)

    # 关闭WebSocket连接
    if ws:
        try:
            ws.close()
        except Exception as e:
            pass

    logger.info(f"\n=== 爬取完成 ===")
    logger.info(f"成功: {success_count}")
    logger.info(f"跳过(已有足够URL): {skipped_count}")
    logger.info(f"失败: {failed_count}")

    # 发送完成汇总通知
    total_processed = success_count + skipped_count + failed_count
    success_rate = round(success_count / total_processed * 100, 1) if total_processed > 0 else 0

    summary_content = f"""**📊 角色URL采集任务完成**

**统计信息**:
- 总处理: {total_processed} 个角色
- ✅ 成功: {success_count} 个
- ⏭️ 跳过: {skipped_count} 个
- ❌ 失败: {failed_count} 个
- 成功率: {success_rate}%

**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"""

    send_spider_notification(summary_content, "采集任务完成", level="success")


if __name__ == "__main__":
    spider_via_api()
