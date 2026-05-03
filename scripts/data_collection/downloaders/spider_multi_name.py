#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版角色URL采集脚本 - 支持多名称采集
使用角色的中文名、英文名、日文名、游戏/动漫名进行多角度采集
所有采集结果合并到同一角色文件中
"""

import os
import sys
import time
import requests
import logging
import json
import threading
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 先定义 WEBSOCKET_AVAILABLE
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("websocket-client 未安装，将使用轮询方式获取进度")

from pypinyin import lazy_pinyin, Style

# 加载飞书配置
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'notification_config.json')
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        notification_config = json.load(f)
    
    os.environ['NOTIFICATION_ENABLED'] = 'true'
    os.environ['NOTIFICATION_PLATFORM'] = notification_config['platform']
    os.environ['FEISHU_APP_ID'] = notification_config['feishu']['app_id']
    os.environ['FEISHU_APP_SECRET'] = notification_config['feishu']['app_secret']
    os.environ['FEISHU_RECEIVE_ID'] = notification_config['feishu']['receive_id']
    os.environ['FEISHU_RECEIVE_ID_TYPE'] = notification_config['feishu']['receive_id_type']
    logging.info(f"已加载通知配置: {config_path}")
else:
    logging.warning(f"未找到通知配置文件: {config_path}")

# 导入统一通知服务
try:
    from src.services.notification_service import get_notification_manager, send_notification
    NOTIFICATION_AVAILABLE = True
except ImportError as e:
    NOTIFICATION_AVAILABLE = False
    logging.warning(f"通知服务未找到: {e}")

# 配置
API_BASE_URL = "http://localhost:33333/api/v1.2.5.260305"
WS_BASE_URL = "ws://localhost:33333/api/v1.2.5.260305"
ROLE_LIST_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'

# 爬取配置
MAX_URLS_PER_ROLE = 500    # 每个角色最多采集500个URL
MIN_URLS_PER_ROLE = 200    # 每个角色至少采集200个URL才跳过
MAX_URLS_PER_NAME = 150    # 每个名称最多采集150个URL（避免重复）
COLLECTION_INTERVAL = 60   # 采集间隔（秒）

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 进度状态
current_progress = {
    'keyword': '',
    'current_count': 0,
    'status': 'idle',
    'page': 0,
    'message': ''
}
progress_event = threading.Event()
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
    """发送爬虫通知"""
    if notification_manager:
        try:
            return notification_manager.send(message, title, level)
        except Exception as e:
            logger.warning(f"发送通知失败: {e}")
            return False
    return False


def send_spider_progress(role, status, count, total, message=""):
    """发送采集进度通知"""
    status_emoji = {
        'running': "🔄",
        'completed': "✅",
        'error': "❌",
        'skipped': "⏭️"
    }
    
    status_text = {
        'running': "采集进行中",
        'completed': "采集完成",
        'error': "采集失败",
        'skipped': "已跳过"
    }
    
    emoji = status_emoji.get(status, "📦")
    text = status_text.get(status, "未知状态")
    
    title = f"{emoji} 角色采集状态更新"
    content = f"**角色**: {role}\n**状态**: {text}\n**进度**: {count}/{total}\n**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    if message:
        content += f"\n**消息**: {message}"
    
    send_spider_notification(content, title, level="success" if status == 'completed' else "error" if status == 'error' else "info")


def parse_role_list():
    """解析角色列表，提取所有可用名称"""
    roles = []
    with open(ROLE_LIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    role_info = {
                        'chinese_name': parts[0],
                        'game': parts[1],
                        'english_name': parts[2] if len(parts) > 2 else '',
                        'japanese_name': parts[3] if len(parts) > 3 else ''
                    }
                    roles.append(role_info)
    return roles


def get_all_search_names(role_info):
    """获取角色的所有搜索名称"""
    names = []
    
    # 中文名
    if role_info.get('chinese_name'):
        names.append(role_info['chinese_name'])
    
    # 英文名
    if role_info.get('english_name') and role_info['english_name'] != '-':
        names.append(role_info['english_name'])
    
    # 日文名（过滤无效的）
    if role_info.get('japanese_name') and role_info['japanese_name'] != '-' and role_info['japanese_name'] != role_info.get('chinese_name'):
        names.append(role_info['japanese_name'])
    
    # 中文名+游戏名组合
    if role_info.get('chinese_name') and role_info.get('game'):
        names.append(f"{role_info['chinese_name']} {role_info['game']}")
    
    # 英文名+游戏名组合
    if role_info.get('english_name') and role_info.get('game') and role_info['english_name'] != '-':
        names.append(f"{role_info['english_name']} {role_info['game']}")
    
    # 去重并返回
    return list(set(names))


def get_role_pinyin(chinese_name):
    """获取角色拼音（用于文件名）"""
    return ''.join(lazy_pinyin(chinese_name, style=Style.TONE3))


def check_url_count(chinese_name):
    """检查角色已有的URL数量"""
    pinyin = get_role_pinyin(chinese_name)
    url_file = os.path.join(URL_DIR, f"{pinyin}_img.txt")
    
    if os.path.exists(url_file):
        with open(url_file, 'r', encoding='utf-8') as f:
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
            if result.get("code") == 0 or (result.get("msg") and "success" in result.get("msg").lower()):
                return True, "success"
            else:
                return False, result.get('msg', 'unknown error')
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


def is_spider_busy():
    """检查爬虫是否正在运行"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/sis/spider_start/single",
            params={"key_word": ""},
            timeout=10
        )
        result = response.json()
        
        msg = result.get("msg", "").strip()
        data = result.get("data", "").strip()
        code = result.get("code", -1)
        
        if "操作进行中" in msg or "操作进行中" in data:
            return True
        elif "关键词不能为空" in data:
            return False
        elif code == 0:
            return False
        
        logger.debug(f"未知响应: {result}")
        return True
    except Exception as e:
        logger.debug(f"检查状态失败: {e}")
        return True


def wait_for_spider_completion(timeout=300):
    """等待爬虫任务完成"""
    global current_progress
    
    start_time = time.time()
    logger.debug("等待爬虫任务完成...")
    
    if not is_spider_busy():
        logger.debug("服务已经空闲")
        return True
    
    current_progress['status'] = 'running'
    progress_event.clear()
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_spider_busy():
            logger.debug("通过轮询检测到任务完成")
            return True
        
        current_count = check_url_count(current_progress['keyword']) if current_progress['keyword'] else 0
        logger.info(f"等待中... 当前进度: {current_count} URLs")
        time.sleep(15)
    
    logger.warning("等待超时")
    return False


def merge_urls(chinese_name):
    """合并去重角色的URL文件"""
    pinyin = get_role_pinyin(chinese_name)
    url_file = os.path.join(URL_DIR, f"{pinyin}_img.txt")
    
    if not os.path.exists(url_file):
        return 0
    
    with open(url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # 去重
    unique_urls = list(set(urls))
    
    # 按长度排序（短的在前）
    unique_urls.sort(key=lambda x: len(x))
    
    # 如果超过最大限制，只保留前MAX_URLS_PER_ROLE个
    if len(unique_urls) > MAX_URLS_PER_ROLE:
        unique_urls = unique_urls[:MAX_URLS_PER_ROLE]
    
    # 写回文件
    with open(url_file, 'w', encoding='utf-8') as f:
        for url in unique_urls:
            f.write(url + '\n')
    
    return len(unique_urls)


def spider_role_with_multi_names(role_info):
    """使用多种名称采集单个角色"""
    global current_progress
    
    chinese_name = role_info['chinese_name']
    search_names = get_all_search_names(role_info)
    
    logger.info(f"\n[角色] {chinese_name}")
    logger.info(f"[游戏] {role_info['game']}")
    logger.info(f"[搜索名称] {', '.join(search_names)}")
    
    # 检查当前URL数量
    initial_count = check_url_count(chinese_name)
    if initial_count >= MIN_URLS_PER_ROLE:
        logger.info(f"⏭️ 跳过 {chinese_name} (已有 {initial_count} 个URL ≥ {MIN_URLS_PER_ROLE})")
        send_spider_progress(chinese_name, 'skipped', initial_count, MAX_URLS_PER_ROLE, 
                            f"已有足够URL ({initial_count} ≥ {MIN_URLS_PER_ROLE})")
        return 'skipped', initial_count, 0
    
    logger.info(f"开始多名称采集，当前URL: {initial_count}")
    send_spider_progress(chinese_name, 'running', initial_count, MAX_URLS_PER_ROLE, 
                        f"开始多名称采集: {', '.join(search_names)}")
    
    success_names = []
    failed_names = []
    
    for name in search_names:
        # 检查当前总量是否已达到上限
        current_total = check_url_count(chinese_name)
        if current_total >= MAX_URLS_PER_ROLE:
            logger.info(f"✅ 已达到URL上限 {MAX_URLS_PER_ROLE}，停止采集")
            break
        
        # 等待服务空闲
        logger.info(f"等待服务空闲...")
        wait_for_spider_completion(300)
        
        # 设置当前关键词
        current_progress['keyword'] = name
        
        logger.info(f"  └── 正在采集: {name}")
        
        success, msg = start_spider_via_api(name)
        if success:
            logger.info(f"      ✓ 启动成功")
            
            # 等待采集完成
            wait_for_spider_completion(300)
            
            # 合并去重
            final_count = merge_urls(chinese_name)
            logger.info(f"      ✓ 采集完成，当前总量: {final_count}")
            success_names.append(name)
        else:
            logger.error(f"      ✗ 采集失败: {msg}")
            failed_names.append(name)
        
        # 间隔时间
        time.sleep(COLLECTION_INTERVAL)
    
    # 最终合并去重
    final_count = merge_urls(chinese_name)
    
    if success_names:
        result = 'completed' if len(failed_names) == 0 else 'partial'
        msg = f"成功: {', '.join(success_names)}"
        if failed_names:
            msg += f" | 失败: {', '.join(failed_names)}"
        send_spider_progress(chinese_name, 'completed', final_count, MAX_URLS_PER_ROLE, msg)
        return result, final_count, len(success_names)
    else:
        send_spider_progress(chinese_name, 'error', final_count, MAX_URLS_PER_ROLE, 
                            f"所有名称采集失败: {', '.join(failed_names)}")
        return 'error', final_count, 0


def spider_multi_name():
    """主函数：多名称采集"""
    global current_progress
    
    init_notification()
    
    logger.info("检查API服务状态...")
    if not check_api_status():
        logger.error("API服务不可用，请先启动爬虫服务！")
        send_spider_notification("❌ API服务不可用，请先启动爬虫服务！", "爬虫服务异常", level="error")
        return
    
    roles = parse_role_list()
    logger.info(f"\n📊 角色列表分析")
    logger.info(f"总角色数: {len(roles)}")
    logger.info(f"采集配置: 每个角色最多 {MAX_URLS_PER_ROLE} 个URL，每个名称最多 {MAX_URLS_PER_NAME} 个URL")
    
    # 发送开始通知
    send_spider_notification(
        f"**🚀 多名称角色URL采集任务开始**\n\n"
        f"**总角色数**: {len(roles)}\n"
        f"**采集策略**: 中文名 + 英文名 + 日文名 + 游戏名组合\n"
        f"**配置**: 每个角色最多 {MAX_URLS_PER_ROLE} 个URL\n"
        f"**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "多名称采集任务开始",
        level="info"
    )
    
    # 逐个角色采集
    success_count = 0
    skipped_count = 0
    failed_count = 0
    total_names_used = 0
    
    for i, role_info in enumerate(roles, 1):
        chinese_name = role_info['chinese_name']
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(roles)}] 处理角色: {chinese_name}")
        logger.info(f"{'='*60}")
        
        result, final_count, names_used = spider_role_with_multi_names(role_info)
        
        if result == 'skipped':
            skipped_count += 1
        elif result in ['completed', 'partial']:
            success_count += 1
            total_names_used += names_used
        else:
            failed_count += 1
    
    # 发送完成汇总通知
    total_processed = success_count + skipped_count + failed_count
    success_rate = round(success_count / total_processed * 100, 1) if total_processed > 0 else 0
    
    summary_content = f"""**📊 多名称角色URL采集任务完成**

**统计信息**:
- 总处理: {total_processed} 个角色
- ✅ 成功: {success_count} 个
- ⏭️ 跳过: {skipped_count} 个
- ❌ 失败: {failed_count} 个
- 成功率: {success_rate}%
- 平均每个角色使用名称数: {round(total_names_used / max(success_count, 1), 1)}

**配置参数**:
- 每个角色最多: {MAX_URLS_PER_ROLE} 个URL
- 每个名称最多: {MAX_URLS_PER_NAME} 个URL
- 跳过阈值: ≥ {MIN_URLS_PER_ROLE} 个URL

**时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"""
    
    send_spider_notification(summary_content, "多名称采集任务完成", level="success")
    
    logger.info(f"\n{'='*60}")
    logger.info("=== 多名称采集完成 ===")
    logger.info(f"成功: {success_count}")
    logger.info(f"跳过(已有足够URL): {skipped_count}")
    logger.info(f"失败: {failed_count}")
    logger.info(f"平均每个角色使用名称数: {round(total_names_used / max(success_count, 1), 1)}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    spider_multi_name()
