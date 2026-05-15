#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门采集缺少URL的角色 - 只处理没有URL文件的角色
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 先定义 WEBSOCKET_AVAILABLE
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("websocket-client 未安装，将使用轮询方式获取进度")

from pypinyin import lazy_pinyin, Style

# 配置
API_BASE_URL = "http://localhost:33333/api/v1.2.5.260305"
WS_BASE_URL = "ws://localhost:33333/api/v1.2.5.260305"
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'

# 爬取配置
MAX_URLS_PER_ROLE = 200  # 每个角色最多采集200个URL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 进度状态
current_progress = {
    'keyword': '',
    'current_count': 0,
    'status': 'idle',
    'page': 0,
    'message': ''
}
progress_event = threading.Event()


def get_pinyin_variants(role):
    """获取角色拼音的多种变体（用于匹配不同声调标注的文件）"""
    variants = []
    # 带声调的拼音
    pinyin_tone3 = ''.join(lazy_pinyin(role, style=Style.TONE3))
    variants.append(pinyin_tone3)
    
    # 不带声调的拼音
    pinyin_normal = ''.join(lazy_pinyin(role, style=Style.NORMAL))
    variants.append(pinyin_normal)
    
    return variants


def check_url_file_exists(role):
    """检查角色是否有URL文件（支持多种拼音变体）"""
    pinyin_variants = get_pinyin_variants(role)
    
    for pinyin in pinyin_variants:
        url_file = os.path.join(URL_DIR, f"{pinyin}_img.txt")
        if os.path.exists(url_file):
            return True, pinyin
    
    # 也检查是否有匹配的文件（不考虑声调数字的精确位置）
    pinyin_normal = ''.join(lazy_pinyin(role, style=Style.NORMAL))
    for filename in os.listdir(URL_DIR):
        if filename.endswith('_img.txt'):
            # 移除声调数字后比较
            filename_no_tone = ''.join([c for c in filename.replace('_img.txt', '') if not c.isdigit()])
            if filename_no_tone == pinyin_normal:
                return True, filename.replace('_img.txt', '')
    
    return False, None


def check_url_count(role):
    """检查角色已有的URL数量（支持多种拼音变体）"""
    exists, pinyin = check_url_file_exists(role)
    
    if exists and pinyin:
        url_file = os.path.join(URL_DIR, f"{pinyin}_img.txt")
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


def on_ws_message(ws, message):
    """WebSocket消息处理"""
    global current_progress
    try:
        data = json.loads(message)
        if data.get('type') == 'spider_progress':
            current_progress = {
                'keyword': data.get('keyword', ''),
                'current_count': data.get('current_count', 0),
                'status': data.get('status', 'idle'),
                'page': data.get('page', 0),
                'message': data.get('message', '')
            }
            actual_count = check_url_count(current_progress['keyword']) if current_progress['keyword'] else 0
            logger.info(f"进度更新 [{current_progress['keyword']}]: {actual_count} URLs - {current_progress['message']}")
            
            if current_progress['status'] in ['completed', 'error']:
                progress_event.set()
    except Exception as e:
        logger.warning(f"WebSocket消息解析失败: {e}")


def start_websocket_listener():
    """启动WebSocket监听器"""
    if not WEBSOCKET_AVAILABLE:
        return None
    
    ws_url = f"{WS_BASE_URL}/sis/spider/progress/ws"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=lambda ws: logger.debug("WebSocket连接已建立"),
        on_message=on_ws_message,
        on_error=lambda ws, error: logger.warning(f"WebSocket错误: {error}"),
        on_close=lambda ws, code, msg: logger.debug("WebSocket连接已关闭")
    )
    
    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()
    time.sleep(2)
    return ws


def wait_for_spider_completion(timeout=300):
    """等待当前爬虫任务完成"""
    global current_progress
    
    start_time = time.time()
    logger.debug("等待爬虫任务完成...")
    
    if not is_spider_busy():
        logger.debug("服务已经空闲")
        return True
    
    current_progress['status'] = 'running'
    progress_event.clear()
    
    if WEBSOCKET_AVAILABLE:
        try:
            while time.time() - start_time < timeout:
                if progress_event.wait(timeout=15):
                    logger.debug("通过WebSocket检测到任务完成")
                    return True
                
                file_count = check_url_count(current_progress['keyword']) if current_progress['keyword'] else 0
                logger.info(f"等待中... [{current_progress['keyword']}] 进度: {file_count} URLs")
            
            logger.warning("等待超时")
            return False
        except Exception as e:
            logger.warning(f"WebSocket等待失败: {e}")
            return False
    else:
        while time.time() - start_time < timeout:
            if not is_spider_busy():
                logger.debug("通过轮询检测到任务完成")
                return True
            current_count = check_url_count(current_progress['keyword']) if current_progress['keyword'] else 0
            logger.info(f"等待中... 当前进度: {current_count} URLs")
            time.sleep(15)
        
        logger.warning("等待超时")
        return False


def load_role_list(role_list_file):
    """从角色名单文件中读取角色列表"""
    roles = []
    if os.path.exists(role_list_file):
        with open(role_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ')
                    if len(parts) >= 1:
                        roles.append(parts[0])
    return roles


def spider_missing_roles():
    """主函数：专门采集缺少URL的角色"""
    global current_progress
    
    logger.info("检查API服务状态...")
    if not check_api_status():
        logger.error("API服务不可用，请先启动爬虫服务！")
        return
    
    # 启动WebSocket监听器
    ws = None
    if WEBSOCKET_AVAILABLE:
        logger.info("启动WebSocket进度监听...")
        ws = start_websocket_listener()
    
    # 从角色名单文件中读取所有角色
    role_list_file = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
    all_roles = load_role_list(role_list_file)
    logger.info(f"从名单读取到 {len(all_roles)} 个角色")
    
    # 检查哪些角色确实缺少URL
    missing_roles = []
    for role in all_roles:
        exists, _ = check_url_file_exists(role)
        if not exists:
            missing_roles.append(role)
        else:
            count = check_url_count(role)
            logger.info(f"⏭️ 跳过 {role} (已有 {count} 个URL)")
    
    if not missing_roles:
        logger.info("所有目标角色的URL都已采集完成！")
        return
    
    logger.info(f"待采集角色数: {len(missing_roles)}")
    logger.info(f"缺失角色: {', '.join(missing_roles)}")
    
    # 逐个爬取
    success_count = 0
    failed_count = 0
    
    for i, role in enumerate(missing_roles, 1):
        logger.info(f"[{i}/{len(missing_roles)}] 等待服务空闲...")
        wait_for_spider_completion(300)
        
        current_progress['keyword'] = role
        
        logger.info(f"[{i}/{len(missing_roles)}] 开始爬取角色: {role}")
        
        success, msg = start_spider_via_api(role)
        if success:
            logger.info(f"✓ 成功启动爬虫: {role}")
            logger.info(f"  等待爬取完成...")
            
            wait_for_spider_completion(300)
            
            final_count = check_url_count(role)
            status = current_progress.get('status', 'completed')
            
            if status == 'completed':
                logger.info(f"  ✅ 采集完成: 获取 {final_count} 个URL")
                success_count += 1
            elif status == 'error':
                error_msg = current_progress.get('message', '未知错误')
                logger.error(f"  ❌ 采集错误: {error_msg}")
                failed_count += 1
            else:
                logger.warning(f"  ⚠️ 采集超时: 获取 {final_count} 个URL")
                success_count += 1
        else:
            logger.error(f"✗ 启动爬虫失败: {role} - {msg}")
            failed_count += 1
        
        # 间隔60秒再爬取下一个
        time.sleep(60)
    
    # 关闭WebSocket连接
    if ws:
        try:
            ws.close()
        except Exception as e:
            pass
    
    logger.info(f"\n=== 爬取完成 ===")
    logger.info(f"成功: {success_count}")
    logger.info(f"失败: {failed_count}")


if __name__ == '__main__':
    spider_missing_roles()
