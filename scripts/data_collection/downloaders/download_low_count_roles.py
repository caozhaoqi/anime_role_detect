#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载图片数量低于50的角色图片
支持进度显示和飞书通知
"""

import os
import sys
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 添加公共模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))

from download_utils import (
    setup_logger,
    download_image,
    get_random_user_agent
)
from notification_utils import NotificationConfig, FeishuNotifier

# 配置
DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
ROLE_LIST_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
SPIDER_DATA_DIRS = [
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url_english',
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'
]
CONFIG_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/config.json'
TARGET_COUNT = 50  # 目标图片数量
MAX_WORKERS = 10
TIMEOUT = 15

# 全局统计
total_stats = {
    'roles_processed': 0,
    'roles_skipped': 0,
    'total_downloaded': 0,
    'total_failed': 0,
    'total_skipped': 0
}


def load_role_list():
    """加载角色列表"""
    roles = []
    with open(ROLE_LIST_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                roles.append({
                    'cn_name': parts[0],
                    'game': parts[1],
                    'en_name': parts[2],
                    'jp_name': parts[3] if len(parts) > 3 else ''
                })
    return roles


def get_current_image_count(role_name):
    """获取角色当前图片数量"""
    role_dir = os.path.join(DATASET_PATH, role_name)
    if os.path.isdir(role_dir):
        return len([f for f in os.listdir(role_dir) if f.endswith('.jpg')])
    return 0


def find_spider_files(role_name, cn_name=''):
    """查找角色的spider数据文件"""
    spider_files = []
    role_variants = [role_name, role_name.lower(), role_name.replace(' ', '_')]
    
    # 添加中文名作为搜索关键词
    if cn_name:
        role_variants.append(cn_name)
        role_variants.append(cn_name.lower())
    
    for spider_dir in SPIDER_DATA_DIRS:
        if os.path.isdir(spider_dir):
            for root, dirs, files in os.walk(spider_dir):
                for file in files:
                    if file.endswith('_img.txt'):
                        file_lower = file.lower()
                        for variant in role_variants:
                            if variant.lower() in file_lower:
                                spider_files.append(os.path.join(root, file))
                                break
    
    # 去重
    return list(set(spider_files))


def load_urls_from_spider_files(spider_files):
    """从spider文件加载URL列表"""
    urls = []
    for spider_file in spider_files:
        try:
            with open(spider_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    url = line.strip()
                    if url and url.startswith('http'):
                        urls.append(url)
        except Exception as e:
            logger.warning(f"读取文件失败 {spider_file}: {e}")
    return urls


def download_role_images(role_info):
    """下载单个角色的图片"""
    en_name = role_info['en_name']
    cn_name = role_info['cn_name']
    
    # 获取当前图片数量
    current_count = get_current_image_count(en_name)
    
    # 如果已经达到目标数量，跳过
    if current_count >= TARGET_COUNT:
        logger.info(f"角色 {cn_name}({en_name}) 已有 {current_count} 张图片，跳过")
        total_stats['roles_skipped'] += 1
        return {
            'role': en_name,
            'cn_name': cn_name,
            'current_count': current_count,
            'downloaded': 0,
            'failed': 0,
            'skipped': 0
        }
    
    # 创建角色目录
    role_dir = os.path.join(DATASET_PATH, en_name)
    os.makedirs(role_dir, exist_ok=True)
    
    # 查找spider文件（同时使用英文名和中文名）
    spider_files = find_spider_files(en_name, cn_name)
    if not spider_files:
        logger.warning(f"角色 {cn_name}({en_name}) 未找到spider数据文件")
        total_stats['roles_skipped'] += 1
        return {
            'role': en_name,
            'cn_name': cn_name,
            'current_count': current_count,
            'downloaded': 0,
            'failed': 0,
            'skipped': 0
        }
    
    # 加载URL列表
    urls = load_urls_from_spider_files(spider_files)
    if not urls:
        logger.warning(f"角色 {cn_name}({en_name}) 未找到图片URL")
        total_stats['roles_skipped'] += 1
        return {
            'role': en_name,
            'cn_name': cn_name,
            'current_count': current_count,
            'downloaded': 0,
            'failed': 0,
            'skipped': 0
        }
    
    # 计算需要下载的数量
    need_count = TARGET_COUNT - current_count
    urls = urls[:need_count * 2]  # 多取一些作为备用
    
    logger.info(f"角色 {cn_name}({en_name}): 当前 {current_count} 张，需要下载 {need_count} 张，可用URL: {len(urls)}")
    
    # 开始下载
    downloaded = 0
    failed = 0
    skipped = 0
    target_reached = False
    
    def download_with_retry(url, max_retries=2):
        """带重试的下载函数"""
        for attempt in range(max_retries):
            try:
                success, message = download_image(url, role_dir, timeout=TIMEOUT)
                if success:
                    return True, message
                elif message == "文件已存在":
                    return False, "exists"
                elif attempt < max_retries - 1:
                    time.sleep(random.uniform(0.5, 1.5))
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(0.5, 1.5))
        return False, "failed"
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_with_retry, url): url for url in urls}
        
        for future in as_completed(futures):
            if target_reached:
                executor.shutdown(wait=False)
                break
            
            success, message = future.result()
            if success:
                downloaded += 1
                total_stats['total_downloaded'] += 1
                current_count += 1
                if current_count >= TARGET_COUNT:
                    target_reached = True
                # 显示进度
                print(f"\r  [{current_count}/{TARGET_COUNT}] {message}", end='', flush=True)
            elif message == "exists":
                skipped += 1
                total_stats['total_skipped'] += 1
                current_count += 1
                if current_count >= TARGET_COUNT:
                    target_reached = True
            else:
                failed += 1
                total_stats['total_failed'] += 1
    
    print()  # 换行
    logger.info(f"角色 {cn_name}({en_name}) 下载完成: 成功 {downloaded}, 跳过 {skipped}, 失败 {failed}, 当前总数 {current_count}")
    total_stats['roles_processed'] += 1
    
    return {
        'role': en_name,
        'cn_name': cn_name,
        'current_count': current_count,
        'downloaded': downloaded,
        'failed': failed,
        'skipped': skipped
    }


def send_feishu_notification(summary):
    """发送飞书通知"""
    try:
        config = NotificationConfig(CONFIG_PATH)
        notifier = FeishuNotifier(config)
        
        message = f"""
【角色图片补充下载完成】

📊 统计结果:
- 处理角色: {summary['roles_processed']} 个
- 跳过角色: {summary['roles_skipped']} 个
- 总下载成功: {summary['total_downloaded']} 张
- 总下载失败: {summary['total_failed']} 张
- 总跳过(已存在): {summary['total_skipped']} 张

⏱️ 耗时: {summary['duration']}

📁 数据集路径: {DATASET_PATH}
        """.strip()
        
        success = notifier.send_message(message)
        if success:
            logger.info("飞书通知发送成功")
        else:
            logger.warning("飞书通知发送失败")
    except Exception as e:
        logger.error(f"发送飞书通知失败: {e}")


def main():
    global logger
    logger = setup_logger("download_low_count_roles")
    
    logger.info("=" * 60)
    logger.info("开始下载图片数量低于50的角色")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # 加载角色列表
    roles = load_role_list()
    logger.info(f"总角色数: {len(roles)}")
    
    # 找出需要下载的角色（图片数量 < 50）
    roles_to_download = []
    for role in roles:
        count = get_current_image_count(role['en_name'])
        if count < TARGET_COUNT:
            roles_to_download.append(role)
            logger.info(f"待下载: {role['cn_name']}({role['en_name']}) - 当前 {count} 张")
    
    logger.info(f"需要下载的角色数: {len(roles_to_download)}")
    logger.info("-" * 60)
    
    # 逐个下载角色图片
    results = []
    for role_info in roles_to_download:
        result = download_role_images(role_info)
        results.append(result)
        logger.info("-" * 60)
    
    # 计算耗时
    duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    
    # 汇总统计
    summary = {
        'roles_processed': total_stats['roles_processed'],
        'roles_skipped': total_stats['roles_skipped'],
        'total_downloaded': total_stats['total_downloaded'],
        'total_failed': total_stats['total_failed'],
        'total_skipped': total_stats['total_skipped'],
        'duration': duration
    }
    
    # 输出汇总报告
    logger.info("=" * 60)
    logger.info("下载完成汇总")
    logger.info("=" * 60)
    logger.info(f"处理角色: {summary['roles_processed']} 个")
    logger.info(f"跳过角色: {summary['roles_skipped']} 个")
    logger.info(f"总下载成功: {summary['total_downloaded']} 张")
    logger.info(f"总下载失败: {summary['total_failed']} 张")
    logger.info(f"总跳过(已存在): {summary['total_skipped']} 张")
    logger.info(f"耗时: {summary['duration']}")
    
    # 发送飞书通知
    send_feishu_notification(summary)
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
