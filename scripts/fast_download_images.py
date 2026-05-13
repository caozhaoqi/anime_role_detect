#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速下载图片脚本 - 多线程+超时处理
"""

import os
import sys
import time
import requests
import logging
import hashlib
import threading
from queue import Queue

# 配置
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url_english'
DATASET_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/merged_english_dataset'

TARGET_COUNT = 100
TIMEOUT = 15  # 超时时间
THREADS = 10  # 线程数

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_worker(queue, success_counter, fail_counter, lock):
    """下载工作线程"""
    while not queue.empty():
        url, save_path = queue.get()
        try:
            response = requests.get(url, timeout=TIMEOUT, stream=True)
            if response.status_code == 200:
                content = response.content
                if len(content) > 1024:  # 至少1KB
                    with open(save_path, 'wb') as f:
                        f.write(content)
                    with lock:
                        success_counter['value'] += 1
                else:
                    with lock:
                        fail_counter['value'] += 1
            else:
                with lock:
                    fail_counter['value'] += 1
        except Exception:
            with lock:
                fail_counter['value'] += 1
        queue.task_done()


def download_role_images(role_name):
    """下载单个角色的图片"""
    url_file = os.path.join(URL_DIR, f"{role_name}_img.txt")
    if not os.path.exists(url_file):
        logger.warning(f"⏭️ {role_name}: 未找到URL文件")
        return 0, 0
    
    save_dir = os.path.join(DATASET_DIR, role_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查当前数量
    existing = [f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    current_count = len(existing)
    
    if current_count >= TARGET_COUNT:
        logger.info(f"⏭️ {role_name}: 已有 {current_count} 张图片")
        return 0, 0
    
    need_count = TARGET_COUNT - current_count
    logger.info(f"🔄 {role_name}: 开始下载，需补充 {need_count} 张")
    
    # 读取URL
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # 去重
    existing_hashes = set()
    for img_file in existing:
        try:
            with open(os.path.join(save_dir, img_file), 'rb') as f:
                existing_hashes.add(hashlib.md5(f.read()).hexdigest())
        except:
            pass
    
    # 创建任务队列
    queue = Queue()
    success_counter = {'value': 0}
    fail_counter = {'value': 0}
    lock = threading.Lock()
    
    for url in urls:
        if success_counter['value'] >= need_count:
            break
        url_hash = hashlib.md5(url.encode()).hexdigest()
        save_path = os.path.join(save_dir, f"{url_hash}.jpg")
        if not os.path.exists(save_path) and url_hash not in existing_hashes:
            queue.put((url, save_path))
    
    # 启动线程
    threads = []
    for _ in range(min(THREADS, queue.qsize())):
        t = threading.Thread(target=download_worker, args=(queue, success_counter, fail_counter, lock))
        t.daemon = True
        t.start()
        threads.append(t)
    
    # 等待完成
    queue.join()
    
    logger.info(f"✅ {role_name}: 成功 {success_counter['value']} 张, 失败 {fail_counter['value']} 张")
    return success_counter['value'], fail_counter['value']


def get_priority_roles():
    """获取需要优先下载的角色（按当前数量排序）"""
    roles = []
    for role_dir in os.listdir(DATASET_DIR):
        role_path = os.path.join(DATASET_DIR, role_dir)
        if not os.path.isdir(role_path):
            continue
        
        current_count = len([f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        url_file = os.path.join(URL_DIR, f"{role_dir}_img.txt")
        
        if current_count < TARGET_COUNT and os.path.exists(url_file):
            with open(url_file, 'r') as f:
                url_count = len([l for l in f if l.strip()])
            roles.append({
                'name': role_dir,
                'current': current_count,
                'url_count': url_count
            })
    
    # 按当前数量排序（数量少的优先）
    roles.sort(key=lambda x: x['current'])
    return roles


def main():
    logger.info("=" * 60)
    logger.info("          快速下载图片 (多线程)")
    logger.info("=" * 60)
    
    roles = get_priority_roles()
    if not roles:
        logger.info("🎉 所有角色已达标！")
        return
    
    logger.info(f"\n待补充角色: {len(roles)} 个")
    for r in roles[:10]:
        logger.info(f"  - {r['name']}: {r['current']}/{TARGET_COUNT} (URL: {r['url_count']})")
    
    total_success = 0
    total_fail = 0
    
    for i, role in enumerate(roles, 1):
        logger.info(f"\n[{i}/{len(roles)}] 处理: {role['name']}")
        success, fail = download_role_images(role['name'])
        total_success += success
        total_fail += fail
    
    logger.info("\n" + "=" * 60)
    logger.info(f"总成功: {total_success} 张")
    logger.info(f"总失败: {total_fail} 张")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
