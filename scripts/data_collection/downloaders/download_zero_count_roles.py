#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anime Role Dataset Supplement Tool
功能：自动补全图片数量不足 100 张的角色数据
"""

import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 路径配置：请根据实际生产环境调整
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'common'))

# 导入自定义下载工具（假设已存在于 common 模块）
try:
    from download_utils import setup_logger, download_image
except ImportError:
    print("Error: 找不到 common.download_utils。请确保路径配置正确。")
    sys.exit(1)

# --- 硬核配置区 ---
DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
SPIDER_DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'
TARGET_THRESHOLD = 100  # 补全阈值：不足100张的都会被处理
MAX_WORKERS = 10        # 线程池并发数
TIMEOUT = 15            # 单次请求超时时间

# 角色映射表：拼音(Spider文件名) -> 英文(数据集目录名)
PINYIN_TO_ENGLISH = {
    'sha1lang2bai2zi': 'Shiroko',
    'pai4meng2': 'Paimon',
    'xiang1feng1zhi4nai3': 'Chino',
    'zhi4nai3': 'Chino',
    'ke4la1la1': 'Clara',
    'ling2lan2': 'Suzuran',
    'bai2xiao4hua1': 'Shirosaki',
    'xing1ye3ri4xiang4': 'Hoshino',
    'xing1ye3': 'Hoshino',
    'zhong3cun1xiao3yi1': 'Tanemura',
    'xiao3zhi1sen1xia4yin1': 'Konomori',
    'chu2he4ai4': 'Hinaatsu',
    'ye4cha1shen2tian1yi1': 'Yashajin',
    'kong1yin2zi': 'Kuonji',
    'zao3lai4you1xiang1': 'Dream!',
    'yi1zhi1lai4ming2ri4nai4': 'Ichinose',
    'sheng4yuan2wei4hua1': 'Mika',
}

def get_current_image_count(role_name):
    """
    扫描文件系统，统计当前角色的有效样本数
    """
    role_dir = os.path.join(DATASET_PATH, role_name)
    if os.path.isdir(role_dir):
        # 统计 jpg 和 png 文件
        return len([f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return 0

def load_urls_from_spider_file(spider_file):
    """
    从本地爬虫日志文件中清洗并加载解析出的 URL
    """
    urls = []
    if not os.path.exists(spider_file):
        return []
    try:
        with open(spider_file, 'r', encoding='utf-8', errors='ignore') as f:
            urls = [line.strip() for line in f if line.strip().startswith('http')]
    except Exception as e:
        logger.error(f"IO Error reading {spider_file}: {e}")
    return list(dict.fromkeys(urls)) # 去重

def download_single_role(pinyin_name, en_name):
    """
    针对单个角色执行增量补全任务
    """
    current_count = get_current_image_count(en_name)
    
    if current_count >= TARGET_THRESHOLD:
        logger.info(f"OK: {en_name} 样本充足 (当前: {current_count})")
        return
    
    need_count = TARGET_THRESHOLD - current_count
    spider_file = os.path.join(SPIDER_DATA_DIR, f'{pinyin_name}_img.txt')
    
    # 路径校验
    role_dir = os.path.join(DATASET_PATH, en_name)
    os.makedirs(role_dir, exist_ok=True)
    
    urls = load_urls_from_spider_file(spider_file)
    if not urls:
        logger.warning(f"Empty: {en_name} 爬虫源文件中无可用 URL")
        return

    # 为了保证成功率，我们获取比需求量多一倍的 URL 进行并发尝试
    urls = urls[current_count : current_count + need_count * 2]
    
    logger.info(f"Running: {en_name} | 进度: {current_count}/{TARGET_THRESHOLD} | 需补全: {need_count}")
    
    stats = {"success": 0, "failed": 0, "skipped": 0}
    is_completed = False
    
    def worker(url):
        # 内部重试机制 + 随机抖动，规避简单的反爬
        for _ in range(2):
            try:
                success, message = download_image(url, role_dir, timeout=TIMEOUT)
                if success:
                    return "SUCCESS"
                if message == "文件已存在":
                    return "EXISTS"
                time.sleep(random.uniform(0.3, 0.8))
            except:
                continue
        return "FAILED"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(worker, url): url for url in urls}
        
        for future in as_completed(future_to_url):
            if current_count >= TARGET_THRESHOLD:
                is_completed = True
                break
            
            res = future.result()
            if res == "SUCCESS":
                stats["success"] += 1
                current_count += 1
            elif res == "EXISTS":
                stats["skipped"] += 1
                current_count += 1
            else:
                stats["failed"] += 1
            
            # 终端实时回显打印（硬核进度条）
            sys.stdout.write(f"\r  Progress: [{current_count}/{TARGET_THRESHOLD}] Succ:{stats['success']} Fail:{stats['failed']}")
            sys.stdout.flush()
    
    print() # 换行
    logger.info(f"Finished {en_name}: 最终样本数 {current_count} (成功补全 {stats['success']})")

def main():
    global logger
    logger = setup_logger("dataset_supplement_engine")
    
    logger.info("=" * 60)
    logger.info("  Anime Role Dataset 增量补全引擎启动")
    logger.info(f"  目标阈值: {TARGET_THRESHOLD} | 并发线程: {MAX_WORKERS}")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # 1. 扫描待处理队列
    target_roles = []
    for pinyin, en in PINYIN_TO_ENGLISH.items():
        count = get_current_image_count(en)
        if count < TARGET_THRESHOLD:
            target_roles.append((pinyin, en, count))
    
    if not target_roles:
        logger.info("所有角色样本均已达标，无需处理。")
        return

    logger.info(f"发现 {len(target_roles)} 个角色样本不足 100 张，准备补全...")
    logger.info("-" * 60)
    
    # 2. 串行处理角色，角色内部并行处理图片（防止单角色并发过高被封 IP）
    try:
        for pinyin, en, count in target_roles:
            download_single_role(pinyin, en)
            logger.info("-" * 60)
    except KeyboardInterrupt:
        logger.warning("用户手动中止任务")
    
    # 3. 统计收尾
    duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    logger.info("=" * 60)
    logger.info(f"任务结束 | 总耗时: {duration}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()