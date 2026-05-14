#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单版本：下载图片数量低于50的角色图片
直接遍历spider文件并下载到对应目录
"""

import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加公共模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))

from download_utils import setup_logger, download_image

# 配置
DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
SPIDER_DATA_DIRS = [
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url_english',
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'
]
TARGET_COUNT = 200  # 目标图片数量
MAX_WORKERS = 10
TIMEOUT = 15

# 拼音到英文名的映射
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
    'ji1ban3nai4ai4': 'Himesaka',
    'zhong3cun1xiao3yi1': 'Tanemura',
    'xiao3zhi1sen1xia4yin1': 'Konomori',
    'chu2he4ai4': 'Hinaatsu',
    'ye4cha1shen2tian1yi1': 'Yashajin',
    'kong1yin2zi': 'Kuonji',
    'zao3lai4you1xiang1': 'Dream!',
    'yi1zhi1lai4ming2ri4nai4': 'Ichinose',
    'sheng4yuan2wei4hua1': 'Mika',
}

def get_role_name_from_spider_file(file_name):
    """从spider文件名提取角色名"""
    # 移除 _img.txt 后缀
    name = file_name.replace('_img.txt', '')
    
    # 尝试拼音映射
    if name in PINYIN_TO_ENGLISH:
        return PINYIN_TO_ENGLISH[name]
    
    # 尝试移除空格和特殊字符
    name_clean = name.replace(' ', '_')
    
    # 如果文件名本身就是英文名，直接返回
    # 检查是否包含数字（拼音格式）
    if not any(c.isdigit() for c in name_clean):
        return name_clean
    
    # 默认返回原始名称（去掉后缀）
    return name_clean

def get_current_image_count(role_name):
    """获取角色当前图片数量"""
    role_dir = os.path.join(DATASET_PATH, role_name)
    if os.path.isdir(role_dir):
        return len([f for f in os.listdir(role_dir) if f.endswith('.jpg')])
    return 0

def load_urls_from_spider_file(spider_file):
    """从spider文件加载URL列表"""
    urls = []
    try:
        with open(spider_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                url = line.strip()
                if url and url.startswith('http'):
                    urls.append(url)
    except Exception as e:
        logger.warning(f"读取文件失败 {spider_file}: {e}")
    return urls

def download_role_images_from_file(spider_file):
    """从单个spider文件下载图片"""
    file_name = os.path.basename(spider_file)
    role_name = get_role_name_from_spider_file(file_name)
    
    # 获取当前图片数量
    current_count = get_current_image_count(role_name)
    
    # 如果已经达到目标数量，跳过
    if current_count >= TARGET_COUNT:
        logger.info(f"跳过 {role_name}: 已有 {current_count} 张图片")
        return {
            'role': role_name,
            'current_count': current_count,
            'downloaded': 0,
            'failed': 0,
            'skipped': 0,
            'skipped_reason': 'already_has_enough'
        }
    
    # 创建角色目录
    role_dir = os.path.join(DATASET_PATH, role_name)
    os.makedirs(role_dir, exist_ok=True)
    
    # 加载URL列表
    urls = load_urls_from_spider_file(spider_file)
    if not urls:
        logger.warning(f"角色 {role_name} 未找到图片URL")
        return {
            'role': role_name,
            'current_count': current_count,
            'downloaded': 0,
            'failed': 0,
            'skipped': 0,
            'skipped_reason': 'no_urls'
        }
    
    # 计算需要下载的数量
    need_count = TARGET_COUNT - current_count
    urls = urls[:need_count * 2]  # 多取一些作为备用
    
    logger.info(f"处理 {role_name}: 当前 {current_count} 张，需要下载 {need_count} 张")
    
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
                current_count += 1
                if current_count >= TARGET_COUNT:
                    target_reached = True
                print(f"\r  [{current_count}/{TARGET_COUNT}] {message}", end='', flush=True)
            elif message == "exists":
                skipped += 1
                current_count += 1
                if current_count >= TARGET_COUNT:
                    target_reached = True
            else:
                failed += 1
    
    print()  # 换行
    logger.info(f"{role_name} 下载完成: 成功 {downloaded}, 跳过 {skipped}, 失败 {failed}")
    
    return {
        'role': role_name,
        'current_count': current_count,
        'downloaded': downloaded,
        'failed': failed,
        'skipped': skipped,
        'skipped_reason': None
    }

def main():
    global logger
    logger = setup_logger("download_low_count_simple")
    
    logger.info("=" * 60)
    logger.info("开始下载图片数量低于50的角色")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # 收集所有spider文件
    spider_files = []
    for spider_dir in SPIDER_DATA_DIRS:
        if os.path.isdir(spider_dir):
            for root, dirs, files in os.walk(spider_dir):
                for file in files:
                    if file.endswith('_img.txt'):
                        spider_files.append(os.path.join(root, file))
    
    logger.info(f"找到 {len(spider_files)} 个spider文件")
    
    # 统计需要下载的角色
    roles_to_download = []
    for spider_file in spider_files:
        file_name = os.path.basename(spider_file)
        role_name = get_role_name_from_spider_file(file_name)
        count = get_current_image_count(role_name)
        if count < TARGET_COUNT:
            roles_to_download.append((role_name, count, spider_file))
    
    # 按图片数量排序（先处理数量少的）
    roles_to_download.sort(key=lambda x: x[1])
    
    logger.info(f"需要下载的角色数: {len(roles_to_download)}")
    for role_name, count, spider_file in roles_to_download:
        logger.info(f"  {role_name}: 当前 {count} 张")
    logger.info("-" * 60)
    
    # 逐个下载
    results = []
    for role_name, count, spider_file in roles_to_download:
        result = download_role_images_from_file(spider_file)
        results.append(result)
        logger.info("-" * 60)
    
    # 汇总统计
    total_downloaded = sum(r['downloaded'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    total_skipped = sum(r['skipped'] for r in results)
    processed = len([r for r in results if r['skipped_reason'] is None])
    skipped = len([r for r in results if r['skipped_reason'] is not None])
    
    duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    
    # 输出汇总报告
    logger.info("=" * 60)
    logger.info("下载完成汇总")
    logger.info("=" * 60)
    logger.info(f"处理角色: {processed} 个")
    logger.info(f"跳过角色: {skipped} 个")
    logger.info(f"总下载成功: {total_downloaded} 张")
    logger.info(f"总下载失败: {total_failed} 张")
    logger.info(f"总跳过(已存在): {total_skipped} 张")
    logger.info(f"耗时: {duration}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
