#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带存储空间监控的下载脚本
当可用空间小于1GB时暂停下载并整理数据
"""

import os
import sys
import time
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加公共模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))

from download_utils import setup_logger, download_image

# 配置
DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
SPIDER_DATA_DIRS = [
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url_english'
    # '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'
]
TARGET_COUNT = 100  # 目标图片数量
MAX_WORKERS = 10
TIMEOUT = 15
MIN_FREE_SPACE_GB = 1  # 最小可用空间（GB）
PAUSE_CHECK_INTERVAL = 10  # 检查间隔（下载图片数）

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

def get_free_space_gb(path):
    """获取指定路径所在磁盘的可用空间（GB）"""
    try:
        stat = os.statvfs(path)
        free_bytes = stat.f_bavail * stat.f_frsize
        return free_bytes / (1024 ** 3)
    except Exception as e:
        logger.error(f"获取磁盘空间失败: {e}")
        return 0

def check_storage_and_cleanup(min_free_gb=MIN_FREE_SPACE_GB, cleanup_threshold=0.5):
    """检查存储空间，不足时进行清理"""
    free_gb = get_free_space_gb(DATASET_PATH)
    logger.info(f"当前可用空间: {free_gb:.2f} GB")
    
    if free_gb < min_free_gb:
        logger.warning(f"可用空间不足 {min_free_gb} GB，需要清理...")
        
        # 尝试清理，目标释放到至少 min_free_gb * (1 + cleanup_threshold) GB
        target_free = min_free_gb * (1 + cleanup_threshold)
        need_free = target_free - free_gb
        
        logger.info(f"需要释放空间: {need_free:.2f} GB")
        cleaned_bytes = cleanup_dataset(need_free * (1024 ** 3))
        
        # 重新检查空间
        free_gb = get_free_space_gb(DATASET_PATH)
        logger.info(f"清理后可用空间: {free_gb:.2f} GB")
        
        if free_gb < min_free_gb:
            logger.error(f"空间仍然不足 {min_free_gb} GB，暂停下载")
            return False
        else:
            logger.info("空间清理完成，继续下载")
    
    return True

def cleanup_dataset(target_free_bytes):
    """清理数据集，释放指定字节数的空间"""
    cleaned_bytes = 0
    candidates = []
    
    # 收集所有超过TARGET_COUNT的角色目录
    for dir_name in os.listdir(DATASET_PATH):
        dir_path = os.path.join(DATASET_PATH, dir_name)
        if os.path.isdir(dir_path):
            images = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
            if len(images) > TARGET_COUNT:
                # 按图片数量排序，优先清理数量多的
                images.sort()
                excess_images = images[TARGET_COUNT:]  # 保留前TARGET_COUNT张
                for img in excess_images:
                    img_path = os.path.join(dir_path, img)
                    try:
                        size = os.path.getsize(img_path)
                        candidates.append((size, img_path, dir_name))
                    except:
                        pass
    
    # 按文件大小排序，优先删除大文件
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 删除文件直到达到目标
    for size, img_path, dir_name in candidates:
        if cleaned_bytes >= target_free_bytes:
            break
        try:
            os.remove(img_path)
            cleaned_bytes += size
            logger.debug(f"删除 {dir_name}/{os.path.basename(img_path)}: {size/1024/1024:.2f} MB")
        except Exception as e:
            logger.warning(f"删除失败 {img_path}: {e}")
    
    logger.info(f"共清理: {cleaned_bytes/1024/1024:.2f} MB")
    return cleaned_bytes

def get_role_name_from_spider_file(file_name):
    """从spider文件名提取角色名"""
    name = file_name.replace('_img.txt', '')
    
    if name in PINYIN_TO_ENGLISH:
        return PINYIN_TO_ENGLISH[name]
    
    name_clean = name.replace(' ', '_')
    
    if not any(c.isdigit() for c in name_clean):
        return name_clean
    
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
    """从单个spider文件下载图片，带有存储空间监控"""
    file_name = os.path.basename(spider_file)
    role_name = get_role_name_from_spider_file(file_name)
    
    current_count = get_current_image_count(role_name)
    
    if current_count >= TARGET_COUNT:
        logger.info(f"跳过 {role_name}: 已有 {current_count} 张图片")
        return {
            'role': role_name,
            'current_count': current_count,
            'downloaded': 0,
            'failed': 0,
            'skipped': 0,
            'skipped_reason': 'already_has_enough',
            'stopped_early': False
        }
    
    role_dir = os.path.join(DATASET_PATH, role_name)
    os.makedirs(role_dir, exist_ok=True)
    
    urls = load_urls_from_spider_file(spider_file)
    if not urls:
        logger.warning(f"角色 {role_name} 未找到图片URL")
        return {
            'role': role_name,
            'current_count': current_count,
            'downloaded': 0,
            'failed': 0,
            'skipped': 0,
            'skipped_reason': 'no_urls',
            'stopped_early': False
        }
    
    need_count = TARGET_COUNT - current_count
    urls = urls[:need_count * 2]
    
    logger.info(f"处理 {role_name}: 当前 {current_count} 张，需要下载 {need_count} 张")
    
    downloaded = 0
    failed = 0
    skipped = 0
    target_reached = False
    stopped_early = False
    download_counter = 0
    
    def download_with_retry(url, max_retries=2):
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
            if target_reached or stopped_early:
                executor.shutdown(wait=False)
                break
            
            success, message = future.result()
            if success:
                downloaded += 1
                current_count += 1
                download_counter += 1
                if current_count >= TARGET_COUNT:
                    target_reached = True
                print(f"\r  [{current_count}/{TARGET_COUNT}] {message}", end='', flush=True)
                
                # 定期检查存储空间
                if download_counter % PAUSE_CHECK_INTERVAL == 0:
                    if not check_storage_and_cleanup():
                        stopped_early = True
                        logger.warning("存储空间不足，暂停下载")
                        executor.shutdown(wait=False)
                        break
                        
            elif message == "exists":
                skipped += 1
                current_count += 1
                if current_count >= TARGET_COUNT:
                    target_reached = True
            else:
                failed += 1
    
    print()
    logger.info(f"{role_name} 下载完成: 成功 {downloaded}, 跳过 {skipped}, 失败 {failed}" + 
                (" (空间不足暂停)" if stopped_early else ""))
    
    return {
        'role': role_name,
        'current_count': current_count,
        'downloaded': downloaded,
        'failed': failed,
        'skipped': skipped,
        'skipped_reason': None,
        'stopped_early': stopped_early
    }

def main():
    global logger
    logger = setup_logger("download_with_storage_monitor")
    
    logger.info("=" * 60)
    logger.info("开始下载图片（带存储空间监控）")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # 初始检查存储空间
    if not check_storage_and_cleanup():
        logger.error("初始空间检查失败，退出")
        return
    
    spider_files = []
    for spider_dir in SPIDER_DATA_DIRS:
        if os.path.isdir(spider_dir):
            for root, dirs, files in os.walk(spider_dir):
                for file in files:
                    if file.endswith('_img.txt'):
                        spider_files.append(os.path.join(root, file))
    
    logger.info(f"找到 {len(spider_files)} 个spider文件")
    
    roles_to_download = []
    for spider_file in spider_files:
        file_name = os.path.basename(spider_file)
        role_name = get_role_name_from_spider_file(file_name)
        count = get_current_image_count(role_name)
        if count < TARGET_COUNT:
            roles_to_download.append((role_name, count, spider_file))
    
    roles_to_download.sort(key=lambda x: x[1])
    
    logger.info(f"需要下载的角色数: {len(roles_to_download)}")
    for role_name, count, spider_file in roles_to_download[:10]:  # 只显示前10个
        logger.info(f"  {role_name}: 当前 {count} 张")
    if len(roles_to_download) > 10:
        logger.info(f"  ... 还有 {len(roles_to_download) - 10} 个角色")
    logger.info("-" * 60)
    
    results = []
    for role_name, count, spider_file in roles_to_download:
        result = download_role_images_from_file(spider_file)
        results.append(result)
        
        # 如果因为空间不足停止，退出循环
        if result.get('stopped_early', False):
            logger.warning("由于存储空间不足，暂停下载任务")
            break
        
        logger.info("-" * 60)
    
    total_downloaded = sum(r['downloaded'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    total_skipped = sum(r['skipped'] for r in results)
    processed = len([r for r in results if r['skipped_reason'] is None])
    skipped = len([r for r in results if r['skipped_reason'] is not None])
    stopped_early_count = sum(1 for r in results if r.get('stopped_early', False))
    
    duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    
    logger.info("=" * 60)
    logger.info("下载完成汇总")
    logger.info("=" * 60)
    logger.info(f"处理角色: {processed} 个")
    logger.info(f"跳过角色: {skipped} 个")
    logger.info(f"因空间不足停止: {stopped_early_count} 个")
    logger.info(f"总下载成功: {total_downloaded} 张")
    logger.info(f"总下载失败: {total_failed} 张")
    logger.info(f"总跳过(已存在): {total_skipped} 张")
    logger.info(f"耗时: {duration}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
