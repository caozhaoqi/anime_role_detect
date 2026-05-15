#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优先下载数据不足的角色
"""
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# 配置
DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
SPIDER_DATA_DIRS = [
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url',
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url_english'
]
TARGET_COUNT = 100  # 目标数量
MAX_WORKERS = 10
TIMEOUT = 15

# 需要优先下载的角色列表
PRIORITY_ROLES = {
    '姬坂乃爱': {'en_name': 'Himesaka', 'needed': 100},
    '克拉拉': {'en_name': 'Clara', 'needed': 13},
    '猫宫又奈': {'en_name': 'Yanagi', 'needed': 4},
    '铃兰': {'en_name': 'Suzuran', 'needed': 4},
    '圣园未花': {'en_name': 'Mika', 'needed': 3},
    '香风智乃': {'en_name': 'Chino', 'needed': 2},
    '早濑优香': {'en_name': 'Dream!', 'needed': 2},
    '派蒙': {'en_name': 'Paimon', 'needed': 1},
    '白咲花': {'en_name': 'Shirosaki', 'needed': 1},
    '星野日向': {'en_name': 'Hoshino', 'needed': 1},
    '小之森夏音': {'en_name': 'Konomori', 'needed': 1},
    '雏鹤爱': {'en_name': 'Hinaatsu', 'needed': 1},
    '一之濑明日奈': {'en_name': 'Ichinose', 'needed': 1},
    '小鸟游星野': {'en_name': 'Hoshino', 'needed': 1},
}

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_current_image_count(role_name):
    """获取角色当前图片数量"""
    role_dir = os.path.join(DATASET_PATH, role_name)
    if os.path.isdir(role_dir):
        return len([f for f in os.listdir(role_dir) if f.lower().endswith('.jpg')])
    return 0

def download_image(url, save_path):
    """下载单张图片"""
    try:
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception:
        return False

def download_role_images(role_name, cn_name, needed_count):
    """下载角色图片"""
    logger = logging.getLogger('download_priority_roles')
    
    # 创建目录
    role_dir = os.path.join(DATASET_PATH, role_name)
    os.makedirs(role_dir, exist_ok=True)
    
    # 查找URL文件
    url_file = None
    for spider_dir in SPIDER_DATA_DIRS:
        possible_files = [
            f'{role_name}_img.txt',
            # 尝试拼音变体
        ]
        for fname in possible_files:
            fpath = os.path.join(spider_dir, fname)
            if os.path.exists(fpath):
                url_file = fpath
                break
    
    if not url_file:
        # 尝试查找拼音文件名
        pinyin_variants = [
            'ji1ban3nai4ai4',  # 姬坂乃爱
            'ke4la1la1',       # 克拉拉
            'mao1gong1you4nai4', # 猫宫又奈
            'ling2lan2',        # 铃兰
            'sheng4yuan2wei4hua1', # 圣园未花
            'xiang1feng1zhi4nai3', # 香风智乃
            'zao3lai4you1xiang1', # 早濑优香
            'pai4meng2',        # 派蒙
            'bai2xiao4hua1',    # 白咲花
            'xing1ye3ri4xiang4', # 星野日向
            'xiao3zhi1sen1xia4yin1', # 小之森夏音
            'chu2he4ai4',       # 雏鹤爱
            'yi1zhi1lai4ming2ri4nai4', # 一之濑明日奈
            'xing1ye3',         # 小鸟游星野
        ]
        
        for pinyin in pinyin_variants:
            for spider_dir in SPIDER_DATA_DIRS:
                fpath = os.path.join(spider_dir, f'{pinyin}_img.txt')
                if os.path.exists(fpath):
                    url_file = fpath
                    break
            if url_file:
                break
    
    if not url_file:
        logger.error(f"❌ 角色 {cn_name} ({role_name}) 未找到URL文件")
        return {'success': 0, 'failed': 0, 'skipped': 0}
    
    # 读取URL列表
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    current_count = get_current_image_count(role_name)
    need_download = max(0, needed_count)
    downloaded = 0
    failed = 0
    skipped = 0
    
    logger.info(f"📥 开始下载 {cn_name} ({role_name}): 当前 {current_count} 张，需要补充 {needed_count} 张")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, url in enumerate(urls[:need_download + 50]):  # 多下载一些以防失败
            save_path = os.path.join(role_dir, f'{i+1:04d}.jpg')
            if os.path.exists(save_path):
                skipped += 1
                continue
            futures.append(executor.submit(download_image, url, save_path))
        
        for idx, future in enumerate(as_completed(futures)):
            if future.result():
                downloaded += 1
                if downloaded >= need_download:
                    break
            else:
                failed += 1
            
            if (idx + 1) % 20 == 0:
                logger.info(f"  [{downloaded+failed}/{need_download}]")
    
    logger.info(f"✅ {cn_name} 下载完成: 成功 {downloaded}, 跳过 {skipped}, 失败 {failed}")
    return {'success': downloaded, 'failed': failed, 'skipped': skipped}

def main():
    logger = setup_logger('download_priority_roles')
    
    logger.info("=" * 60)
    logger.info("优先下载数据不足的角色")
    logger.info("=" * 60)
    
    total_success = 0
    total_failed = 0
    
    for cn_name, info in PRIORITY_ROLES.items():
        en_name = info['en_name']
        needed = info['needed']
        
        current_count = get_current_image_count(en_name)
        actual_needed = max(0, needed)
        
        if actual_needed <= 0:
            logger.info(f"⏭️ 跳过 {cn_name} ({en_name}): 当前 {current_count} 张，已满足")
            continue
        
        result = download_role_images(en_name, cn_name, actual_needed)
        total_success += result['success']
        total_failed += result['failed']
        
        # 更新需要的数量
        PRIORITY_ROLES[cn_name]['needed'] = max(0, needed - result['success'])
        
        time.sleep(1)  # 避免请求过快
    
    logger.info("=" * 60)
    logger.info("优先下载任务完成")
    logger.info(f"总计: 成功 {total_success}, 失败 {total_failed}")
    logger.info("=" * 60)
    
    # 输出最终状态
    logger.info("\n📊 最终状态:")
    for cn_name, info in PRIORITY_ROLES.items():
        current = get_current_image_count(info['en_name'])
        status = "✅" if current >= 100 else "⚠️"
        logger.info(f"  {status} {cn_name}: {current} 张")

if __name__ == '__main__':
    main()