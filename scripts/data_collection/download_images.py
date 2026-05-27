#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定时下载采集到的图片URL
支持并发下载、断点续传、错误重试
"""

import os
import sys
import time
import json
import shutil
import logging
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# 配置
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/spider_image_system/data/img_url'
DOWNLOAD_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/downloaded_images'
LOG_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/logs/download'

# 下载配置
MAX_WORKERS = 10  # 并发下载数
TIMEOUT = 30  # 请求超时时间（秒）
RETRY_COUNT = 3  # 重试次数
SKIP_EXISTING = True  # 是否跳过已存在的文件

# 设置日志
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"download_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def download_image(url, save_path):
    """下载单张图片"""
    for attempt in range(RETRY_COUNT):
        try:
            # 添加User-Agent模拟浏览器
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, timeout=TIMEOUT, stream=True, headers=headers)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 验证文件大小
            if os.path.getsize(save_path) < 100:
                os.remove(save_path)
                return False, "文件太小，可能是占位图"
                
            return True, None
        except requests.exceptions.RequestException as e:
            if attempt < RETRY_COUNT - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            return False, str(e)
        except Exception as e:
            if attempt < RETRY_COUNT - 1:
                time.sleep(2 ** attempt)
                continue
            return False, f"未知错误: {str(e)}"
    return False, "达到最大重试次数"


def download_role_images(role_name, url_file, target_dir):
    """下载单个角色的所有图片"""
    role_dir = os.path.join(target_dir, role_name)
    os.makedirs(role_dir, exist_ok=True)
    
    # 读取URL列表
    with open(url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    errors = []
    
    logger.info(f"开始下载角色 [{role_name}]: {len(urls)} 张图片")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        
        for url in urls:
            # 生成文件名
            try:
                filename = os.path.basename(url).split('?')[0]
                save_path = os.path.join(role_dir, filename)
                
                # 跳过已存在的文件
                if SKIP_EXISTING and os.path.exists(save_path):
                    skip_count += 1
                    continue
                
                futures.append(executor.submit(download_image, url, save_path))
            except Exception as e:
                logger.warning(f"跳过无效URL: {url} - {e}")
                fail_count += 1
        
        # 等待所有任务完成
        for future in as_completed(futures):
            success, error = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                errors.append(error)
    
    logger.info(f"角色 [{role_name}] 下载完成: 成功 {success_count}, 失败 {fail_count}, 跳过 {skip_count}")
    
    return {
        'role': role_name,
        'total': len(urls),
        'success': success_count,
        'fail': fail_count,
        'skip': skip_count,
        'errors': errors[:5]  # 只保留前5个错误
    }


def main(args):
    """主函数"""
    # 创建目标目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    target_dir = os.path.join(DOWNLOAD_DIR, f"batch_{timestamp}")
    os.makedirs(target_dir, exist_ok=True)
    
    logger.info(f"📥 开始批量下载图片")
    logger.info(f"目标目录: {target_dir}")
    logger.info(f"并发数: {MAX_WORKERS}")
    logger.info(f"跳过已存在: {SKIP_EXISTING}")
    
    # 获取URL文件列表
    url_files = []
    if args.role:
        # 只下载指定角色
        pinyin = ''.join([str(ord(c)) for c in args.role])  # 简单拼音转换
        pattern = f"*{args.role}*_img.txt"
        import glob
        url_files = glob.glob(os.path.join(URL_DIR, pattern))
        if not url_files:
            logger.error(f"未找到角色 [{args.role}] 的URL文件")
            return
    else:
        # 下载所有角色
        url_files = sorted([f for f in os.listdir(URL_DIR) if f.endswith('_img.txt')])
        url_files = [os.path.join(URL_DIR, f) for f in url_files]
    
    logger.info(f"发现 {len(url_files)} 个角色URL文件")
    
    # 逐个角色下载
    results = []
    total_success = 0
    total_fail = 0
    total_skip = 0
    
    start_time = time.time()
    
    for i, url_file in enumerate(url_files, 1):
        role_name = os.path.basename(url_file).replace('_img.txt', '')
        
        # 进度显示
        logger.info(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[{i}/{len(url_files)}] 处理角色: {role_name}")
        logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        result = download_role_images(role_name, url_file, target_dir)
        results.append(result)
        
        total_success += result['success']
        total_fail += result['fail']
        total_skip += result['skip']
        
        # 每处理10个角色输出一次统计
        if i % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(f"\n📊 进度统计: {i}/{len(url_files)}")
            logger.info(f"已下载: {total_success} 张")
            logger.info(f"失败: {total_fail} 张")
            logger.info(f"跳过: {total_skip} 张")
            logger.info(f"耗时: {elapsed:.2f} 秒")
    
    # 输出最终统计
    elapsed = time.time() - start_time
    logger.info("\n" + "="*50)
    logger.info("📋 下载完成统计")
    logger.info("="*50)
    logger.info(f"处理角色数: {len(results)}")
    logger.info(f"总下载成功: {total_success}")
    logger.info(f"总下载失败: {total_fail}")
    logger.info(f"总跳过: {total_skip}")
    logger.info(f"总耗时: {elapsed:.2f} 秒")
    logger.info(f"平均速度: {total_success/elapsed:.2f} 张/秒")
    logger.info("="*50)
    
    # 保存统计结果
    stats_file = os.path.join(target_dir, 'download_stats.json')
    stats = {
        'timestamp': timestamp,
        'total_roles': len(results),
        'total_success': total_success,
        'total_fail': total_fail,
        'total_skip': total_skip,
        'elapsed_seconds': elapsed,
        'per_role': results
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"统计文件已保存: {stats_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量下载采集到的角色图片')
    parser.add_argument('--role', type=str, help='指定角色名称（可选，不指定则下载所有）')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='并发下载数')
    parser.add_argument('--no-skip', action='store_true', help='不跳过已存在的文件')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.workers:
        MAX_WORKERS = args.workers
    if args.no_skip:
        SKIP_EXISTING = False
    
    main(args)
