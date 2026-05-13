#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能去重图片下载脚本（修复HTTP 403问题）
- 添加模拟浏览器请求头
- 支持MD5去重
- 支持文件大小去重
- 按角色分类存储
- 断点续传
"""

import os
import sys
import hashlib
import requests
import time
import logging
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 配置
URL_DIR = 'spider_image_system/data/img_url_english'
OUTPUT_DIR = 'data/merged_english_dataset'
MAX_WORKERS = 4  # 降低并发避免被封
TIMEOUT = 30
RETRY_MAX = 3

# 模拟浏览器请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.google.com/',
}

# 已下载文件的MD5缓存
md5_cache = set()
size_cache = set()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_md5(file_path):
    """计算文件MD5"""
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def is_duplicate(file_path):
    """检查文件是否重复"""
    if not os.path.exists(file_path):
        return False
    
    file_size = os.path.getsize(file_path)
    
    # 大小去重
    if file_size in size_cache:
        return True
    
    # MD5去重
    md5 = calculate_md5(file_path)
    if md5 in md5_cache:
        return True
    
    # 添加到缓存
    size_cache.add(file_size)
    md5_cache.add(md5)
    return False


def download_image(url, save_path):
    """下载单张图片"""
    for retry in range(RETRY_MAX):
        try:
            response = requests.get(url, timeout=TIMEOUT, stream=True, headers=HEADERS)
            if response.status_code == 200:
                # 先保存到临时文件
                temp_path = save_path + '.tmp'
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # 检查是否重复
                if is_duplicate(temp_path):
                    os.remove(temp_path)
                    return {'status': 'duplicate', 'url': url}
                
                # 重命名为最终文件名
                os.rename(temp_path, save_path)
                return {'status': 'success', 'url': url}
            elif response.status_code == 403:
                # 403错误，增加等待时间后重试
                time.sleep(5 * (retry + 1))
                continue
            else:
                return {'status': 'failed', 'url': url, 'reason': f"HTTP {response.status_code}"}
        except Exception as e:
            if retry < RETRY_MAX - 1:
                time.sleep(2 ** retry)
                continue
            return {'status': 'failed', 'url': url, 'reason': str(e)}


def get_urls_from_file(file_path):
    """从文件中读取URL列表"""
    urls = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url and url.startswith('http'):
                    urls.append(url)
    return urls


def download_role_images(role_name, urls, output_dir):
    """下载单个角色的图片"""
    role_dir = os.path.join(output_dir, role_name)
    os.makedirs(role_dir, exist_ok=True)
    
    # 获取已下载的文件列表
    existing_files = set(os.listdir(role_dir))
    
    results = {
        'success': 0,
        'failed': 0,
        'duplicate': 0,
        'skipped': 0
    }
    
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, url in enumerate(urls, 1):
            # 生成文件名
            ext = os.path.splitext(urlparse(url).path)[1] or '.jpg'
            filename = f"{i:04d}{ext}"
            save_path = os.path.join(role_dir, filename)
            
            # 跳过已存在的文件
            if filename in existing_files:
                results['skipped'] += 1
                continue
            
            tasks.append(executor.submit(download_image, url, save_path))
        
        # 处理结果
        for future in tqdm(as_completed(tasks), total=len(tasks), desc=f"下载 {role_name}"):
            result = future.result()
            results[result['status']] += 1
    
    return results


def load_existing_hashes(output_dir):
    """加载已存在文件的哈希值"""
    logger.info("加载已存在文件的哈希值...")
    for role_name in os.listdir(output_dir):
        role_dir = os.path.join(output_dir, role_name)
        if os.path.isdir(role_dir):
            for filename in os.listdir(role_dir):
                file_path = os.path.join(role_dir, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    size_cache.add(file_size)
                    md5 = calculate_md5(file_path)
                    md5_cache.add(md5)
    
    logger.info(f"已加载 {len(md5_cache)} 个文件哈希值")


def main():
    # 加载已存在的文件哈希
    load_existing_hashes(OUTPUT_DIR)
    
    # 获取所有URL文件
    url_files = []
    if os.path.exists(URL_DIR):
        for filename in os.listdir(URL_DIR):
            if filename.endswith('_img.txt'):
                role_name = filename.replace('_img.txt', '')
                url_files.append((role_name, os.path.join(URL_DIR, filename)))
    
    logger.info(f"发现 {len(url_files)} 个角色URL文件")
    
    # 逐个角色下载
    total_stats = {
        'success': 0,
        'failed': 0,
        'duplicate': 0,
        'skipped': 0
    }
    
    for role_name, url_file in url_files:
        urls = get_urls_from_file(url_file)
        if not urls:
            logger.info(f"{role_name}: 无URL需要下载")
            continue
        
        logger.info(f"\n开始下载 {role_name}: {len(urls)} 个URL")
        results = download_role_images(role_name, urls, OUTPUT_DIR)
        
        # 更新统计
        for key in total_stats:
            total_stats[key] += results[key]
        
        logger.info(f"{role_name} 完成: 成功={results['success']}, 失败={results['failed']}, "
                    f"重复={results['duplicate']}, 跳过={results['skipped']}")
    
    # 输出汇总
    logger.info("\n" + "="*60)
    logger.info("📊 下载完成汇总")
    logger.info("="*60)
    logger.info(f"✅ 成功: {total_stats['success']}")
    logger.info(f"❌ 失败: {total_stats['failed']}")
    logger.info(f"🔄 重复跳过: {total_stats['duplicate']}")
    logger.info(f"⏭️ 已存在跳过: {total_stats['skipped']}")
    logger.info(f"📁 去重缓存大小: {len(md5_cache)}")


if __name__ == '__main__':
    main()
