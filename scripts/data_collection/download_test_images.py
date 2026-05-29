#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载测试图片并运行基准测试
"""

import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 配置
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/img_url'
DOWNLOAD_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/downloaded_images'
MAX_WORKERS = 10
TIMEOUT = 30
RETRY_COUNT = 3

def download_image(url, save_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for attempt in range(RETRY_COUNT):
        try:
            response = requests.get(url, timeout=TIMEOUT, stream=True, headers=headers)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            if os.path.getsize(save_path) < 100:
                os.remove(save_path)
                return False, "文件太小"
            
            return True, None
        except Exception as e:
            if attempt < RETRY_COUNT - 1:
                time.sleep(2 ** attempt)
                continue
            return False, str(e)
    return False, "达到最大重试次数"

def download_images():
    """下载所有URL文件中的图片"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    target_dir = os.path.join(DOWNLOAD_DIR, f"batch_{timestamp}")
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"📥 开始下载测试图片")
    print(f"目标目录: {target_dir}")
    
    # 获取URL文件列表
    url_files = [f for f in os.listdir(URL_DIR) if f.endswith('_img.txt')]
    
    if not url_files:
        print("❌ 未找到URL文件")
        return None
    
    total_success = 0
    total_fail = 0
    
    for url_file in url_files:
        role_name = url_file.replace('_img.txt', '')
        role_dir = os.path.join(target_dir, role_name)
        os.makedirs(role_dir, exist_ok=True)
        
        with open(os.path.join(URL_DIR, url_file), 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and line.strip().endswith('.jpg')]
        
        print(f"\n📦 处理角色: {role_name} ({len(urls)} 张图片)")
        
        success = 0
        fail = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i, url in enumerate(urls[:10]):  # 每个角色最多下载10张
                filename = os.path.basename(url).split('?')[0]
                save_path = os.path.join(role_dir, filename)
                futures.append(executor.submit(download_image, url, save_path))
            
            for future in as_completed(futures):
                ok, _ = future.result()
                if ok:
                    success += 1
                else:
                    fail += 1
        
        print(f"✅ 完成: 成功 {success}, 失败 {fail}")
        total_success += success
        total_fail += fail
    
    print(f"\n📊 下载完成: 总成功 {total_success}, 总失败 {total_fail}")
    return target_dir

def run_benchmark():
    """运行基准测试"""
    print("\n🚀 运行基准测试...")
    benchmark_script = os.path.join(project_root, 'scripts', 'model_evaluation', 'benchmark_new_model.py')
    os.system(f"python3 {benchmark_script}")

def main():
    # 下载图片
    target_dir = download_images()
    
    if target_dir:
        # 更新benchmark脚本使用新数据目录
        benchmark_script = os.path.join(project_root, 'scripts', 'model_evaluation', 'benchmark_new_model.py')
        with open(benchmark_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 更新数据目录路径
        old_path = "self.data_dir = os.path.join(project_root, 'data', 'downloaded_images')"
        new_path = f"self.data_dir = '{target_dir}'"
        content = content.replace(old_path, new_path)
        
        with open(benchmark_script, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 运行基准测试
        run_benchmark()

if __name__ == "__main__":
    main()
