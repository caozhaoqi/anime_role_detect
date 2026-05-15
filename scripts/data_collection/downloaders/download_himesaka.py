#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门下载姬坂乃爱的图片
"""
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception:
        return False

def main():
    DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
    SPIDER_DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'
    
    role_name = 'Himesaka'
    cn_name = '姬坂乃爱'
    pinyin = 'ji1ban3nai3ai4'
    
    # 读取URL文件
    url_file = os.path.join(SPIDER_DATA_DIR, f'{pinyin}_img.txt')
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # 获取当前图片数量
    role_dir = os.path.join(DATASET_PATH, role_name)
    os.makedirs(role_dir, exist_ok=True)
    current_count = len([f for f in os.listdir(role_dir) if f.lower().endswith('.jpg')])
    
    print(f"📥 下载 {cn_name} ({role_name}):")
    print(f"  当前: {current_count} 张")
    print(f"  URL数: {len(urls)} 个")
    
    # 下载需要的数量
    needed = max(0, 100 - current_count)
    print(f"  需要补充: {needed} 张")
    
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for i, url in enumerate(urls[:needed + 50]):
            save_path = os.path.join(role_dir, f'{role_name}_{current_count + i + 1:04d}.jpg')
            if os.path.exists(save_path):
                continue
            future = executor.submit(download_image, url, save_path)
            futures[future] = (save_path, i)
        
        for future in as_completed(futures):
            save_path, idx = futures[future]
            if future.result():
                success_count += 1
                if success_count % 10 == 0:
                    print(f"  [{success_count}/{needed}]")
                if success_count >= needed:
                    break
            else:
                failed_count += 1
    
    print(f"\n✅ 下载完成: 成功 {success_count}, 失败 {failed_count}")
    final_count = current_count + success_count
    print(f"📊 当前总量: {final_count} 张")
    
    if final_count >= 100:
        print("🎉 姬坂乃爱数据已满足要求!")

if __name__ == '__main__':
    main()