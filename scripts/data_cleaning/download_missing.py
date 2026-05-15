#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充剩余数据不足的角色
"""
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

SPIDER_DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'
DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'

# 修正后的拼音映射
ROLE_PINYIN_MAP = {
    'Paimon': ['pai4meng2'],
    'Hoshino': ['xiao3niao3you2xing1ye3']
}

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
    for role_name, pinyin_list in ROLE_PINYIN_MAP.items():
        # 查找URL文件
        url_file = None
        for pinyin in pinyin_list:
            fpath = os.path.join(SPIDER_DATA_DIR, f'{pinyin}_img.txt')
            if os.path.exists(fpath):
                url_file = fpath
                break
        
        if not url_file:
            print(f"❌ 未找到 {role_name} 的URL文件")
            continue
        
        # 读取URL列表
        with open(url_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        # 获取当前图片数量
        role_dir = os.path.join(DATASET_PATH, role_name)
        current_count = len([f for f in os.listdir(role_dir) if f.lower().endswith('.jpg')])
        needed = max(0, 100 - current_count)
        
        if needed == 0:
            print(f"⏭️ {role_name}: 已满足要求 ({current_count} 张)")
            continue
        
        print(f"\n📥 下载 {role_name}: 当前 {current_count} 张，需要补充 {needed} 张")
        
        success_count = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for i, url in enumerate(urls[:needed + 50]):
                save_path = os.path.join(role_dir, f'{role_name}_{current_count + i + 1:04d}.jpg')
                if os.path.exists(save_path):
                    continue
                future = executor.submit(download_image, url, save_path)
                futures[future] = i
            
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                    if success_count >= needed:
                        break
        
        current_count = len([f for f in os.listdir(role_dir) if f.lower().endswith('.jpg')])
        print(f"✅ {role_name}: 成功 {success_count} 张，当前 {current_count} 张")

if __name__ == '__main__':
    main()