#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量补充数据不足的角色
"""
import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 角色拼音映射
ROLE_PINYIN_MAP = {
    'Himesaka': ['ji1ban3nai3ai4'],
    'Tsukiyo': ['yue4qian1ye4', 'tsukiyo'],
    'Clara': ['ke4la1la1', 'clara'],
    'Nagan': ['na4gan1'],
    'Koshenia': ['ke1she4ni2ya4'],
    'Shirosaki': ['bai2xiao4hua1'],
    'Shakri': ['xia4ke4li3'],
    'March': ['san1yue4qi1'],
    'Tanemura': ['zhong3cun1xiao3yi1'],
    'Paimon': ['pai4meng1'],
    'Dream!': ['zao3lai4you1xiang1'],
    'Yanagi': ['mao1gong1you4nai4'],
    'Hoshino': ['xiao3niao3you2xing1ye4'],
    'Hinaatsu': ['chu2he4ai4'],
    'Suzuran': ['ling2lan2'],
    'Mika': ['sheng4yuan2wei4hua1'],
    'Sayu': ['zao3you4'],
    'Fu': ['fu2xuan2'],
    'Konomori': ['xiao3zhi1sen1xia4yin1'],
    'Ichinose': ['yi1zhi1lai4ming2ri4nai4']
}

SPIDER_DATA_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'
DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'

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

def download_role_images(role_name, pinyin_list, needed):
    """下载单个角色的图片"""
    # 查找URL文件
    url_file = None
    for pinyin in pinyin_list:
        fpath = os.path.join(SPIDER_DATA_DIR, f'{pinyin}_img.txt')
        if os.path.exists(fpath):
            url_file = fpath
            break
    
    if not url_file:
        print(f"❌ 未找到 {role_name} 的URL文件")
        return 0
    
    # 读取URL列表
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # 获取当前图片数量
    role_dir = os.path.join(DATASET_PATH, role_name)
    os.makedirs(role_dir, exist_ok=True)
    current_count = len([f for f in os.listdir(role_dir) if f.lower().endswith('.jpg')])
    
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for i, url in enumerate(urls[:needed + 100]):
            save_path = os.path.join(role_dir, f'{role_name}_{current_count + i + 1:04d}.jpg')
            if os.path.exists(save_path):
                continue
            future = executor.submit(download_image, url, save_path)
            futures[future] = (save_path, i)
        
        for future in as_completed(futures):
            if future.result():
                success_count += 1
                if success_count >= needed:
                    break
            else:
                failed_count += 1
    
    print(f"  ✅ 下载完成: 成功 {success_count}, 失败 {failed_count}")
    return success_count

def main():
    print("📥 开始批量补充角色图片")
    print("=" * 80)
    
    total_success = 0
    
    for role_name, pinyin_list in ROLE_PINYIN_MAP.items():
        role_dir = os.path.join(DATASET_PATH, role_name)
        if not os.path.isdir(role_dir):
            continue
        
        current_count = len([f for f in os.listdir(role_dir) if f.lower().endswith('.jpg')])
        needed = max(0, 100 - current_count)
        
        if needed == 0:
            print(f"⏭️ {role_name}: 已满足要求 ({current_count} 张)")
            continue
        
        print(f"\n📥 下载 {role_name}: 当前 {current_count} 张，需要补充 {needed} 张")
        success = download_role_images(role_name, pinyin_list, needed)
        total_success += success
        
        # 更新数量
        current_count = len([f for f in os.listdir(role_dir) if f.lower().endswith('.jpg')])
        if current_count >= 100:
            print(f"🎉 {role_name} 已满足要求 ({current_count} 张)")
    
    print("\n" + "=" * 80)
    print(f"✅ 批量下载完成，总计成功 {total_success} 张")

if __name__ == '__main__':
    main()