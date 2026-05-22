#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1. 检测重复图片
2. 建立数据表记录角色名和图片名
3. 规范化图片命名
4. 补充数据不足的角色
"""
import os
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# 配置
DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
OUTPUT_CSV = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/dataset_metadata.csv'
OUTPUT_JSON = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/dataset_metadata.json'
SPIDER_DATA_DIRS = [
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url',
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url_english'
]
TARGET_COUNT = 100
MAX_WORKERS = 10
TIMEOUT = 15

def calculate_md5(file_path):
    """计算文件MD5哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None

def detect_duplicates():
    """检测重复图片"""
    print("🔍 开始检测重复图片...")
    hash_to_files = {}
    duplicates = []
    
    for role_dir in os.listdir(DATASET_PATH):
        role_path = os.path.join(DATASET_PATH, role_dir)
        if not os.path.isdir(role_path) or role_dir.startswith('.'):
            continue
        
        for filename in os.listdir(role_path):
            if not filename.lower().endswith('.jpg'):
                continue
            
            file_path = os.path.join(role_path, filename)
            file_hash = calculate_md5(file_path)
            
            if file_hash:
                if file_hash not in hash_to_files:
                    hash_to_files[file_hash] = []
                hash_to_files[file_hash].append((role_dir, filename))
    
    # 找出重复
    for file_hash, files in hash_to_files.items():
        if len(files) > 1:
            duplicates.append({
                'hash': file_hash,
                'files': files
            })
    
    print(f"✅ 检测完成，发现 {len(duplicates)} 组重复图片")
    return duplicates, hash_to_files

def build_metadata_table():
    """建立数据表记录"""
    print("\n📋 建立数据表...")
    metadata = []
    
    for role_dir in sorted(os.listdir(DATASET_PATH)):
        role_path = os.path.join(DATASET_PATH, role_dir)
        if not os.path.isdir(role_path) or role_dir.startswith('.') or role_dir.endswith('.json'):
            continue
        
        for filename in sorted(os.listdir(role_path)):
            if not filename.lower().endswith('.jpg'):
                continue
            
            file_path = os.path.join(role_path, filename)
            file_size = os.path.getsize(file_path)
            
            metadata.append({
                'role_name': role_dir,
                'filename': filename,
                'file_size': file_size
            })
    
    # 保存为CSV
    with open(OUTPUT_CSV, 'w', encoding='utf-8') as f:
        f.write('role_name,filename,file_size\n')
        for item in metadata:
            f.write(f"{item['role_name']},{item['filename']},{item['file_size']}\n")
    
    # 保存为JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 数据表已保存，共 {len(metadata)} 条记录")
    return metadata

def normalize_naming():
    """规范化图片命名"""
    print("\n🔄 规范化图片命名...")
    renamed_count = 0
    
    for role_dir in os.listdir(DATASET_PATH):
        role_path = os.path.join(DATASET_PATH, role_dir)
        if not os.path.isdir(role_path) or role_dir.startswith('.'):
            continue
        
        jpg_files = sorted([f for f in os.listdir(role_path) if f.lower().endswith('.jpg')])
        
        for idx, filename in enumerate(jpg_files, 1):
            old_path = os.path.join(role_path, filename)
            new_name = f"{role_dir}_{idx:04d}.jpg"
            new_path = os.path.join(role_path, new_name)
            
            if filename != new_name:
                os.rename(old_path, new_path)
                renamed_count += 1
    
    print(f"✅ 命名规范化完成，重命名 {renamed_count} 个文件")

def download_missing_images():
    """补充数据不足的角色"""
    print("\n📥 补充数据不足的角色...")
    
    # 需要补充的角色
    roles_to补充 = {
        'Himesaka': {'cn_name': '姬坂乃爱', 'needed': 60}  # 还需60张
    }
    
    total_success = 0
    
    for en_name, info in roles_to补充.items():
        cn_name = info['cn_name']
        needed = info['needed']
        
        role_dir = os.path.join(DATASET_PATH, en_name)
        os.makedirs(role_dir, exist_ok=True)
        
        current_count = len([f for f in os.listdir(role_dir) if f.lower().endswith('.jpg')])
        if current_count >= TARGET_COUNT:
            print(f"⏭️ {cn_name} 已满足要求 ({current_count} 张)")
            continue
        
        actual_needed = max(0, TARGET_COUNT - current_count)
        print(f"📥 下载 {cn_name} ({en_name}): 当前 {current_count} 张，需要补充 {actual_needed} 张")
        
        # 查找URL文件
        url_file = None
        pinyin_variants = ['ji1ban3nai3ai4', 'ji1ban3nai4ai4', 'himesaka']
        
        for spider_dir in SPIDER_DATA_DIRS:
            for variant in pinyin_variants:
                fpath = os.path.join(spider_dir, f'{variant}_img.txt')
                if os.path.exists(fpath):
                    url_file = fpath
                    break
            if url_file:
                break
        
        if not url_file:
            print(f"❌ 未找到 {cn_name} 的URL文件")
            continue
        
        # 读取URL列表
        with open(url_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        downloaded = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i, url in enumerate(urls[:actual_needed + 50]):
                save_path = os.path.join(role_dir, f'{en_name}_{current_count + i + 1:04d}.jpg')
                if os.path.exists(save_path):
                    continue
                futures.append(executor.submit(download_image, url, save_path))
            
            for future in as_completed(futures):
                if future.result():
                    downloaded += 1
                    total_success += 1
                    if downloaded >= actual_needed:
                        break
        
        print(f"✅ {cn_name} 下载完成: 成功 {downloaded} 张")
    
    print(f"\n📊 补充下载总计: 成功 {total_success} 张")

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

def main():
    print("=" * 80)
    print("🧹 数据清洗与规范化")
    print("=" * 80)
    
    # 1. 检测重复图片
    duplicates, hash_to_files = detect_duplicates()
    if duplicates:
        print("\n⚠️ 发现重复图片:")
        for dup in duplicates[:5]:  # 只显示前5组
            print(f"  Hash: {dup['hash']}")
            for role, filename in dup['files']:
                print(f"    - {role}/{filename}")
        if len(duplicates) > 5:
            print(f"  ... 还有 {len(duplicates) - 5} 组重复")
    
    # 2. 建立数据表
    build_metadata_table()
    
    # 3. 规范化图片命名
    normalize_naming()
    
    # 4. 补充数据不足的角色
    download_missing_images()
    
    # 输出最终统计
    print("\n" + "=" * 80)
    print("📊 最终数据集统计")
    print("=" * 80)
    
    total_roles = 0
    total_images = 0
    low_count_roles = []
    
    for role_dir in os.listdir(DATASET_PATH):
        role_path = os.path.join(DATASET_PATH, role_dir)
        if not os.path.isdir(role_path) or role_dir.startswith('.') or role_dir.endswith('.json'):
            continue
        
        count = len([f for f in os.listdir(role_path) if f.lower().endswith('.jpg')])
        total_roles += 1
        total_images += count
        if count < TARGET_COUNT:
            low_count_roles.append((role_dir, count))
    
    print(f"总角色数: {total_roles}")
    print(f"总图片数: {total_images:,}")
    print(f"平均每角色: {total_images // total_roles} 张")
    
    if low_count_roles:
        print("\n⚠️ 数据不足的角色:")
        for role, count in low_count_roles:
            print(f"  {role}: {count} 张 (还差 {TARGET_COUNT - count} 张)")
    else:
        print("\n✅ 所有角色数据均已满足要求!")

if __name__ == '__main__':
    main()