#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时统计下载进度并合并角色图片
"""

import os
import time
import shutil
from collections import defaultdict

# 配置
ROLE_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
SOURCE_DIRS = [
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset',
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/merged_english_dataset',
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images',
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/role_images',
    '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/src/data/downloaded_images'
]
DEST_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset'
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'

def load_role_list():
    """加载角色列表（英文名称）"""
    roles = {}
    with open(ROLE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    english_name = parts[2]
                    roles[english_name] = line  # 保存完整信息
    return roles

def count_images_in_dir(base_dir):
    """统计目录中每个角色的图片数量"""
    counts = defaultdict(int)
    if not os.path.exists(base_dir):
        return counts
    
    for role_name in os.listdir(base_dir):
        role_dir = os.path.join(base_dir, role_name)
        if os.path.isdir(role_dir):
            try:
                images = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
                counts[role_name] += len(images)
            except:
                pass
    return counts

def get_url_count(role_name):
    """获取角色的URL数量"""
    url_file = os.path.join(URL_DIR, f'{role_name}_img.txt')
    if os.path.exists(url_file):
        with open(url_file, 'r', encoding='utf-8') as f:
            return len([line for line in f if line.strip()])
    return 0

def merge_role_images(role_name):
    """合并角色图片到最终目录"""
    dest_role_dir = os.path.join(DEST_DIR, role_name)
    os.makedirs(dest_role_dir, exist_ok=True)
    
    total_copied = 0
    copied_files = set()
    
    for src_dir in SOURCE_DIRS:
        src_role_dir = os.path.join(src_dir, role_name)
        if os.path.exists(src_role_dir) and os.path.isdir(src_role_dir):
            for filename in os.listdir(src_role_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                    src_path = os.path.join(src_role_dir, filename)
                    dest_path = os.path.join(dest_role_dir, filename)
                    
                    # 避免重复复制
                    if filename not in copied_files:
                        try:
                            if not os.path.exists(dest_path):
                                shutil.copy2(src_path, dest_path)
                            copied_files.add(filename)
                            total_copied += 1
                        except Exception as e:
                            pass
    
    return total_copied

def print_progress():
    """打印实时进度"""
    roles = load_role_list()
    print(f"\n{'='*70}")
    print(f"实时下载进度统计 - 更新时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # 统计所有来源的图片
    all_counts = defaultdict(int)
    for src_dir in SOURCE_DIRS:
        counts = count_images_in_dir(src_dir)
        for role, count in counts.items():
            all_counts[role] += count
    
    # 按图片数量排序
    sorted_roles = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 打印统计
    total_images = 0
    roles_with_images = 0
    
    print(f"\n{'排名':<4} {'角色':<15} {'图片数':<8} {'URL数':<8} {'状态'}")
    print(f"{'-'*4} {'-'*15} {'-'*8} {'-'*8} {'-'*20}")
    
    for i, (role_name, count) in enumerate(sorted_roles[:20], 1):
        url_count = get_url_count(role_name)
        status = "✅" if count >= 100 else "⚠️" if count > 0 else "❌"
        print(f"{i:<4} {role_name:<15} {count:<8} {url_count:<8} {status}")
        total_images += count
        roles_with_images += 1
    
    print(f"\n{'='*70}")
    print(f"统计汇总:")
    print(f"  - 角色总数: {len(roles)}")
    print(f"  - 有图片的角色: {roles_with_images}")
    print(f"  - 图片总数: {total_images}")
    print(f"  - 平均每个角色: {round(total_images / roles_with_images, 1) if roles_with_images > 0 else 0} 张")
    
    # 统计图片少于50张的角色
    low_count_roles = [(r, c) for r, c in all_counts.items() if c < 50]
    print(f"\n  - 图片少于50张的角色: {len(low_count_roles)} 个")
    
    return all_counts

def main():
    print("=== 实时下载进度统计 ===")
    print("按 Ctrl+C 退出")
    
    try:
        while True:
            # 清屏
            os.system('clear')
            
            # 打印进度
            all_counts = print_progress()
            
            # 自动合并角色
            print("\n正在合并角色图片...")
            for role_name in all_counts:
                copied = merge_role_images(role_name)
                if copied > 0:
                    print(f"  {role_name}: 复制 {copied} 张图片")
            
            # 等待5秒后刷新
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n退出统计")

if __name__ == '__main__':
    main()
