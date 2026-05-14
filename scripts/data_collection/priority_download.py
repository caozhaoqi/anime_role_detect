#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优先下载图片数量少于50张的角色
"""

import os
import subprocess
import time

def get_priority_roles():
    """获取需要优先下载的角色列表（图片少于50张）"""
    role_file = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt'
    image_dirs = [
        '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset',
        '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/merged_english_dataset',
        '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'
    ]

    priority_roles = []
    with open(role_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    english_name = parts[2]
                    total_count = 0
                    for img_dir in image_dirs:
                        role_dir = os.path.join(img_dir, english_name)
                        if os.path.exists(role_dir) and os.path.isdir(role_dir):
                            try:
                                files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
                                total_count += len(files)
                            except:
                                pass
                    if total_count < 50:
                        priority_roles.append((english_name, total_count))
    
    # 按图片数量排序（少的优先）
    priority_roles.sort(key=lambda x: x[1])
    return priority_roles

def download_role_images(role_name):
    """下载指定角色的图片"""
    print(f"\n=== 开始下载: {role_name} ===")
    
    # 调用下载API或脚本
    try:
        # 使用curl调用下载API
        cmd = [
            'curl',
            '-X', 'POST',
            'http://localhost:8000/api/v1/download/start',
            '-H', 'Content-Type: application/json',
            '-d', f'{{"role_name": "{role_name}", "max_count": 100}}'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"API响应: {result.stdout}")
        if result.stderr:
            print(f"错误: {result.stderr}")
        
        # 等待下载完成（假设每个角色下载最多5分钟）
        time.sleep(300)
        
    except Exception as e:
        print(f"下载失败: {e}")

def main():
    print("=== 优先下载任务启动 ===")
    print("目标: 下载图片少于50张的角色")
    
    priority_roles = get_priority_roles()
    print(f"\n需要优先下载的角色: {len(priority_roles)} 个")
    
    for i, (role_name, current_count) in enumerate(priority_roles, 1):
        print(f"\n{i}/{len(priority_roles)}: {role_name} (当前: {current_count} 张)")
        download_role_images(role_name)
    
    print("\n=== 优先下载任务完成 ===")

if __name__ == '__main__':
    main()
