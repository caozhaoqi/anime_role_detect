#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能数据采集脚本
从URL文件下载图片并自动分类到不同的角色目录
"""

import os
import requests
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image
import io
import time
import re

# 配置参数
URL_FILE = "./data/img_url/arona_img.txt"
DATA_DIR = "./data/downloaded_images"

# 角色配置
ROLES = {
    "a1luo2na4": {
        "dir": "a1luo2na4",
        "keywords": ["arona", "阿罗娜", "a1luo2na4"],
        "chinese_name": "阿罗娜"
    },
    "ri4nai4": {
        "dir": "ri4nai4", 
        "keywords": ["rinai", "日奈", "ri4nai4"],
        "chinese_name": "日奈"
    },
    "plana": {
        "dir": "plana",
        "keywords": ["plana", "普拉娜"],
        "chinese_name": "普拉娜"
    }
}

# 创建目录
for role_name, role_config in ROLES.items():
    role_dir = os.path.join(DATA_DIR, role_config["dir"])
    os.makedirs(role_dir, exist_ok=True)

def is_valid_image(content):
    """检查是否为有效的图片"""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except:
        return False

def classify_url(url):
    """根据URL分类图片到对应角色"""
    url_lower = url.lower()
    
    for role_name, role_config in ROLES.items():
        for keyword in role_config["keywords"]:
            if keyword.lower() in url_lower:
                return role_name
    
    # 默认分类到阿罗娜
    return "a1luo2na4"

def download_image(url, save_dir, index, role_name, timeout=10):
    """下载单张图片"""
    try:
        # 发送请求
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            # 检查是否为有效图片
            if is_valid_image(response.content):
                # 生成文件名
                filename = f"{role_name}_{index:04d}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                # 保存图片
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                chinese_name = ROLES[role_name]["chinese_name"]
                print(f"✓ 下载成功: {filename} ({chinese_name})")
                return True
            else:
                print(f"✗ 无效图片: {url}")
                return False
        else:
            print(f"✗ 下载失败 (状态码 {response.status_code}): {url}")
            return False
            
    except Exception as e:
        print(f"✗ 下载错误: {url} - {str(e)}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("开始采集训练数据")
    print("=" * 60)
    
    # 读取URL文件
    if not os.path.exists(URL_FILE):
        print(f"错误: URL文件不存在: {URL_FILE}")
        return
    
    with open(URL_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and line.startswith('http')]
    
    print(f"找到 {len(urls)} 个URL")
    print()
    
    # 统计现有图片数量
    print("现有数据统计:")
    for role_name, role_config in ROLES.items():
        role_dir = os.path.join(DATA_DIR, role_config["dir"])
        existing_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {role_config['chinese_name']}: {existing_count} 张")
    print()
    
    # 下载图片
    success_count = 0
    fail_count = 0
    role_stats = {role_name: 0 for role_name in ROLES.keys()}
    
    for i, url in enumerate(urls, start=1):
        # 分类URL到对应角色
        role_name = classify_url(url)
        save_dir = os.path.join(DATA_DIR, ROLES[role_name]["dir"])
        
        print(f"[{i}/{len(urls)}] 正在下载: {url}")
        
        if download_image(url, save_dir, i, role_name):
            success_count += 1
            role_stats[role_name] += 1
        else:
            fail_count += 1
        
        # 避免请求过快
        time.sleep(0.3)
    
    # 统计结果
    print()
    print("=" * 60)
    print("数据采集完成")
    print("=" * 60)
    print(f"成功下载: {success_count} 张")
    print(f"下载失败: {fail_count} 张")
    print()
    
    print("本次下载统计:")
    for role_name, count in role_stats.items():
        chinese_name = ROLES[role_name]["chinese_name"]
        print(f"  {chinese_name}: {count} 张")
    print()
    
    print("最终数据统计:")
    for role_name, role_config in ROLES.items():
        role_dir = os.path.join(DATA_DIR, role_config["dir"])
        final_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {role_config['chinese_name']}: {final_count} 张")
    
    total_count = sum(len([f for f in os.listdir(os.path.join(DATA_DIR, config["dir"])) if f.endswith(('.jpg', '.jpeg', '.png'))]) for config in ROLES.values())
    print(f"总图片数量: {total_count} 张")
    print("=" * 60)

if __name__ == "__main__":
    main()
