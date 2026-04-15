#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据采集脚本
从URL文件下载图片到训练数据目录
"""

import os
import requests
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image
import io
import time

# 配置参数
URL_FILE = "./data/img_url/arona_img.txt"
DATA_DIR = "./data/downloaded_images"
ARONA_DIR = os.path.join(DATA_DIR, "a1luo2na4")
RINAI_DIR = os.path.join(DATA_DIR, "ri4nai4")

# 创建目录
os.makedirs(ARONA_DIR, exist_ok=True)
os.makedirs(RINAI_DIR, exist_ok=True)

def is_valid_image(content):
    """检查是否为有效的图片"""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except:
        return False

def download_image(url, save_dir, index, timeout=10):
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
                filename = f"image_{index:04d}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                # 保存图片
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"✓ 下载成功: {filename}")
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
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"找到 {len(urls)} 个URL")
    print(f"阿罗娜目录: {ARONA_DIR}")
    print(f"日奈目录: {RINAI_DIR}")
    print()
    
    # 统计现有图片数量
    existing_arona = len([f for f in os.listdir(ARONA_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
    existing_rinai = len([f for f in os.listdir(RINAI_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"现有阿罗娜图片: {existing_arona} 张")
    print(f"现有日奈图片: {existing_rinai} 张")
    print()
    
    # 下载图片
    success_count = 0
    fail_count = 0
    
    for i, url in enumerate(urls, start=1):
        # 简单的分类逻辑：根据URL中的关键词
        # 这里假设所有图片都是阿罗娜的，可以根据实际情况调整
        save_dir = ARONA_DIR
        
        # 如果URL中包含特定关键词，可以分类到日奈目录
        # 例如：if 'rinai' in url.lower(): save_dir = RINAI_DIR
        
        print(f"[{i}/{len(urls)}] 正在下载: {url}")
        
        if download_image(url, save_dir, i):
            success_count += 1
        else:
            fail_count += 1
        
        # 避免请求过快
        time.sleep(0.5)
    
    # 统计结果
    print()
    print("=" * 60)
    print("数据采集完成")
    print("=" * 60)
    print(f"成功下载: {success_count} 张")
    print(f"下载失败: {fail_count} 张")
    
    # 统计最终图片数量
    final_arona = len([f for f in os.listdir(ARONA_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
    final_rinai = len([f for f in os.listdir(RINAI_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"最终阿罗娜图片: {final_arona} 张")
    print(f"最终日奈图片: {final_rinai} 张")
    print(f"总图片数量: {final_arona + final_rinai} 张")
    print("=" * 60)

if __name__ == "__main__":
    main()
