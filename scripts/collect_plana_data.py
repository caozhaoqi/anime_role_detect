#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
普拉娜(plana)数据采集脚本
从spider_image_system数据源采集普拉娜的图片
"""

import os
import requests
from PIL import Image
import io
import time
import random

# 配置参数
DATA_DIR = "./data/downloaded_images"
PLANA_DATA_FILE = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url/a1luo2na4_img.txt"
ROLE_NAME = "pu3la1na4"
CHINESE_NAME = "普拉娜"

def is_valid_image(content):
    """检查是否为有效图片"""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except:
        return False

def download_image(url, save_dir, timeout=15):
    """下载单张图片"""
    try:
        headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ])
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            if is_valid_image(response.content):
                # 生成文件名
                url_hash = abs(hash(url)) % 1000000
                filename = f"{url_hash:06d}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                # 避免重复下载
                if os.path.exists(filepath):
                    return False, "文件已存在"
                
                # 保存图片
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                return True, f"{filename}"
            else:
                return False, "无效图片"
        else:
            return False, f"HTTP {response.status_code}"
            
    except Exception as e:
        return False, str(e)

def main():
    """主函数"""
    print("=" * 60)
    print(f"普拉娜({CHINESE_NAME})数据采集")
    print("=" * 60)
    
    # 创建普拉娜目录
    plana_dir = os.path.join(DATA_DIR, ROLE_NAME)
    os.makedirs(plana_dir, exist_ok=True)
    
    # 检查数据源文件
    if not os.path.exists(PLANA_DATA_FILE):
        print(f"错误: 数据源文件不存在: {PLANA_DATA_FILE}")
        return
    
    # 读取URL
    with open(PLANA_DATA_FILE, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"找到 {len(urls)} 个URL")
    
    # 统计现有数量
    existing_count = len([f for f in os.listdir(plana_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"现有图片: {existing_count} 张")
    
    # 下载图片
    successful = 0
    failed = 0
    
    for i, url in enumerate(urls, start=1):
        # 过滤非图片URL
        if not url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            failed += 1
            if i % 10 == 0:
                print(f"  进度: {i}/{len(urls)} (过滤非图片URL)")
            continue
        
        success, result = download_image(url, plana_dir)
        
        if success:
            successful += 1
            if successful % 5 == 0:
                print(f"  ✓ 进度: {i}/{len(urls)} (成功: {successful})")
        else:
            if "文件已存在" not in result:
                failed += 1
        
        # 避免请求过快
        time.sleep(0.1)
    
    # 最终统计
    final_count = len([f for f in os.listdir(plana_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print("\n" + "=" * 60)
    print("普拉娜数据采集完成")
    print("=" * 60)
    print(f"总URL数: {len(urls)} 个")
    print(f"成功下载: {successful} 张")
    print(f"下载失败: {failed} 张")
    print(f"现有图片: {final_count} 张")
    print(f"新增图片: {final_count - existing_count} 张")
    print("=" * 60)

if __name__ == "__main__":
    main()
