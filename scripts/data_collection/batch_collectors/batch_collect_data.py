#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量数据采集脚本
从spider_image_system数据源批量采集所有角色的图片
"""

import os
import requests
from PIL import Image
import io
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置参数
DATA_DIR = "./data/downloaded_images"
SPIDER_DATA_DIR = "./spider_image_system/data/img_url"

# 角色映射 (拼音 -> 中文名)
ROLE_MAPPING = {
    "a1luo2na4": "阿罗娜",
    "ri4nai4": "日奈", 
    "pu3la1na4": "普拉娜",
    "ya4zi": "亚子",
    "yi1zhi1": "伊织",
    "qian1xia4": "千夏",
    "yi1lv3bo1": "伊吕波",
    "a1lu4": "阿露",
    "mu4yue4": "睦月",
    "jia1dai4zi": "佳代子",
    "xing1ye3": "星野",
    "xiao3chun1": "小春",
    "xiao3xia4": "小夏",
    "qiu1nai3": "秋奈",
    "dong1you1zi": "冬优子",
    "hua1yin1": "花音",
    "ling2yin1": "铃音",
    "li3shi4": "莉丝",
    "zhen1bu4": "真部",
    "jing4hua2": "镜华",
    "kan1": "看",
    "zhi4": "知",
    "ren3": "人",
    "shen1yue4": "神月"
}

def is_valid_image(content):
    """检查是否为有效图片"""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True
    except:
        return False

def download_image(url, save_dir, role_name, timeout=15):
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
                
                chinese_name = ROLE_MAPPING.get(role_name, role_name)
                return True, f"{filename} ({chinese_name})"
            else:
                return False, "无效图片"
        else:
            return False, f"HTTP {response.status_code}"
            
    except Exception as e:
        return False, str(e)

def process_role_data(role_file):
    """处理单个角色的数据采集"""
    role_name = role_file.replace("_img.txt", "")
    
    print(f"\n处理角色: {role_name} ({ROLE_MAPPING.get(role_name, role_name)})")
    
    # 创建角色目录
    role_dir = os.path.join(DATA_DIR, role_name)
    os.makedirs(role_dir, exist_ok=True)
    
    # 检查文件是否存在
    file_path = os.path.join(SPIDER_DATA_DIR, role_file)
    if not os.path.exists(file_path):
        print(f"  ✗ 文件不存在: {file_path}")
        return 0, 0
    
    # 读取URL
    with open(file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"  找到 {len(urls)} 个URL")
    
    # 统计现有数量
    existing_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"  现有图片: {existing_count} 张")
    
    # 下载图片
    successful = 0
    failed = 0
    
    for i, url in enumerate(urls, start=1):
        success, result = download_image(url, role_dir, role_name)
        
        if success:
            successful += 1
            if successful % 10 == 0:
                print(f"  ✓ 进度: {successful}/{len(urls)}")
        else:
            if "文件已存在" not in result:
                failed += 1
        
        # 避免请求过快
        time.sleep(0.05)
    
    print(f"  ✓ 成功下载: {successful} 张")
    print(f"  ✗ 下载失败: {failed} 张")
    
    return successful, failed

def main():
    """主函数"""
    print("=" * 60)
    print("批量数据采集系统")
    print("=" * 60)
    
    # 获取所有角色文件
    if not os.path.exists(SPIDER_DATA_DIR):
        print(f"错误: 数据源目录不存在: {SPIDER_DATA_DIR}")
        return
    
    role_files = [f for f in os.listdir(SPIDER_DATA_DIR) if f.endswith("_img.txt")]
    print(f"找到 {len(role_files)} 个角色数据文件")
    
    # 统计现有数据
    print("\n现有数据统计:")
    for role_name in ROLE_MAPPING.keys():
        role_dir = os.path.join(DATA_DIR, role_name)
        if os.path.exists(role_dir):
            count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if count > 0:
                chinese_name = ROLE_MAPPING.get(role_name, role_name)
                print(f"  {chinese_name} ({role_name}): {count} 张")
    
    # 开始采集
    print(f"\n开始批量采集...")
    
    total_successful = 0
    total_failed = 0
    
    for role_file in role_files:
        successful, failed = process_role_data(role_file)
        total_successful += successful
        total_failed += failed
    
    # 最终统计
    print("\n" + "=" * 60)
    print("批量采集完成")
    print("=" * 60)
    print(f"总成功下载: {total_successful} 张")
    print(f"总下载失败: {total_failed} 张")
    
    print("\n最终数据统计:")
    for role_name in ROLE_MAPPING.keys():
        role_dir = os.path.join(DATA_DIR, role_name)
        if os.path.exists(role_dir):
            count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if count > 0:
                chinese_name = ROLE_MAPPING.get(role_name, role_name)
                print(f"  {chinese_name} ({role_name}): {count} 张")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
