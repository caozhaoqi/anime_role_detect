#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZIP文件数据采集系统
从ZIP文件中提取图片数据
"""

import os
import requests
import zipfile
import io
from PIL import Image
import time
from pathlib import Path

# 配置参数
DATA_DIR = "./data/downloaded_images"
ZIP_FILE = "./data/href_url/arona_zip.txt"
TEMP_DIR = "./data/temp_zip_extract"

def download_and_extract_zip(zip_url):
    """下载并解压ZIP文件"""
    try:
        print(f"正在下载: {zip_url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(zip_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # 从内存中读取ZIP文件
            zip_content = io.BytesIO(response.content)
            
            # 解压ZIP文件
            with zipfile.ZipFile(zip_content, 'r') as zip_ref:
                # 创建临时目录
                os.makedirs(TEMP_DIR, exist_ok=True)
                
                # 解压所有文件
                zip_ref.extractall(TEMP_DIR)
                
                print(f"  ✓ 解压成功")
                return True
        else:
            print(f"  ✗ 下载失败: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ✗ 处理失败: {str(e)}")
        return False

def process_extracted_files():
    """处理解压后的文件"""
    extracted_images = []
    
    if not os.path.exists(TEMP_DIR):
        return extracted_images
    
    # 遍历解压后的文件
    for root, dirs, files in os.walk(TEMP_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            
            # 检查是否为图片文件
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                try:
                    # 验证图片
                    img = Image.open(file_path)
                    img.verify()
                    
                    # 重新打开以获取图片信息
                    img = Image.open(file_path).convert('RGB')
                    
                    # 检查图片尺寸
                    if img.width >= 100 and img.height >= 100:
                        extracted_images.append(file_path)
                        
                except Exception as e:
                    print(f"  ✗ 无效图片: {file} - {str(e)}")
    
    return extracted_images

def classify_and_save_images(image_paths):
    """分类并保存图片"""
    # 简单分类逻辑 - 默认保存到阿罗娜目录
    target_dir = os.path.join(DATA_DIR, "阿罗娜")
    os.makedirs(target_dir, exist_ok=True)
    
    saved_count = 0
    existing_files = set(os.listdir(target_dir))
    
    for img_path in image_paths:
        try:
            # 生成新文件名
            url_hash = abs(hash(img_path)) % 1000000
            filename = f"{url_hash:06d}.jpg"
            
            # 避免重复
            if filename in existing_files:
                continue
            
            # 保存图片
            target_path = os.path.join(target_dir, filename)
            
            # 读取并保存图片
            img = Image.open(img_path).convert('RGB')
            img.save(target_path, 'JPEG', quality=85)
            
            saved_count += 1
            existing_files.add(filename)
            
            if saved_count % 10 == 0:
                print(f"  已保存: {saved_count} 张")
                
        except Exception as e:
            print(f"  ✗ 保存失败: {img_path} - {str(e)}")
    
    return saved_count

def cleanup_temp_files():
    """清理临时文件"""
    try:
        if os.path.exists(TEMP_DIR):
            import shutil
            shutil.rmtree(TEMP_DIR)
            print(f"✓ 临时文件已清理")
    except Exception as e:
        print(f"✗ 清理失败: {str(e)}")

def main():
    """主函数"""
    print("=" * 60)
    print("ZIP文件数据采集系统")
    print("=" * 60)
    
    if not os.path.exists(ZIP_FILE):
        print(f"错误: ZIP文件列表不存在: {ZIP_FILE}")
        return
    
    # 读取ZIP文件URL
    with open(ZIP_FILE, 'r', encoding='utf-8') as f:
        zip_urls = [line.strip() for line in f if line.strip()]
    
    print(f"找到 {len(zip_urls)} 个ZIP文件")
    
    total_images = 0
    
    for i, zip_url in enumerate(zip_urls, start=1):
        print(f"\n处理第 {i}/{len(zip_urls)} 个ZIP文件:")
        
        # 下载并解压
        if download_and_extract_zip(zip_url):
            # 处理解压后的文件
            extracted_images = process_extracted_files()
            print(f"  提取到 {len(extracted_images)} 张图片")
            
            if extracted_images:
                # 分类并保存图片
                saved_count = classify_and_save_images(extracted_images)
                print(f"  成功保存: {saved_count} 张")
                total_images += saved_count
            
            # 清理临时文件
            cleanup_temp_files()
        
        # 避免请求过快
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("ZIP文件采集完成")
    print("=" * 60)
    print(f"总共保存: {total_images} 张图片")
    print("=" * 60)

if __name__ == "__main__":
    main()
