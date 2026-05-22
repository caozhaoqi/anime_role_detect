#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗脚本 - 过滤低质量图片，确保符合模型训练要求
"""
import os
import cv2
import numpy as np
from PIL import Image

# 配置
DATASET_PATH = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset'
MIN_WIDTH = 100
MIN_HEIGHT = 100
MIN_FILE_SIZE_KB = 5
MAX_FILE_SIZE_KB = 5000

def is_low_quality_image(file_path):
    """检查图片是否为低质量"""
    # 检查文件大小
    file_size_kb = os.path.getsize(file_path) / 1024
    if file_size_kb < MIN_FILE_SIZE_KB or file_size_kb > MAX_FILE_SIZE_KB:
        return True, f"文件大小异常 ({file_size_kb:.1f}KB)"
    
    try:
        # 尝试用PIL打开
        with Image.open(file_path) as img:
            width, height = img.size
            
            # 检查尺寸
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                return True, f"尺寸过小 ({width}x{height})"
            
            # 检查是否为损坏图片
            if img.format not in ['JPEG', 'PNG']:
                return True, f"格式不支持 ({img.format})"
            
            # 检查是否为空白图片
            img_array = np.array(img)
            if len(np.unique(img_array)) < 10:
                return True, "空白或纯色图片"
            
            # 检查是否为黑白图片
            if len(img_array.shape) == 2:
                return True, "灰度图片"
            
            # 检查颜色通道
            if img_array.shape[2] == 1:
                return True, "单通道灰度图片"
            
            # 检查是否为严重模糊（使用边缘检测）
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / (width * height)
            if edge_ratio < 0.01:
                return True, f"严重模糊 (边缘比例 {edge_ratio:.4f})"
            
            # 检查宽高比是否异常
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 5:
                return True, f"宽高比异常 ({width}x{height})"
            
    except Exception as e:
        return True, f"无法打开图片: {str(e)}"
    
    return False, "正常"

def clean_dataset():
    """清洗整个数据集"""
    total_files = 0
    cleaned_files = 0
    stats = []
    
    print("=" * 80)
    print("🧹 开始数据清洗")
    print("=" * 80)
    
    for role_dir in os.listdir(DATASET_PATH):
        role_path = os.path.join(DATASET_PATH, role_dir)
        if not os.path.isdir(role_path):
            continue
        
        role_cleaned = 0
        role_total = 0
        
        for filename in os.listdir(role_path):
            if not filename.lower().endswith('.jpg'):
                continue
            
            file_path = os.path.join(role_path, filename)
            role_total += 1
            total_files += 1
            
            is_bad, reason = is_low_quality_image(file_path)
            if is_bad:
                os.remove(file_path)
                role_cleaned += 1
                cleaned_files += 1
                print(f"❌ 删除 {role_dir}/{filename}: {reason}")
        
        if role_cleaned > 0:
            stats.append({
                'role': role_dir,
                'total': role_total,
                'cleaned': role_cleaned,
                'remaining': role_total - role_cleaned
            })
    
    print("\n" + "=" * 80)
    print("🧹 数据清洗完成")
    print("=" * 80)
    print(f"总文件数: {total_files}")
    print(f"清洗文件数: {cleaned_files}")
    print(f"剩余文件数: {total_files - cleaned_files}")
    
    if stats:
        print("\n📊 各角色清洗统计:")
        for stat in stats:
            print(f"  {stat['role']}: {stat['total']} -> {stat['remaining']} (删除 {stat['cleaned']})")
    
    # 保存清洗报告
    report_path = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/cleaning_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 数据清洗报告\n\n")
        f.write(f"## 清洗统计\n\n")
        f.write(f"- 总文件数: {total_files}\n")
        f.write(f"- 清洗文件数: {cleaned_files}\n")
        f.write(f"- 剩余文件数: {total_files - cleaned_files}\n\n")
        f.write("## 各角色清洗详情\n\n")
        for stat in stats:
            f.write(f"- {stat['role']}: {stat['total']} -> {stat['remaining']} (删除 {stat['cleaned']})\n")
    
    print(f"\n📄 清洗报告已保存到: {report_path}")

if __name__ == '__main__':
    clean_dataset()