#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测并筛选黑白（灰度）图片
"""

import os
import sys
import shutil
import logging
from PIL import Image

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("filter_bw_images")

def is_black_and_white(image_path):
    """
    判断图片是否为黑白图片
    :param image_path: 图片路径
    :return: (是否为黑白, 原因)
    """
    try:
        with Image.open(image_path) as img:
            # 规则1: 直接检查图片模式
            if img.mode == 'L':
                return True, "图片模式为灰度模式"
            
            # 转换为RGB模式
            img_rgb = img.convert('RGB')
            
            # 规则2: 使用直方图判断
            # 获取RGB直方图 - 返回长度为768的列表（256*3）
            hist = img_rgb.histogram()
            r_hist = hist[:256]
            g_hist = hist[256:512]
            b_hist = hist[512:]
            
            # 计算每个通道的标准差
            def calculate_std(hist_data):
                total = sum(hist_data)
                if total == 0:
                    return 0
                mean = sum(i * hist_data[i] for i in range(256)) / total
                variance = sum(hist_data[i] * (i - mean) ** 2 for i in range(256)) / total
                return variance ** 0.5
            
            r_std = calculate_std(r_hist)
            g_std = calculate_std(g_hist)
            b_std = calculate_std(b_hist)
            
            # 计算三个通道标准差的平均值
            avg_std = (r_std + g_std + b_std) / 3
            
            # 黑白图片的标准差通常较小
            if avg_std < 25:
                return True, f"颜色标准差低 ({avg_std:.1f})"
            
            # 规则3: 检查RGB通道的相似度
            total_diff = 0
            for i in range(256):
                total_diff += abs(r_hist[i] - g_hist[i]) + abs(g_hist[i] - b_hist[i])
            
            # 获取图片尺寸计算总像素
            width, height = img_rgb.size
            total_pixels = width * height
            
            if total_pixels > 0:
                diff_ratio = total_diff / (2 * total_pixels)
                # 如果通道之间差异很小，说明是灰度图
                if diff_ratio < 0.02:
                    return True, f"RGB通道差异小 ({diff_ratio:.2%})"
            
    except Exception as e:
        logger.warning(f"分析图片失败 {image_path}: {e}")
        return False, f"分析失败: {e}"
    
    return False, "彩色图片"

def filter_bw_images(dataset_path, output_path, dry_run=False):
    """
    筛选数据集中的黑白图片
    """
    logger.info("=" * 60)
    logger.info(f"开始检测黑白图片")
    logger.info(f"数据集路径: {dataset_path}")
    logger.info(f"模式: {'预览' if dry_run else '实际移动'}")
    logger.info("=" * 60)
    
    os.makedirs(output_path, exist_ok=True)
    
    role_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    total_scanned = 0
    total_bw = 0
    
    for role_name in role_dirs:
        role_dir = os.path.join(dataset_path, role_name)
        img_files = sorted([f for f in os.listdir(role_dir) if f.lower().endswith('.jpg')])
        
        if not img_files:
            continue
        
        bw_count = 0
        for img_file in img_files:
            img_path = os.path.join(role_dir, img_file)
            is_bw, reason = is_black_and_white(img_path)
            
            total_scanned += 1
            
            if is_bw:
                bw_count += 1
                total_bw += 1
                
                if not dry_run:
                    output_role_dir = os.path.join(output_path, role_name)
                    os.makedirs(output_role_dir, exist_ok=True)
                    output_path_full = os.path.join(output_role_dir, img_file)
                    shutil.move(img_path, output_path_full)
        
        if bw_count > 0:
            logger.info(f"{role_name}: 扫描 {len(img_files)} 张, 检测到 {bw_count} 张黑白图片")
    
    logger.info("=" * 60)
    logger.info(f"检测完成！")
    logger.info(f"总扫描图片: {total_scanned} 张")
    logger.info(f"检测到黑白图片: {total_bw} 张")
    logger.info("=" * 60)

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("用法:")
            print("  python filter_bw_images.py          # 执行筛选")
            print("  python filter_bw_images.py --dry-run # 预览模式")
            print("  python filter_bw_images.py --help   # 显示帮助")
            return
        elif sys.argv[1] == '--dry-run':
            filter_bw_images(
                dataset_path='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset',
                output_path='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/bw_images',
                dry_run=True
            )
            return
    
    # 默认执行筛选
    filter_bw_images(
        dataset_path='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset',
        output_path='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/bw_images',
        dry_run=False
    )

if __name__ == "__main__":
    main()
