#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用CLIP模型识别漫画风格图片（简化版）
"""

import os
import sys
import shutil
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("comic_detector")

def is_comic_style_simple(image_path):
    """
    使用简单规则判断图片是否为漫画风格
    :param image_path: 图片路径
    :return: (是否为漫画风格, 原因)
    """
    try:
        from PIL import Image
        
        with Image.open(image_path) as img:
            width, height = img.size
            ratio = width / height
            
            # 规则1: 宽高比异常（漫画分镜通常很宽）
            if ratio > 3.0 or ratio < 0.3:
                return True, f"宽高比异常 {width}x{height} ({ratio:.2f})"
            
            # 规则2: 图片尺寸过小
            if width < 300 or height < 300:
                return True, f"图片尺寸过小 {width}x{height}"
            
            # 规则3: 检查是否为灰度图（漫画扫描件特征）
            if img.mode == 'L' or img.mode == 'P':
                return True, "灰度图片（漫画扫描件特征）"
            
            # 规则4: 检测漫画特有的黑白对比
            # 计算高对比度区域比例
            gray = img.convert('L')
            hist = gray.histogram()
            black_pixels = sum(hist[:20])  # 接近黑色
            white_pixels = sum(hist[235:])  # 接近白色
            total_pixels = sum(hist)
            
            if total_pixels > 0:
                contrast_ratio = (black_pixels + white_pixels) / total_pixels
                if contrast_ratio > 0.4:
                    # 高对比度是漫画风格的特征之一
                    # 但需要结合其他特征
                    pass
            
            # 规则5: 检测网格线（漫画分镜特征）
            if detect_grid_lines(img):
                return True, "检测到网格线（漫画分镜特征）"
            
            # 规则6: 检测文字气泡（漫画特征）
            if detect_text_bubbles(img):
                return True, "检测到文字气泡（漫画特征）"
            
    except Exception as e:
        logger.warning(f"分析图片失败 {image_path}: {e}")
        return False, f"分析失败: {e}"
    
    return False, "通过检测"

def detect_grid_lines(img, threshold=150):
    """检测漫画分镜网格线"""
    gray = img.convert('L')
    width, height = gray.size
    
    # 检查垂直线
    vertical_lines = 0
    for x in range(0, width, 50):
        dark_count = sum(1 for y in range(height) if gray.getpixel((x, y)) < threshold)
        if dark_count > height * 0.8:
            vertical_lines += 1
    
    # 检查水平线
    horizontal_lines = 0
    for y in range(0, height, 50):
        dark_count = sum(1 for x in range(width) if gray.getpixel((x, y)) < threshold)
        if dark_count > width * 0.8:
            horizontal_lines += 1
    
    # 如果有多个网格线，很可能是漫画
    return vertical_lines >= 2 or horizontal_lines >= 2

def detect_text_bubbles(img):
    """检测漫画中的文字气泡"""
    # 气泡通常是白色或浅色的圆形区域
    hsv = img.convert('HSV')
    width, height = hsv.size
    
    # 统计高亮度、低饱和度区域
    light_area = 0
    total_pixels = width * height
    
    # 采样检测
    sample_points = 0
    light_points = 0
    
    for y in range(0, height, 20):
        for x in range(0, width, 20):
            sample_points += 1
            h, s, v = hsv.getpixel((x, y))
            if v > 230 and s < 20:
                light_points += 1
    
    # 如果浅色区域占比过高，可能包含气泡
    if sample_points > 0 and light_points / sample_points > 0.1:
        return True
    return False

def detect_comic_images(dataset_path, output_path, dry_run=False):
    """
    检测数据集中的漫画风格图片
    """
    logger.info("=" * 60)
    logger.info(f"开始检测漫画风格图片")
    logger.info(f"数据集路径: {dataset_path}")
    logger.info(f"模式: {'预览' if dry_run else '实际移动'}")
    logger.info("=" * 60)
    
    os.makedirs(output_path, exist_ok=True)
    
    role_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    total_scanned = 0
    total_comic = 0
    
    for role_name in role_dirs:
        role_dir = os.path.join(dataset_path, role_name)
        img_files = sorted([f for f in os.listdir(role_dir) if f.lower().endswith('.jpg')])
        
        if not img_files:
            continue
        
        comic_count = 0
        for img_file in img_files:
            img_path = os.path.join(role_dir, img_file)
            is_comic, reason = is_comic_style_simple(img_path)
            
            total_scanned += 1
            
            if is_comic:
                comic_count += 1
                total_comic += 1
                
                if not dry_run:
                    output_role_dir = os.path.join(output_path, role_name)
                    os.makedirs(output_role_dir, exist_ok=True)
                    output_path_full = os.path.join(output_role_dir, img_file)
                    shutil.move(img_path, output_path_full)
        
        if comic_count > 0:
            logger.info(f"{role_name}: 扫描 {len(img_files)} 张, 检测到 {comic_count} 张漫画风格图片")
    
    logger.info("=" * 60)
    logger.info(f"检测完成！")
    logger.info(f"总扫描图片: {total_scanned} 张")
    logger.info(f"检测到漫画风格: {total_comic} 张")
    logger.info("=" * 60)

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("用法:")
            print("  python comic_detector.py          # 执行检测")
            print("  python comic_detector.py --dry-run # 预览模式")
            print("  python comic_detector.py --help   # 显示帮助")
            return
        elif sys.argv[1] == '--dry-run':
            detect_comic_images(
                dataset_path='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset',
                output_path='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/comic_images',
                dry_run=True
            )
            return
    
    # 默认执行检测
    detect_comic_images(
        dataset_path='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset',
        output_path='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/comic_images',
        dry_run=False
    )

if __name__ == "__main__":
    main()
