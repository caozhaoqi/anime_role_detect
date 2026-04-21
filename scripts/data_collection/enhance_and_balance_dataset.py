#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强和平衡脚本
为样本数量不足的角色生成更多训练数据
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
from torchvision import transforms

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("data_enhancer")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("data_enhancer")

# 配置参数
DATA_DIR = './data/downloaded_images'
OUTPUT_DIR = './data/enhanced_images'
MIN_SAMPLES = 100  # 每个角色的最小样本数
IMAGE_SIZE = 224

# 数据增强变换
augment_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
])

def get_image_count(directory):
    """获取目录中的图像数量"""
    count = 0
    for file in os.listdir(directory):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            count += 1
    return count

def augment_image(image_path, output_path, transform):
    """增强单张图像"""
    try:
        image = Image.open(image_path).convert('RGB')
        augmented = transform(image)
        augmented.save(output_path)
        return True
    except Exception as e:
        logger.error(f"增强图像失败 {image_path}: {e}")
        return False

def process_class(class_name, class_dir, output_class_dir):
    """处理单个类别"""
    # 确保输出目录存在
    os.makedirs(output_class_dir, exist_ok=True)
    
    # 获取现有图像
    images = []
    for file in os.listdir(class_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            images.append(os.path.join(class_dir, file))
    
    current_count = len(images)
    logger.info(f"处理类别 {class_name}: 当前 {current_count} 张图像")
    
    # 如果已经达到最小样本数，直接复制
    if current_count >= MIN_SAMPLES:
        logger.info(f"  样本数已足够，直接复制")
        for i, img_path in enumerate(images):
            output_path = os.path.join(output_class_dir, f"{i:04d}.jpg")
            try:
                img = Image.open(img_path).convert('RGB')
                img.save(output_path)
            except Exception as e:
                logger.error(f"  复制失败 {img_path}: {e}")
        return
    
    # 计算需要生成的额外样本数
    needed = MIN_SAMPLES - current_count
    logger.info(f"  需要生成 {needed} 张额外图像")
    
    # 复制原有图像
    for i, img_path in enumerate(images):
        output_path = os.path.join(output_class_dir, f"original_{i:04d}.jpg")
        try:
            img = Image.open(img_path).convert('RGB')
            img.save(output_path)
        except Exception as e:
            logger.error(f"  复制失败 {img_path}: {e}")
    
    # 生成增强样本
    generated = 0
    while generated < needed:
        for img_path in images:
            if generated >= needed:
                break
            
            output_path = os.path.join(output_class_dir, f"augmented_{generated:04d}.jpg")
            if augment_image(img_path, output_path, augment_transforms):
                generated += 1
                if generated % 10 == 0:
                    logger.info(f"  已生成 {generated}/{needed} 张增强图像")
    
    logger.info(f"  完成: 总计 {current_count + generated} 张图像")

def main():
    logger.info("=" * 60)
    logger.info("开始数据增强和平衡")
    logger.info("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有类别
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    logger.info(f"找到 {len(classes)} 个类别")
    
    # 处理每个类别
    for class_name in classes:
        class_dir = os.path.join(DATA_DIR, class_name)
        output_class_dir = os.path.join(OUTPUT_DIR, class_name)
        
        logger.info(f"\n处理: {class_name}")
        process_class(class_name, class_dir, output_class_dir)
    
    # 统计结果
    logger.info("\n" + "=" * 60)
    logger.info("数据增强和平衡完成")
    logger.info("=" * 60)
    
    total_images = 0
    for class_name in classes:
        output_class_dir = os.path.join(OUTPUT_DIR, class_name)
        count = get_image_count(output_class_dir)
        total_images += count
        logger.info(f"{class_name}: {count} 张图像")
    
    logger.info(f"\n总计: {total_images} 张图像")
    logger.info(f"平均每个类别: {total_images / len(classes):.1f} 张图像")

if __name__ == '__main__':
    main()
