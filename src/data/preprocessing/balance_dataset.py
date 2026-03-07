#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强脚本 - 平衡数据集
为数据量较少的角色生成增强图像
"""

import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('balance_dataset')


def augment_image(image, augment_type):
    """对图像进行增强
    
    Args:
        image: PIL Image对象
        augment_type: 增强类型
    
    Returns:
        增强后的图像
    """
    if augment_type == 'rotate':
        angle = random.choice([-15, -10, -5, 5, 10, 15])
        return image.rotate(angle, expand=False, fillcolor=(255, 255, 255))
    
    elif augment_type == 'flip':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    
    elif augment_type == 'brightness':
        enhancer = ImageEnhance.Brightness(image)
        factor = random.choice([0.8, 0.9, 1.1, 1.2])
        return enhancer.enhance(factor)
    
    elif augment_type == 'contrast':
        enhancer = ImageEnhance.Contrast(image)
        factor = random.choice([0.8, 0.9, 1.1, 1.2])
        return enhancer.enhance(factor)
    
    elif augment_type == 'color':
        enhancer = ImageEnhance.Color(image)
        factor = random.choice([0.8, 0.9, 1.1, 1.2])
        return enhancer.enhance(factor)
    
    elif augment_type == 'sharpness':
        enhancer = ImageEnhance.Sharpness(image)
        factor = random.choice([0.8, 1.2])
        return enhancer.enhance(factor)
    
    elif augment_type == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=random.choice([0.5, 1.0])))
    
    elif augment_type == 'crop':
        width, height = image.size
        crop_size = int(min(width, height) * random.uniform(0.85, 0.95))
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((width, height), Image.LANCZOS)
    
    else:
        return image


def balance_dataset(data_dir, target_count=100, output_dir=None):
    """平衡数据集
    
    Args:
        data_dir: 原始数据目录
        target_count: 每个角色的目标图像数量
        output_dir: 输出目录，如果为None则覆盖原目录
    """
    if output_dir is None:
        output_dir = data_dir
    
    augment_types = [
        'rotate', 'flip', 'brightness', 'contrast', 
        'color', 'sharpness', 'blur', 'crop'
    ]
    
    total_augmented = 0
    
    for character in os.listdir(data_dir):
        character_dir = os.path.join(data_dir, character)
        if not os.path.isdir(character_dir):
            continue
        
        images = []
        for img_name in os.listdir(character_dir):
            img_path = os.path.join(character_dir, img_name)
            if os.path.isfile(img_path) and img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                images.append(img_path)
        
        current_count = len(images)
        
        if current_count >= target_count:
            logger.info(f"{character}: {current_count} 张图像（已达到目标）")
            continue
        
        needed = target_count - current_count
        logger.info(f"{character}: {current_count} 张图像 -> 需要生成 {needed} 张增强图像")
        
        output_char_dir = os.path.join(output_dir, character)
        os.makedirs(output_char_dir, exist_ok=True)
        
        augmented_count = 0
        attempts = 0
        max_attempts = needed * 3
        
        while augmented_count < needed and attempts < max_attempts:
            img_path = random.choice(images)
            img = Image.open(img_path).convert('RGB')
            
            augment_type = random.choice(augment_types)
            augmented_img = augment_image(img, augment_type)
            
            output_name = f"aug_{augmented_count:04d}_{augment_type}.jpg"
            output_path = os.path.join(output_char_dir, output_name)
            augmented_img.save(output_path, 'JPEG', quality=95)
            
            augmented_count += 1
            attempts += 1
            
            if augmented_count % 10 == 0:
                logger.info(f"  已生成 {augmented_count}/{needed} 张增强图像")
        
        total_augmented += augmented_count
        logger.info(f"{character}: 完成，共生成 {augmented_count} 张增强图像\n")
    
    logger.info(f"数据集平衡完成，共生成 {total_augmented} 张增强图像")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='平衡数据集')
    parser.add_argument('--data-dir', type=str, default='../../data/train', help='数据目录')
    parser.add_argument('--target-count', type=int, default=100, help='每个角色的目标图像数量')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    balance_dataset(args.data_dir, args.target_count, args.output_dir)


if __name__ == '__main__':
    main()
