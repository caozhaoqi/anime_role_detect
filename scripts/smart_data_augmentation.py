#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能数据增强系统
为现有数据生成更多训练样本
"""

import os
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from pathlib import Path

# 配置参数
DATA_DIR = "./data/downloaded_images"
TARGET_IMAGES_PER_ROLE = 150  # 目标每个角色150张图片
AUGMENTATION_MULTIPLIER = 3  # 数据增强倍数

def rotate_image(image, angle):
    """旋转图片"""
    return image.rotate(angle, expand=False, fillcolor=(255, 255, 255))

def flip_image(image, direction):
    """翻转图片"""
    if direction == 'horizontal':
        return ImageOps.mirror(image)
    elif direction == 'vertical':
        return ImageOps.flip(image)
    return image

def adjust_brightness(image, factor):
    """调整亮度"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    """调整对比度"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_color(image, factor):
    """调整颜色"""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def add_noise(image, intensity=0.05):
    """添加噪声"""
    img_array = np.array(image)
    noise = np.random.normal(0, intensity * 255, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def blur_image(image, radius=0.5):
    """轻微模糊"""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def sharpen_image(image, factor=1.5):
    """锐化"""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def crop_image(image, crop_ratio=0.9):
    """随机裁剪"""
    width, height = image.size
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)
    
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    
    cropped = image.crop((left, top, left + new_width, top + new_height))
    return cropped.resize((width, height), Image.LANCZOS)

def augment_image(image, augment_count=1):
    """对图像进行数据增强"""
    augmented_images = []
    
    for _ in range(augment_count):
        aug_image = image.copy()
        
        # 随机旋转 (-25到25度)
        if random.random() > 0.4:
            angle = random.uniform(-25, 25)
            aug_image = rotate_image(aug_image, angle)
        
        # 随机翻转
        if random.random() > 0.5:
            direction = random.choice(['horizontal', 'vertical'])
            aug_image = flip_image(aug_image, direction)
        
        # 随机调整亮度 (0.7到1.3)
        if random.random() > 0.4:
            factor = random.uniform(0.7, 1.3)
            aug_image = adjust_brightness(aug_image, factor)
        
        # 随机调整对比度 (0.7到1.3)
        if random.random() > 0.4:
            factor = random.uniform(0.7, 1.3)
            aug_image = adjust_contrast(aug_image, factor)
        
        # 随机调整颜色 (0.7到1.3)
        if random.random() > 0.4:
            factor = random.uniform(0.7, 1.3)
            aug_image = adjust_color(aug_image, factor)
        
        # 随机添加轻微噪声
        if random.random() > 0.7:
            aug_image = add_noise(aug_image, intensity=0.03)
        
        # 随机轻微模糊
        if random.random() > 0.8:
            aug_image = blur_image(aug_image, radius=0.3)
        
        # 随机锐化
        if random.random() > 0.7:
            aug_image = sharpen_image(aug_image, factor=1.3)
        
        # 随机裁剪
        if random.random() > 0.6:
            aug_image = crop_image(aug_image, crop_ratio=random.uniform(0.88, 0.95))
        
        augmented_images.append(aug_image)
    
    return augmented_images

def process_role_directory(role_dir, role_name, target_count):
    """处理单个角色目录"""
    print(f"\n处理角色: {role_name}")
    
    # 获取现有图片
    images = [f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    current_count = len(images)
    
    print(f"  现有图片: {current_count} 张")
    
    if current_count == 0:
        print(f"  ✗ 没有原始图片，无法进行数据增强")
        return 0
    
    if current_count >= target_count:
        print(f"  ✓ 已达到目标数量")
        return 0
    
    needed = target_count - current_count
    print(f"  需要增加: {needed} 张")
    
    # 加载现有图片
    existing_images = []
    for img_file in images:
        try:
            img_path = os.path.join(role_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            existing_images.append(img)
        except Exception as e:
            print(f"  ✗ 无法加载图片: {img_file} - {str(e)}")
    
    if not existing_images:
        print(f"  ✗ 没有可用的原始图片")
        return 0
    
    # 计算需要增强的图片数量
    augment_per_image = max(1, needed // len(existing_images))
    remaining = needed % len(existing_images)
    
    print(f"  每张图片增强: {augment_per_image} 次")
    
    # 进行数据增强
    augmented_count = 0
    for i, img in enumerate(existing_images):
        # 计算当前图片需要增强的数量
        current_augment = augment_per_image
        if i < remaining:
            current_augment += 1
        
        # 进行数据增强
        augmented_images = augment_image(img, current_augment)
        
        # 保存增强后的图片
        for aug_img in augmented_images:
            # 生成新文件名
            base_name = os.path.splitext(images[i])[0]
            aug_filename = f"{base_name}_aug{augmented_count + 1:03d}.jpg"
            aug_path = os.path.join(role_dir, aug_filename)
            
            # 保存图片
            try:
                aug_img.save(aug_path, 'JPEG', quality=85)
                augmented_count += 1
                
                if augmented_count % 10 == 0:
                    print(f"  进度: {augmented_count}/{needed}")
            except Exception as e:
                print(f"  ✗ 保存失败: {aug_filename} - {str(e)}")
    
    print(f"  ✓ 成功增强: {augmented_count} 张")
    return augmented_count

def main():
    """主函数"""
    print("=" * 60)
    print("智能数据增强系统")
    print("=" * 60)
    
    # 统计现有数据
    print("\n现有数据统计:")
    role_stats = {}
    total_images = 0
    total_roles = 0
    
    for role_dir in os.listdir(DATA_DIR):
        role_path = os.path.join(DATA_DIR, role_dir)
        if not os.path.isdir(role_path):
            continue
        
        images = [f for f in os.listdir(role_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        count = len(images)
        
        if count > 0:
            role_stats[role_dir] = count
            total_images += count
            total_roles += 1
            print(f"  {role_dir}: {count} 张")
    
    print(f"\n总角色数: {total_roles}")
    print(f"总图片数: {total_images}")
    
    # 数据增强
    print(f"\n开始数据增强，目标: 每个角色 {TARGET_IMAGES_PER_ROLE} 张")
    
    total_augmented = 0
    processed_roles = 0
    
    for role_dir, count in role_stats.items():
        if count < TARGET_IMAGES_PER_ROLE:
            role_path = os.path.join(DATA_DIR, role_dir)
            augmented = process_role_directory(role_path, role_dir, TARGET_IMAGES_PER_ROLE)
            total_augmented += augmented
            processed_roles += 1
    
    # 最终统计
    print("\n" + "=" * 60)
    print("数据增强完成")
    print("=" * 60)
    print(f"处理角色数: {processed_roles}")
    print(f"增强图片数: {total_augmented} 张")
    
    print("\n最终数据统计:")
    final_total = 0
    final_roles = 0
    
    for role_dir, count in role_stats.items():
        role_path = os.path.join(DATA_DIR, role_dir)
        final_count = len([f for f in os.listdir(role_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        final_total += final_count
        if final_count > 0:
            final_roles += 1
    
    print(f"总角色数: {final_roles}")
    print(f"总图片数: {final_total} 张")
    print(f"增加图片数: {final_total - total_images} 张")
    print("=" * 60)

if __name__ == "__main__":
    main()
