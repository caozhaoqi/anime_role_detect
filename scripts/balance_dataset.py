#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强和平衡脚本
通过数据增强技术扩充数据集并平衡各角色的数据量
"""

import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
from pathlib import Path

# 配置参数
DATA_DIR = "./data/downloaded_images"
TARGET_COUNT = 100  # 每个角色的目标图片数量

# 角色配置
ROLES = {
    "a1luo2na4": {
        "dir": "a1luo2na4",
        "chinese_name": "阿罗娜"
    },
    "ri4nai4": {
        "dir": "ri4nai4", 
        "chinese_name": "日奈"
    },
    "plana": {
        "dir": "plana",
        "chinese_name": "普拉娜"
    }
}

def rotate_image(image, angle):
    """旋转图像"""
    return image.rotate(angle, expand=True, fillcolor=(255, 255, 255))

def flip_image(image, direction='horizontal'):
    """翻转图像"""
    if direction == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == 'vertical':
        return image.transpose(Image.FLIP_TOP_BOTTOM)
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
    """调整颜色饱和度"""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def add_noise(image, intensity=0.1):
    """添加噪声"""
    img_array = np.array(image)
    noise = np.random.normal(0, intensity * 255, img_array.shape).astype(np.uint8)
    noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def blur_image(image, radius=1):
    """模糊图像"""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def sharpen_image(image, factor=2.0):
    """锐化图像"""
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def crop_image(image, crop_ratio=0.8):
    """随机裁剪图像"""
    width, height = image.size
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)
    
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    right = left + new_width
    bottom = top + new_height
    
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((width, height), Image.LANCZOS)

def augment_image(image, augment_count=1):
    """对图像进行数据增强"""
    augmented_images = []
    
    for _ in range(augment_count):
        # 随机选择增强方法
        aug_image = image.copy()
        
        # 随机旋转 (-30到30度)
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            aug_image = rotate_image(aug_image, angle)
        
        # 随机翻转
        if random.random() > 0.5:
            direction = random.choice(['horizontal', 'vertical'])
            aug_image = flip_image(aug_image, direction)
        
        # 随机调整亮度 (0.7到1.3)
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            aug_image = adjust_brightness(aug_image, factor)
        
        # 随机调整对比度 (0.7到1.3)
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            aug_image = adjust_contrast(aug_image, factor)
        
        # 随机调整颜色 (0.7到1.3)
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)
            aug_image = adjust_color(aug_image, factor)
        
        # 随机添加轻微噪声
        if random.random() > 0.7:
            aug_image = add_noise(aug_image, intensity=0.05)
        
        # 随机轻微模糊
        if random.random() > 0.8:
            aug_image = blur_image(aug_image, radius=0.5)
        
        # 随机锐化
        if random.random() > 0.7:
            aug_image = sharpen_image(aug_image, factor=1.5)
        
        # 随机裁剪
        if random.random() > 0.6:
            aug_image = crop_image(aug_image, crop_ratio=random.uniform(0.85, 0.95))
        
        augmented_images.append(aug_image)
    
    return augmented_images

def balance_dataset():
    """平衡数据集"""
    print("=" * 60)
    print("开始数据增强和平衡")
    print("=" * 60)
    
    # 统计现有数据
    print("\n现有数据统计:")
    role_counts = {}
    for role_name, role_config in ROLES.items():
        role_dir = os.path.join(DATA_DIR, role_config["dir"])
        images = [f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        role_counts[role_name] = len(images)
        print(f"  {role_config['chinese_name']}: {len(images)} 张")
    
    print(f"\n目标数量: 每个角色 {TARGET_COUNT} 张")
    
    # 对每个角色进行数据增强
    for role_name, role_config in ROLES.items():
        role_dir = os.path.join(DATA_DIR, role_config["dir"])
        current_count = role_counts[role_name]
        chinese_name = role_config["chinese_name"]
        
        if current_count >= TARGET_COUNT:
            print(f"\n{chinese_name} 已达到目标数量，跳过")
            continue
        
        if current_count == 0:
            print(f"\n{chinese_name} 没有原始图片，无法进行数据增强")
            print(f"  请先为 {chinese_name} 收集一些原始图片")
            continue
        
        needed = TARGET_COUNT - current_count
        print(f"\n{chinese_name} 需要增加 {needed} 张图片")
        
        # 获取现有图片
        images = [f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # 计算每张图片需要增强的次数
        augment_per_image = max(1, needed // len(images))
        remaining = needed % len(images)
        
        print(f"  每张图片增强 {augment_per_image} 次，前 {remaining} 张额外增强 1 次")
        
        # 对每张图片进行增强
        augmented_count = 0
        for i, img_name in enumerate(images):
            img_path = os.path.join(role_dir, img_name)
            
            try:
                # 加载图片
                image = Image.open(img_path).convert('RGB')
                
                # 计算增强次数
                aug_count = augment_per_image
                if i < remaining:
                    aug_count += 1
                
                # 进行数据增强
                augmented_images = augment_image(image, aug_count)
                
                # 保存增强后的图片
                for aug_img in augmented_images:
                    if augmented_count >= needed:
                        break
                    
                    # 生成新文件名
                    new_name = f"aug_{augmented_count:04d}_{img_name}"
                    new_path = os.path.join(role_dir, new_name)
                    
                    # 保存图片
                    aug_img.save(new_path, 'JPEG', quality=95)
                    augmented_count += 1
                
                print(f"  处理 {img_name}: 生成 {len(augmented_images)} 张增强图片")
                
            except Exception as e:
                print(f"  ✗ 处理 {img_name} 失败: {str(e)}")
            
            if augmented_count >= needed:
                break
        
        print(f"  {chinese_name} 增强完成，共生成 {augmented_count} 张图片")
    
    # 最终统计
    print("\n" + "=" * 60)
    print("数据增强完成")
    print("=" * 60)
    print("\n最终数据统计:")
    total_images = 0
    for role_name, role_config in ROLES.items():
        role_dir = os.path.join(DATA_DIR, role_config["dir"])
        final_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {role_config['chinese_name']}: {final_count} 张")
        total_images += final_count
    
    print(f"\n总图片数量: {total_images} 张")
    print("=" * 60)

def main():
    """主函数"""
    balance_dataset()

if __name__ == "__main__":
    main()
