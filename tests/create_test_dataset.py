#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建简单的测试数据集，用于验证系统功能
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_test_image(role_name, index, size=(224, 224)):
    """创建一个简单的测试图片"""
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 根据角色名称生成不同的颜色
    color_hash = hash(role_name) % 0xFFFFFF
    color = (color_hash & 0xFF, (color_hash >> 8) & 0xFF, (color_hash >> 16) & 0xFF)
    
    # 绘制背景
    draw.rectangle([0, 0, size[0], size[1]], fill=color)
    
    # 绘制一些随机图形
    for _ in range(5):
        x1 = np.random.randint(0, size[0])
        y1 = np.random.randint(0, size[1])
        x2 = np.random.randint(0, size[0])
        y2 = np.random.randint(0, size[1])
        shape_color = tuple(np.random.randint(0, 255, 3).tolist())
        # 确保x1 <= x2和y1 <= y2
        draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=shape_color, outline=(0, 0, 0))
    
    # 绘制角色名称
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    text = f"{role_name}_{index}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    text_x = (size[0] - text_width) // 2
    text_y = (size[1] - text_height) // 2
    
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    return img

def create_test_dataset(data_dir, roles, images_per_role=10):
    """创建测试数据集"""
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    for role_name in roles:
        role_dir = os.path.join(data_dir, role_name)
        os.makedirs(role_dir, exist_ok=True)
        
        print(f"创建角色 '{role_name}' 的测试图片...")
        
        for i in range(images_per_role):
            img = create_test_image(role_name, i)
            img_path = os.path.join(role_dir, f"{role_name}_{i:03d}.jpg")
            img.save(img_path, 'JPEG', quality=95)
            print(f"  创建: {img_path}")
    
    print(f"\n测试数据集创建完成！")
    print(f"角色数量: {len(roles)}")
    print(f"每个角色图片数: {images_per_role}")
    print(f"总图片数: {len(roles) * images_per_role}")

def main():
    """主函数"""
    # 定义测试角色
    roles = ['日奈', '普拉娜', '伊织', '亚子', '阿罗娜']
    
    # 创建测试数据集
    create_test_dataset('data/train', roles, images_per_role=10)

if __name__ == "__main__":
    main()
