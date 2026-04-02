#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建测试图片
"""

from PIL import Image, ImageDraw, ImageFont
import os

# 创建一个简单的测试图片
def create_test_image():
    # 创建一个白色背景的图片
    img = Image.new('RGB', (600, 400), color='white')
    d = ImageDraw.Draw(img)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype('Arial', 24)
    except:
        font = ImageFont.load_default()
    
    # 绘制一些文本
    d.text((50, 50), "Anime Role Test", fill=(0, 0, 0), font=font)
    d.text((50, 100), "Single character detection", fill=(0, 0, 0), font=font)
    
    # 保存图片
    test_image_path = "test_image.jpg"
    img.save(test_image_path)
    print(f"测试图片已创建: {test_image_path}")
    return test_image_path

if __name__ == "__main__":
    create_test_image()
