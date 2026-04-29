#!/usr/bin/env python3
"""
创建测试图片的脚本
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image(output_path, size=(400, 400)):
    """创建一个测试图片"""
    # 创建白色背景图片
    image = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(image)
    
    # 添加测试文字
    text = "Test Image"
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    draw.text((x, y), text, fill='black', font=font)
    
    # 绘制一些图形
    draw.rectangle([50, 50, 150, 150], fill='red')
    draw.ellipse([250, 50, 350, 150], fill='blue')
    draw.rectangle([50, 250, 150, 350], fill='green')
    draw.ellipse([250, 250, 350, 350], fill='yellow')
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"测试图片已创建: {output_path}")

if __name__ == "__main__":
    create_test_image('test_images/sample.jpg')
