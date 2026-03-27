#!/usr/bin/env python3
"""
测试PIL是否能直接打开SVG文件
"""

from PIL import Image
import os

# 测试图像路径
test_image_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train', '日奈', '日奈_1.svg')
print(f"使用测试图像: {test_image_path}")

# 确保文件存在
if not os.path.exists(test_image_path):
    print(f"测试图像不存在: {test_image_path}")
    exit(1)

# 尝试使用PIL打开SVG文件
try:
    print("尝试使用PIL打开SVG文件...")
    img = Image.open(test_image_path)
    print(f"成功打开SVG文件，格式: {img.format}, 大小: {img.size}")
    # 尝试将SVG转换为PNG
    png_path = test_image_path.replace('.svg', '.png')
    img.save(png_path, 'PNG')
    print(f"成功将SVG转换为PNG: {png_path}")
except Exception as e:
    print(f"PIL 加载SVG文件失败: {e}")
