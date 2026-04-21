#!/usr/bin/env python3
"""
测试NSFW模型加载
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.services.nsfw_detector_pytorch import load_model, detect_nsfw_with_pytorch
from io import BytesIO
from PIL import Image

# 创建一个简单的测试图像
def create_test_image():
    """创建一个测试图像"""
    img = Image.new('RGB', (224, 224), color='white')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_nsfw_model():
    """测试NSFW模型"""
    print("测试NSFW模型加载...")
    
    # 加载模型
    model = load_model()
    if model is not None:
        print("✅ NSFW模型加载成功")
    else:
        print("❌ NSFW模型加载失败")
        return
    
    # 测试检测功能
    test_image = create_test_image()
    result = detect_nsfw_with_pytorch(test_image)
    
    if result is not None:
        print("✅ NSFW检测功能正常")
        print(f"检测结果: {result}")
    else:
        print("❌ NSFW检测功能失败")

if __name__ == "__main__":
    test_nsfw_model()
