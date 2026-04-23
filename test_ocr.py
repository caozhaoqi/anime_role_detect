#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR 功能测试脚本
"""

import os
import sys
from src.core.ocr.easyocr_detector import detect_text

# 测试图像路径
# 注意：需要替换为实际存在的测试图像路径
test_image_path = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/test_images/test_text.jpg"

def test_ocr():
    """测试 OCR 功能"""
    print("=== OCR 功能测试 ===")
    print(f"测试图像: {test_image_path}")
    
    if not os.path.exists(test_image_path):
        print(f"错误: 测试图像不存在 - {test_image_path}")
        print("请替换为实际存在的测试图像路径")
        return
    
    try:
        print("开始 OCR 文字检测...")
        results = detect_text(test_image_path)
        
        if results:
            print(f"成功检测到 {len(results)} 个文本区域:")
            for i, result in enumerate(results, 1):
                print(f"\n文本 {i}:")
                print(f"  内容: {result['text']}")
                print(f"  置信度: {result['confidence']:.4f}")
                print(f"  边界框: {result['bbox']}")
        else:
            print("未检测到文本")
            
    except Exception as e:
        print(f"OCR 测试失败: {e}")

if __name__ == "__main__":
    test_ocr()
