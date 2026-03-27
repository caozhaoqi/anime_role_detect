#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试Core ML分类器
"""

import os
import sys
import time
import numpy as np
from PIL import Image

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 使用全局日志系统
from core.logging.global_logger import get_logger
logger = get_logger("simple_coreml_test")


def create_test_image():
    """创建测试图像"""
    # 创建一个简单的彩色图像
    img = Image.new('RGB', (224, 224), color='white')
    
    # 在图像上绘制一些简单的形状
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(img)
    
    # 绘制一个蓝色的矩形
    draw.rectangle([50, 50, 174, 174], fill='blue')
    
    # 绘制一个红色的圆形
    draw.ellipse([80, 80, 144, 144], fill='red')
    
    return img


def test_coreml_classification():
    """测试Core ML分类器"""
    try:
        # 导入分类模块
        from core.classification.classification import Classification
        
        # 创建Core ML分类器
        logger.info("创建Core ML分类器")
        classifier = Classification.use_coreml()
        
        # 创建测试图像
        img = create_test_image()
        logger.info("创建测试图像成功")
        
        # 测试分类性能
        start_time = time.time()
        role, similarity = classifier.classify_image(img)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        logger.info(f"分类结果: {role}")
        logger.info(f"相似度: {similarity:.4f}")
        logger.info(f"耗时: {elapsed_time:.4f}秒")
        
        logger.info("Core ML分类器测试完成")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


def main():
    """主函数"""
    try:
        logger.info("开始测试Core ML分类器...")
        test_coreml_classification()
        logger.info("测试完成!")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")


if __name__ == "__main__":
    main()
