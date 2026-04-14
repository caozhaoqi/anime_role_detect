#!/usr/bin/env python3
"""
测试分类功能
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.classification.general_classification import get_classifier
from src.backend.config import DEFAULT_INDEX_PATH


def test_classification():
    """测试分类功能"""
    print("=== 测试分类功能 ===")
    
    # 初始化分类器
    classifier = get_classifier(index_path=DEFAULT_INDEX_PATH)
    classifier.initialize()
    
    # 测试图像路径（使用项目中的示例图像）
    test_image = "test_images/sample1.jpg"
    
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        # 尝试使用其他测试图像
        test_image = "test_images/sample2.jpg"
        if not os.path.exists(test_image):
            print("没有找到测试图像，使用默认图像路径")
            test_image = "test.jpg"
    
    print(f"测试图像: {test_image}")
    
    # 测试分类
    try:
        role, similarity, boxes, attributes, text_detections = classifier.classify_image(test_image, use_model=False, use_attributes=True)
        print(f"分类结果: 角色={role}, 相似度={similarity:.4f}")
        print(f"边界框数量: {len(boxes)}")
        print(f"属性数量: {len(attributes)}")
        print(f"文本检测数量: {len(text_detections)}")
        
        if attributes:
            print("前3个属性:")
            for attr in attributes[:3]:
                print(f"  - {attr['tag']}: {attr['confidence']:.4f}")
        
        if text_detections:
            print("前3个文本检测:")
            for text in text_detections[:3]:
                print(f"  - {text['text']}")
                
    except Exception as e:
        print(f"分类失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_classification()
