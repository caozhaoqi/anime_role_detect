#!/usr/bin/env python3
"""
测试所有模型的分类效果
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.backend.services.classification_service import classify_image


def test_all_models():
    """测试所有模型的分类效果"""
    print("=== 测试所有模型 ===")
    
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
    print()
    
    # 测试配置列表
    test_configs = [
        # (模型名称, use_coreml, use_model, use_deepdanbooru, use_attributes, 描述)
        ("default", False, False, False, False, "默认模型 (CLIP + FAISS)"),
        ("default", False, True, False, False, "专用模型 (EfficientNet)"),
        ("default", True, False, False, False, "Core ML 模型"),
        ("default", False, False, True, False, "DeepDanbooru 集成模型"),
        ("default", False, False, False, True, "带属性预测的模型"),
    ]
    
    for model_name, use_coreml, use_model, use_deepdanbooru, use_attributes, description in test_configs:
        print(f"\n=== 测试: {description} ===")
        start_time = time.time()
        
        try:
            role, similarity, boxes, mode, attributes, text_detections = classify_image(
                test_image,
                use_coreml=use_coreml,
                use_model=use_model,
                use_deepdanbooru=use_deepdanbooru,
                use_attributes=use_attributes,
                model_name=model_name
            )
            
            elapsed_time = time.time() - start_time
            print(f"分类结果: 角色={role}, 相似度={similarity:.4f}")
            print(f"使用模式: {mode}")
            print(f"边界框数量: {len(boxes)}")
            print(f"属性数量: {len(attributes)}")
            print(f"文本检测数量: {len(text_detections)}")
            print(f"处理时间: {elapsed_time:.2f}秒")
            
            if attributes:
                print("前3个属性:")
                for attr in attributes[:3]:
                    print(f"  - {attr['tag']}: {attr['confidence']:.4f}")
            
        except Exception as e:
            print(f"分类失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 50)


if __name__ == "__main__":
    test_all_models()
