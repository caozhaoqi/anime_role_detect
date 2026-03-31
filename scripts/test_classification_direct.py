#!/usr/bin/env python3
"""
直接测试分类器
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification


def test_classification():
    """测试分类器"""
    print("=" * 60)
    print("开始直接测试分类器")
    print("=" * 60)
    
    # 初始化组件
    preprocessor = Preprocessing()
    extractor = FeatureExtraction()
    classifier = Classification(index_path="role_index", threshold=0.4)
    
    # 测试图像列表
    test_images = [
        ("data/train/日奈/日奈_000.jpg", "日奈"),
        ("data/train/伊织/伊织_000.jpg", "伊织"),
        ("data/train/亚子/亚子_000.jpg", "亚子"),
    ]
    
    for image_path, expected_role in test_images:
        full_path = project_root / image_path
        print(f"\n测试图像: {full_path}")
        print(f"预期角色: {expected_role}")
        
        try:
            # 预处理图像
            normalized_img, boxes = preprocessor.process(str(full_path))
            print(f"预处理完成，检测到 {len(boxes)} 个角色")
            
            # 提取特征
            feature = extractor.extract_features(normalized_img)
            print(f"特征提取完成，特征维度: {feature.shape}")
            
            # 分类
            role, similarity = classifier.classify(feature)
            print(f"分类结果: 角色={role}, 相似度={similarity:.4f}")
            
        except Exception as e:
            print(f"测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_classification()
