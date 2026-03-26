#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试分类功能，绕过API服务
"""

import os
import sys
import time
from PIL import Image
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from core.feature_extraction.feature_extraction import FeatureExtraction
from core.classification.classification import Classification

def test_classification():
    """测试分类功能"""
    # 数据目录
    data_dir = 'data/train'
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    # 获取角色目录列表
    role_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"发现 {len(role_dirs)} 个角色目录")
    
    # 初始化特征提取器
    print("初始化特征提取器...")
    extractor = FeatureExtraction()
    print("特征提取器初始化完成")
    
    # 初始化分类器
    print("初始化分类器...")
    index_path = 'role_index.faiss'
    if not os.path.exists(index_path):
        print(f"索引文件不存在: {index_path}")
        return
    
    classifier = Classification()
    classifier.load_index(index_path)
    print("分类器初始化完成")
    
    # 测试结果
    total_tests = 0
    correct_predictions = 0
    
    # 遍历每个角色目录
    for role_name in role_dirs:
        role_dir = os.path.join(data_dir, role_name)
        # 获取目录下的图片文件
        image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        # 跳过没有图片的目录
        if len(image_files) == 0:
            print(f"跳过角色 '{role_name}'，没有图片")
            continue
        
        print(f"\n测试角色 '{role_name}'，发现 {len(image_files)} 张图片")
        
        # 测试每张图片（只测试前5张）
        for i, img_file in enumerate(image_files[:5]):
            img_path = os.path.join(role_dir, img_file)
            total_tests += 1
            
            try:
                # 读取图片
                img = Image.open(img_path).convert('RGB')
                
                # 提取特征
                feature = extractor.extract_features(img)
                
                # 分类
                predicted_role, similarity = classifier.classify(feature, top_k=1)
                
                # 检查分类是否正确
                if predicted_role == role_name:
                    correct_predictions += 1
                    print(f"✓ {img_file}: 正确分类为 '{predicted_role}'，相似度: {similarity:.4f}")
                else:
                    print(f"✗ {img_file}: 错误分类为 '{predicted_role}' (应为 '{role_name}')，相似度: {similarity:.4f}")
                    
            except Exception as e:
                print(f"✗ {img_file}: 测试失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 计算准确率
    if total_tests > 0:
        accuracy = (correct_predictions / total_tests) * 100
        print(f"\n测试完成: 共测试 {total_tests} 张图片，正确 {correct_predictions} 张，准确率: {accuracy:.2f}%")
    else:
        print("\n没有测试任何图片")

if __name__ == "__main__":
    test_classification()
