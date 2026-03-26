#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试分类功能，不依赖索引文件
"""

import os
import sys
import numpy as np
from PIL import Image

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from core.feature_extraction.feature_extraction import FeatureExtraction

def test_feature_extraction():
    """测试特征提取功能"""
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
    
    # 测试结果
    total_tests = 0
    successful_extractions = 0
    
    # 遍历每个角色目录
    for role_name in role_dirs:
        role_dir = os.path.join(data_dir, role_name)
        # 获取目录下的图片文件
        image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.svg'))]
        
        # 跳过没有图片的目录
        if len(image_files) == 0:
            print(f"跳过角色 '{role_name}'，没有图片")
            continue
        
        print(f"\n测试角色 '{role_name}'，发现 {len(image_files)} 张图片")
        
        # 测试每张图片（只测试前2张）
        for i, img_file in enumerate(image_files[:2]):
            img_path = os.path.join(role_dir, img_file)
            total_tests += 1
            
            try:
                # 读取图片
                img = Image.open(img_path).convert('RGB')
                
                # 提取特征
                feature = extractor.extract(img)
                
                # 检查特征向量
                if feature is not None and len(feature) > 0:
                    successful_extractions += 1
                    print(f"✓ {img_file}: 特征提取成功，特征维度: {feature.shape}")
                else:
                    print(f"✗ {img_file}: 特征提取失败，返回空特征")
                    
            except Exception as e:
                print(f"✗ {img_file}: 测试失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 计算成功率
    if total_tests > 0:
        success_rate = (successful_extractions / total_tests) * 100
        print(f"\n测试完成: 共测试 {total_tests} 张图片，成功提取 {successful_extractions} 张，成功率: {success_rate:.2f}%")
    else:
        print("\n没有测试任何图片")

if __name__ == "__main__":
    test_feature_extraction()
