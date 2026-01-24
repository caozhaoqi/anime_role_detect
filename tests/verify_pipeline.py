#!/usr/bin/env python3
"""
验证流水线脚本：下载数据 -> 构建索引 -> 验证分类
"""
import os
import sys
import shutil
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.collect_test_data import collect_single_character_data
from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification

def verify_pipeline():
    print("=== 开始验证项目功能 ===")
    
    # 1. 定义测试角色
    # 使用比较有特征的角色，例如 Re:Zero 的 Rem 和 Ram
    characters = ["Rem", "Ram"] 
    base_dir = "tests/temp_verification"
    
    # 清理旧数据
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    # 2. 下载数据
    print("\n[步骤 1] 下载测试数据...")
    data_dirs = {}
    for char in characters:
        output_dir = os.path.join(base_dir, "images", char)
        print(f"正在下载 {char} 的图片...")
        # 每个角色下载 5 张图片
        # 注意：collect_single_character_data 会尝试从 Pixiv/Danbooru/Konachan 下载，或者生成本地样本
        count = collect_single_character_data(char, 5, output_dir)
        
        if count == 0:
            print(f"警告: 无法下载 {char} 的图片")
            continue
            
        data_dirs[char] = output_dir
        print(f"成功为 {char} 准备了 {count} 张图片")

    if not data_dirs:
        print("错误: 没有准备好任何数据，终止验证")
        return

    # 3. 划分数据集 (索引库 vs 测试集)
    print("\n[步骤 2] 划分数据集...")
    index_images = {}
    test_images = {}
    
    for char, dir_path in data_dirs.items():
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if len(files) < 2:
            print(f"警告: {char} 的图片数量不足 ({len(files)} 张)，跳过")
            continue
        
        # 使用前 3 张作为底库，剩下的作为测试
        split_idx = min(3, len(files) - 1)
        index_images[char] = files[:split_idx]
        test_images[char] = files[split_idx:]
        
        print(f"角色 {char}: 底库 {len(index_images[char])} 张, 测试 {len(test_images[char])} 张")

    if not index_images:
        print("错误: 没有足够的图片用于构建索引，终止验证")
        return

    # 4. 构建索引
    print("\n[步骤 3] 构建向量索引...")
    try:
        preprocessor = Preprocessing()
        extractor = FeatureExtraction()
        classifier = Classification()
    except Exception as e:
        print(f"错误: 模块初始化失败 - {e}")
        return
    
    all_features = []
    all_roles = []
    
    for char, files in index_images.items():
        print(f"正在处理 {char} 的底库图片...")
        for file_path in files:
            try:
                # 预处理
                normalized_img, _ = preprocessor.process(file_path)
                # 特征提取
                feature = extractor.extract_features(normalized_img)
                
                all_features.append(feature)
                all_roles.append(char)
            except Exception as e:
                print(f"  处理图片 {os.path.basename(file_path)} 失败: {e}")

    if not all_features:
        print("错误: 未能提取任何特征，终止验证")
        return

    # 构建索引
    features_np = np.array(all_features).astype(np.float32)
    classifier.build_index(features_np, all_roles)
    print(f"索引构建完成，包含 {len(all_roles)} 个向量")

    # 5. 验证分类
    print("\n[步骤 4] 验证分类功能...")
    correct = 0
    total = 0
    
    for char, files in test_images.items():
        for file_path in files:
            total += 1
            try:
                # 预处理
                normalized_img, _ = preprocessor.process(file_path)
                # 特征提取
                feature = extractor.extract_features(normalized_img)
                # 分类
                predicted_role, similarity = classifier.classify(feature)
                
                print(f"测试图片: {os.path.basename(file_path)}")
                print(f"  真实角色: {char}")
                print(f"  预测角色: {predicted_role} (相似度: {similarity:.4f})")
                
                if predicted_role == char:
                    correct += 1
                    print("  结果: [正确]")
                else:
                    print("  结果: [错误]")
            except Exception as e:
                print(f"  分类图片 {os.path.basename(file_path)} 失败: {e}")
                
    if total > 0:
        print(f"\n=== 验证结果 ===")
        print(f"总测试数: {total}")
        print(f"正确数: {correct}")
        print(f"准确率: {correct/total*100:.2f}%")
        if correct / total > 0.5:
            print("结论: 系统功能验证通过 (准确率 > 50%)")
        else:
            print("结论: 系统功能验证未通过 (准确率 <= 50%)，请检查模型或数据质量")
    else:
        print("\n没有进行任何测试")

if __name__ == "__main__":
    verify_pipeline()
