#!/usr/bin/env python3
"""
基础验证脚本，测试系统是否能正常分类图片
"""
import os
import sys
import numpy as np
from PIL import Image

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification

def basic_process(image_path):
    """基础预处理"""
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"预处理失败: {e}")
        return None

def verify_system():
    """验证系统功能"""
    print("=== 验证系统分类功能 ===")
    
    test_dir = "tests/test_images/single_character"
    
    # 检查测试目录是否存在
    if not os.path.exists(test_dir):
        print(f"测试目录不存在: {test_dir}")
        return False
    
    # 初始化模块
    print("\n[步骤 1] 初始化系统模块...")
    try:
        extractor = FeatureExtraction()
        classifier = Classification()
        print("✓ 系统模块初始化成功")
    except Exception as e:
        print(f"✗ 系统模块初始化失败: {e}")
        return False
    
    # 获取测试角色
    role_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    if not role_dirs:
        print(f"测试目录中没有角色子目录: {test_dir}")
        return False
    
    print(f"找到 {len(role_dirs)} 个测试角色: {role_dirs}")
    
    # 构建索引
    print("\n[步骤 2] 构建特征索引...")
    all_features = []
    all_roles = []
    
    for role in role_dirs[:4]:  # 只使用前4个角色进行测试
        role_dir = os.path.join(test_dir, role)
        image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(image_files) < 1:
            print(f"角色 {role} 没有足够的图片")
            continue
        
        # 为每个角色选择1张图片构建索引
        img_file = image_files[0]
        img_path = os.path.join(role_dir, img_file)
        
        try:
            # 预处理
            normalized_img = basic_process(img_path)
            if normalized_img is None:
                print(f"无法处理角色 {role} 的图片")
                continue
            
            # 提取特征
            feature = extractor.extract_features(normalized_img)
            
            all_features.append(feature)
            all_roles.append(role)
            print(f"✓ 成功为角色 {role} 提取特征")
        except Exception as e:
            print(f"✗ 处理角色 {role} 失败: {e}")
            continue
    
    if len(all_features) < 2:
        print("✗ 无法构建索引：特征数量不足")
        return False
    
    # 构建索引
    try:
        features_np = np.array(all_features).astype(np.float32)
        classifier.build_index(features_np, all_roles)
        print(f"✓ 索引构建成功，包含 {len(all_roles)} 个角色")
    except Exception as e:
        print(f"✗ 索引构建失败: {e}")
        return False
    
    # 测试分类
    print("\n[步骤 3] 测试分类功能...")
    test_results = []
    
    for role in role_dirs[:4]:
        role_dir = os.path.join(test_dir, role)
        image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(image_files) < 2:
            print(f"角色 {role} 没有足够的测试图片")
            continue
        
        # 使用第二张图片进行测试
        img_file = image_files[1]
        img_path = os.path.join(role_dir, img_file)
        
        try:
            # 预处理
            normalized_img = basic_process(img_path)
            if normalized_img is None:
                print(f"无法处理测试图片: {img_file}")
                continue
            
            # 提取特征
            feature = extractor.extract_features(normalized_img)
            
            # 分类
            predicted_role, similarity = classifier.classify(feature)
            
            # 记录结果
            is_correct = predicted_role == role
            test_results.append({
                'role': role,
                'test_image': img_file,
                'predicted': predicted_role,
                'similarity': similarity,
                'correct': is_correct
            })
            
            status = "✓" if is_correct else "✗"
            print(f"{status} 角色 {role}: 预测为 {predicted_role} (相似度: {similarity:.4f})")
        except Exception as e:
            print(f"✗ 测试角色 {role} 失败: {e}")
            continue
    
    # 分析结果
    if test_results:
        correct_count = sum(1 for r in test_results if r['correct'])
        total_count = len(test_results)
        accuracy = (correct_count / total_count) * 100
        
        print(f"\n[步骤 4] 验证结果...")
        print(f"总测试数: {total_count}")
        print(f"正确数: {correct_count}")
        print(f"准确率: {accuracy:.2f}%")
        
        if accuracy > 0:
            print("✓ 系统分类功能验证成功！")
            return True
        else:
            print("✗ 系统分类功能验证失败")
            return False
    else:
        print("✗ 没有测试结果")
        return False

if __name__ == "__main__":
    success = verify_system()
    if success:
        print("\n🎉 系统验证通过！")
        sys.exit(0)
    else:
        print("\n❌ 系统验证失败！")
        sys.exit(1)
