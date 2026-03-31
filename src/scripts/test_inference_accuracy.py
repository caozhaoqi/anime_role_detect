#!/usr/bin/env python3
"""
测试推理准确性脚本
使用训练图片测试增强后的特征库的推理准确性
"""
import os
import sys
import json

# 添加项目根目录到Python路径
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(current_file))
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)
print(f"Python路径: {sys.path}")

from core.preprocessing.preprocessing import Preprocessing
from core.feature_extraction.feature_extraction import FeatureExtraction
from core.classification.classification import Classification
from core.tagging.wd_vit_v3_tagger import WDViTV3Tagger

def test_inference_accuracy(train_dir="data/train", index_path="role_index_augmented"):
    """
    测试推理准确性
    
    Args:
        train_dir: 训练数据目录
        index_path: 特征库索引路径
    """
    print("开始测试推理准确性...")
    print(f"训练数据目录: {train_dir}")
    print(f"特征库索引路径: {index_path}")
    
    # 初始化模型
    preprocessor = Preprocessing()
    extractor = FeatureExtraction(quantize=False)
    classifier = Classification(index_path)
    tagger = WDViTV3Tagger()
    tagger.load_model()
    
    # 收集测试数据
    test_data = []
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    print(f"找到 {len(classes)} 个类别")
    
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        image_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(image_files) > 0:
            print(f"类别 '{cls}' 包含 {len(image_files)} 张图片")
            # 每个类别取前5张图片进行测试
            for img_file in image_files[:5]:
                img_path = os.path.join(cls_dir, img_file)
                test_data.append((img_path, cls))
    
    print(f"总测试样本数: {len(test_data)}")
    
    # 开始测试
    correct_count = 0
    total_count = len(test_data)
    
    for i, (img_path, expected_role) in enumerate(test_data):
        print(f"\n测试样本 {i+1}/{total_count}")
        print(f"图片路径: {img_path}")
        print(f"期望角色: {expected_role}")
        
        try:
            # 预处理图像
            normalized_img, _ = preprocessor.process(img_path)
            
            # 提取特征
            feature = extractor.extract_features(normalized_img)
            
            # 生成标签
            tags = tagger.generate_tags(img_path)
            
            # 分类
            result = classifier.classify(feature, tags=tags)
            predicted_role, similarity = result
            
            print(f"预测角色: {predicted_role}")
            print(f"相似度: {similarity:.4f}")
            
            # 判断是否正确
            if predicted_role == expected_role:
                correct_count += 1
                print("✓ 预测正确")
            else:
                print("✗ 预测错误")
                
        except Exception as e:
            print(f"✗ 处理失败: {e}")
    
    # 计算准确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\n测试完成!")
    print(f"总测试样本数: {total_count}")
    print(f"正确预测数: {correct_count}")
    print(f"准确率: {accuracy:.4f} ({correct_count}/{total_count})")
    
    # 保存测试结果
    test_result = {
        "total_count": total_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "test_time": "2026-03-31"
    }
    
    with open("test_inference_accuracy_result.json", "w", encoding="utf-8") as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)
    
    print(f"测试结果已保存到 test_inference_accuracy_result.json")

if __name__ == "__main__":
    test_inference_accuracy()
