#!/usr/bin/env python3
"""
测试已有模型的准确率
使用 data/train 目录下的图像测试模型分类准确率
"""

import os
import sys
import json
from PIL import Image

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入必要的模块
from src.core.classification.classification import Classification
from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_model_accuracy')

def test_model_accuracy(model_path):
    """
    测试模型分类精度
    
    Args:
        model_path: 模型路径
    """
    # 初始化分类器
    if model_path:
        index_path = os.path.join(model_path, 'role_index')
    else:
        index_path = 'role_index'
    
    classifier = Classification(index_path=index_path)
    
    # 初始化预处理和特征提取模块
    preprocessor = Preprocessing()
    extractor = FeatureExtraction()
    
    # 收集测试数据
    test_data = []
    data_dir = 'data/train'
    
    role_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for role_dir in role_dirs:
        role_path = os.path.join(data_dir, role_dir)
        image_files = [f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        # 只测试前5张图片，加快测试速度
        for img_file in image_files[:5]:
            img_path = os.path.join(role_path, img_file)
            test_data.append((img_path, role_dir))
    
    logger.info(f"测试数据收集完成，共 {len(test_data)} 张图片")
    
    # 测试分类精度
    correct = 0
    total = 0
    results = []
    
    for img_path, true_role in test_data:
        try:
            # 预处理图像
            normalized_img, _ = preprocessor.process(img_path)
            
            # 提取特征
            feature = extractor.extract_features(normalized_img)
            
            # 分类特征
            predicted_role, similarity = classifier.classify(feature)
            
            # 检查预测结果是否正确
            is_correct = true_role in predicted_role or predicted_role in true_role
            
            if is_correct:
                correct += 1
            total += 1
            
            # 记录结果
            results.append({
                'image_path': img_path,
                'true_role': true_role,
                'predicted_role': predicted_role,
                'similarity': similarity,
                'is_correct': is_correct
            })
            
            logger.info(f"图片: {os.path.basename(img_path)}, 真实角色: {true_role}, 预测角色: {predicted_role}, 相似度: {similarity:.4f}, {'正确' if is_correct else '错误'}")
            
        except Exception as e:
            logger.error(f"处理图片 {img_path} 时出错: {e}")
            continue
    
    # 计算准确率
    if total > 0:
        accuracy = correct / total
        logger.info(f"测试完成，准确率: {accuracy * 100:.2f}% ({correct}/{total})")
    else:
        logger.error("没有测试数据")
        return
    
    # 保存测试结果
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'model_accuracy_{os.path.basename(model_path) if model_path else "default"}.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_path': model_path,
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"测试结果已保存到: {output_path}")
    return accuracy
if __name__ == '__main__':
    # 只测试默认模型
    model_path = ''
    model_name = 'default'
    
    logger.info(f"\n测试模型: {model_name}")
    try:
        accuracy = test_model_accuracy(model_path)
        if accuracy is not None:
            logger.info(f"\n=== 模型准确率测试总结 ===")
            logger.info(f"模型: {model_name}, 准确率: {accuracy * 100:.2f}%")
    except Exception as e:
        logger.error(f"测试模型 {model_name} 时出错: {e}")
