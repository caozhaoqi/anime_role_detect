#!/usr/bin/env python3
"""
测试API分类精度
使用API接口测试模型的分类精度
"""

import os
import sys
import json
import requests
from PIL import Image
import io

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_api_accuracy')


def test_api_accuracy(data_dir):
    """
    测试API分类精度
    
    Args:
        data_dir: 数据目录，包含角色子目录
    """
    # API接口地址
    api_url = "http://localhost:5001/api/classify"
    
    # 收集测试数据
    test_data = []
    role_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for role_dir in role_dirs:
        role_path = os.path.join(data_dir, role_dir)
        image_files = [f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        for img_file in image_files:
            img_path = os.path.join(role_path, img_file)
            test_data.append((img_path, role_dir))
    
    logger.info(f"测试数据收集完成，共 {len(test_data)} 张图片")
    
    # 测试分类精度
    correct = 0
    total = 0
    results = []
    
    for img_path, true_role in test_data:
        try:
            # 准备文件
            with open(img_path, 'rb') as f:
                img_data = f.read()
            
            files = {'file': (os.path.basename(img_path), img_data, 'image/jpeg')}
            
            # 发送请求
            response = requests.post(api_url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                predicted_role = result.get('role', '未知')
                similarity = result.get('similarity', 0.0)
                
                # 检查预测结果是否正确
                # 这里简单地比较角色名称是否匹配，实际应用中可能需要更复杂的匹配逻辑
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
            else:
                logger.error(f"API请求失败: {response.status_code}, {response.text}")
                continue
            
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
    output_dir = '../test_results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'api_accuracy_results.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"测试结果已保存到: {output_path}")


if __name__ == '__main__':
    # 测试数据目录
    data_dir = 'data/sdv50_train'
    
    # 测试API分类精度
    logger.info("测试API分类精度...")
    test_api_accuracy(data_dir)
