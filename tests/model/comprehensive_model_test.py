#!/usr/bin/env python3
"""
全面模型测试脚本
测试所有推理模式的准确性和误检率
生成详细的测试报告
"""
import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from PIL import Image
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('comprehensive_model_test')

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入通用分类模块
from src.core.general_classification import GeneralClassification, get_classifier

def collect_test_data(data_dir, max_images_per_class=10):
    """收集测试数据
    
    Args:
        data_dir: 数据目录
        max_images_per_class: 每个类别的最大图像数量
    
    Returns:
        测试数据列表，每个元素为 (image_path, expected_role)
    """
    test_data = []
    
    logger.info(f"从目录收集测试数据: {data_dir}")
    
    # 遍历所有角色目录
    for role_dir in os.listdir(data_dir):
        role_path = os.path.join(data_dir, role_dir)
        
        if not os.path.isdir(role_path):
            continue
        
        # 提取角色名称（使用完整的目录名）
        # 因为索引文件中使用的是完整的目录名作为角色名称
        role_name = role_dir
        
        # 收集图像
        image_files = []
        for file in os.listdir(role_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_files.append(os.path.join(role_path, file))
        
        # 限制图像数量
        image_files = image_files[:max_images_per_class]
        
        logger.info(f"类别 '{role_name}' 收集了 {len(image_files)} 张图像")
        
        # 添加到测试数据
        for image_path in image_files:
            test_data.append((image_path, role_name))
    
    logger.info(f"总共收集了 {len(test_data)} 张测试图像")
    return test_data

def test_model_accuracy(test_data, use_model=False, model_name="Unknown"):
    """测试模型准确性
    
    Args:
        test_data: 测试数据列表 [(image_path, expected_role)]
        use_model: 是否使用EfficientNet模型
        model_name: 模型名称（用于日志）
    
    Returns:
        测试结果字典
    """
    logger.info(f"开始测试 {model_name} 模型...")
    
    # 初始化分类器
    classifier = get_classifier()
    classifier.initialize()
    
    # 测试结果
    results = {
        'model_name': model_name,
        'use_model': use_model,
        'total_tests': len(test_data),
        'correct_predictions': 0,
        'incorrect_predictions': 0,
        'accuracy': 0.0,
        'average_inference_time': 0.0,
        'confusion_matrix': {},
        'false_positives': [],
        'false_negatives': [],
        'detailed_results': []
    }
    
    inference_times = []
    
    # 测试每张图像
    for i, (image_path, expected_role) in enumerate(test_data):
        logger.info(f"测试 {i+1}/{len(test_data)}: {os.path.basename(image_path)}")
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 分类图像
            predicted_role, similarity, boxes = classifier.classify_image(image_path, use_model=use_model)
            
            # 记录推理时间
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 验证预测结果
            is_correct = predicted_role == expected_role
            
            # 更新结果
            if is_correct:
                results['correct_predictions'] += 1
            else:
                results['incorrect_predictions'] += 1
                
                # 记录误检
                if predicted_role:
                    results['false_positives'].append({
                        'image_path': image_path,
                        'expected_role': expected_role,
                        'predicted_role': predicted_role,
                        'similarity': similarity
                    })
                else:
                    results['false_negatives'].append({
                        'image_path': image_path,
                        'expected_role': expected_role,
                        'predicted_role': predicted_role,
                        'similarity': similarity
                    })
            
            # 更新混淆矩阵
            if expected_role not in results['confusion_matrix']:
                results['confusion_matrix'][expected_role] = {}
            
            if predicted_role not in results['confusion_matrix'][expected_role]:
                results['confusion_matrix'][expected_role][predicted_role] = 0
            
            results['confusion_matrix'][expected_role][predicted_role] += 1
            
            # 记录详细结果
            results['detailed_results'].append({
                'image_path': image_path,
                'expected_role': expected_role,
                'predicted_role': predicted_role,
                'similarity': similarity,
                'is_correct': is_correct,
                'inference_time': inference_time,
                'boxes': boxes
            })
            
            logger.info(f"  结果: 期望='{expected_role}', 预测='{predicted_role}', 相似度={similarity:.4f}, 正确={is_correct}")
            
        except Exception as e:
            logger.error(f"  测试失败: {e}")
            results['incorrect_predictions'] += 1
    
    # 计算准确率
    if results['total_tests'] > 0:
        results['accuracy'] = results['correct_predictions'] / results['total_tests'] * 100
    
    # 计算平均推理时间
    if inference_times:
        results['average_inference_time'] = sum(inference_times) / len(inference_times) * 1000  # 转换为毫秒
    
    logger.info(f"{model_name} 模型测试完成:")
    logger.info(f"  准确率: {results['accuracy']:.2f}%")
    logger.info(f"  正确预测: {results['correct_predictions']}")
    logger.info(f"  错误预测: {results['incorrect_predictions']}")
    logger.info(f"  平均推理时间: {results['average_inference_time']:.2f}ms")
    logger.info(f"  误检数量: {len(results['false_positives'])}")
    
    return results

def generate_test_report(results_list, output_dir="test_reports"):
    """生成测试报告
    
    Args:
        results_list: 测试结果列表
        output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成报告文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"model_test_report_{timestamp}.json")
    report_md_file = os.path.join(output_dir, f"model_test_report_{timestamp}.md")
    
    # 保存JSON报告
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    
    logger.info(f"JSON报告已保存到: {report_file}")
    
    # 生成Markdown报告
    with open(report_md_file, 'w', encoding='utf-8') as f:
        f.write(f"# 模型测试报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 整体统计
        f.write("## 整体统计\n\n")
        
        for results in results_list:
            f.write(f"### {results['model_name']} 模型\n")
            f.write(f"- 使用模型: {'是' if results['use_model'] else '否'}\n")
            f.write(f"- 测试总数: {results['total_tests']}\n")
            f.write(f"- 正确预测: {results['correct_predictions']}\n")
            f.write(f"- 错误预测: {results['incorrect_predictions']}\n")
            f.write(f"- 准确率: {results['accuracy']:.2f}%\n")
            f.write(f"- 平均推理时间: {results['average_inference_time']:.2f}ms\n")
            f.write(f"- 误检数量: {len(results['false_positives'])}\n\n")
        
        # 误检分析
        f.write("## 误检分析\n\n")
        
        for results in results_list:
            f.write(f"### {results['model_name']} 模型误检\n")
            
            if results['false_positives']:
                f.write("#### 错误正例 (False Positives)\n")
                f.write("| 图像文件 | 期望角色 | 预测角色 | 相似度 |\n")
                f.write("|---------|---------|---------|--------|\n")
                
                for fp in results['false_positives'][:20]:  # 只显示前20个
                    f.write(f"| {os.path.basename(fp['image_path'])} | {fp['expected_role']} | {fp['predicted_role']} | {fp['similarity']:.4f} |\n")
                
                if len(results['false_positives']) > 20:
                    f.write(f"| ... | ... | ... | ... |\n")
                    f.write(f"| 总计 | - | - | {len(results['false_positives'])} 个 |\n")
            else:
                f.write("#### 错误正例 (False Positives)\n")
                f.write("无\n")
            
            f.write("\n")
        
        # 改进建议
        f.write("## 改进建议\n\n")
        f.write("1. **数据增强**: 增加更多训练数据，特别是误检率高的角色\n")
        f.write("2. **模型调优**: 调整模型参数，如学习率、批量大小等\n")
        f.write("3. **阈值调整**: 根据测试结果调整分类阈值\n")
        f.write("4. **特征工程**: 考虑使用更强大的特征提取方法\n")
        f.write("5. **模型集成**: 结合多种模型的预测结果\n")
        f.write("6. **错误分析**: 深入分析误检案例，了解模型的弱点\n")
    
    logger.info(f"Markdown报告已保存到: {report_md_file}")
    return report_file, report_md_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='全面模型测试工具')
    
    # 输入参数
    parser.add_argument('--data_dir', type=str, default='data/train', help='测试数据目录')
    parser.add_argument('--max_images_per_class', type=int, default=10, help='每个类别的最大测试图像数量')
    parser.add_argument('--test_efficientnet', action='store_true', help='是否测试EfficientNet模型')
    parser.add_argument('--output_dir', type=str, default='test_reports', help='测试报告输出目录')
    
    args = parser.parse_args()
    
    # 收集测试数据
    test_data = collect_test_data(args.data_dir, args.max_images_per_class)
    
    if not test_data:
        logger.error("没有找到测试数据")
        return
    
    # 测试结果列表
    test_results = []
    
    # 1. 测试CLIP模型（默认模式）
    clip_results = test_model_accuracy(test_data, use_model=False, model_name="CLIP")
    test_results.append(clip_results)
    
    # 2. 测试EfficientNet模型（如果指定）
    if args.test_efficientnet:
        efficientnet_results = test_model_accuracy(test_data, use_model=True, model_name="EfficientNet")
        test_results.append(efficientnet_results)
    
    # 生成测试报告
    report_file, report_md_file = generate_test_report(test_results, args.output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("测试完成！")
    logger.info(f"JSON报告: {report_file}")
    logger.info(f"Markdown报告: {report_md_file}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
