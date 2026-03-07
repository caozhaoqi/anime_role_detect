#!/usr/bin/env python3
"""
模型对比测试脚本
测试所有训练出的模型并生成对比报告
"""
import os
import sys
import time
import json
import logging
import argparse
import psutil
import numpy as np
from datetime import datetime
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('compare_models')

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

def get_memory_usage():
    """获取当前内存使用情况
    
    Returns:
        内存使用情况字典
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss': mem_info.rss / 1024 / 1024,  # MB
        'vms': mem_info.vms / 1024 / 1024,  # MB
        'used_percent': psutil.virtual_memory().percent
    }

def test_model_accuracy(classifier, test_data, use_model=False, model_name="Unknown"):
    """测试模型准确性
    
    Args:
        classifier: 分类器实例
        test_data: 测试数据列表 [(image_path, expected_role)]
        use_model: 是否使用EfficientNet模型
        model_name: 模型名称（用于日志）
    
    Returns:
        测试结果字典
    """
    logger.info(f"开始测试 {model_name} 模型...")
    
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

def test_model_performance(classifier, test_image_path, iterations=10):
    """测试模型性能
    
    Args:
        classifier: 分类器实例
        test_image_path: 测试图像路径
        iterations: 测试迭代次数
    
    Returns:
        性能测试结果字典
    """
    logger.info("开始测试模型性能...")
    
    # 预热
    logger.info("  预热中...")
    for i in range(2):
        classifier.classify_image(test_image_path, use_model=False)
        classifier.classify_image(test_image_path, use_model=True)
    
    # 测试CLIP模型
    logger.info(f"  测试CLIP模型 {iterations} 次...")
    clip_times = []
    for i in range(iterations):
        start_time = time.time()
        classifier.classify_image(test_image_path, use_model=False)
        end_time = time.time()
        clip_times.append(end_time - start_time)
    
    # 测试EfficientNet模型
    logger.info(f"  测试EfficientNet模型 {iterations} 次...")
    efficientnet_times = []
    for i in range(iterations):
        start_time = time.time()
        classifier.classify_image(test_image_path, use_model=True)
        end_time = time.time()
        efficientnet_times.append(end_time - start_time)
    
    # 计算统计信息
    clip_stats = {
        'average_time': np.mean(clip_times) * 1000,  # 转换为毫秒
        'std_time': np.std(clip_times) * 1000,
        'min_time': np.min(clip_times) * 1000,
        'max_time': np.max(clip_times) * 1000
    }
    
    efficientnet_stats = {
        'average_time': np.mean(efficientnet_times) * 1000,  # 转换为毫秒
        'std_time': np.std(efficientnet_times) * 1000,
        'min_time': np.min(efficientnet_times) * 1000,
        'max_time': np.max(efficientnet_times) * 1000
    }
    
    logger.info(f"CLIP模型性能:")
    logger.info(f"  平均时间: {clip_stats['average_time']:.2f}ms")
    logger.info(f"  标准差: {clip_stats['std_time']:.2f}ms")
    logger.info(f"  最小时间: {clip_stats['min_time']:.2f}ms")
    logger.info(f"  最大时间: {clip_stats['max_time']:.2f}ms")
    
    logger.info(f"EfficientNet模型性能:")
    logger.info(f"  平均时间: {efficientnet_stats['average_time']:.2f}ms")
    logger.info(f"  标准差: {efficientnet_stats['std_time']:.2f}ms")
    logger.info(f"  最小时间: {efficientnet_stats['min_time']:.2f}ms")
    logger.info(f"  最大时间: {efficientnet_stats['max_time']:.2f}ms")
    
    return {
        'clip': clip_stats,
        'efficientnet': efficientnet_stats
    }

def generate_model_comparison_report(results_list, performance_results, output_dir="model_comparison_reports"):
    """生成模型对比报告
    
    Args:
        results_list: 测试结果列表
        performance_results: 性能测试结果
        output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成报告文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"model_comparison_report_{timestamp}.json")
    report_md_file = os.path.join(output_dir, f"model_comparison_report_{timestamp}.md")
    
    # 保存JSON报告
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'accuracy_results': results_list,
            'performance_results': performance_results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"JSON报告已保存到: {report_file}")
    
    # 生成Markdown报告
    with open(report_md_file, 'w', encoding='utf-8') as f:
        f.write(f"# 模型对比测试报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 系统信息
        f.write("## 系统信息\n\n")
        f.write(f"- 操作系统: {sys.platform}\n")
        f.write(f"- Python版本: {sys.version}\n")
        try:
            cpu_info = psutil.cpu_info()
            if isinstance(cpu_info, list) and cpu_info:
                f.write(f"- 处理器: {cpu_info[0].model}\n")
            elif isinstance(cpu_info, dict):
                f.write(f"- 处理器: {cpu_info.get('model', 'Unknown')}\n")
            else:
                f.write(f"- 处理器: Unknown\n")
        except:
            f.write(f"- 处理器: Unknown\n")
        f.write(f"- CPU核心数: {psutil.cpu_count()}\n")
        f.write(f"- 总内存: {psutil.virtual_memory().total / 1024 / 1024:.2f} MB\n\n")
        
        # 准确率对比
        f.write("## 准确率对比\n\n")
        f.write("| 模型 | 测试总数 | 正确预测 | 错误预测 | 准确率 | 平均推理时间 (ms) | 误检数量 |\n")
        f.write("|------|----------|----------|----------|--------|------------------|----------|\n")
        
        for results in results_list:
            f.write(f"| {results['model_name']} | {results['total_tests']} | {results['correct_predictions']} | {results['incorrect_predictions']} | {results['accuracy']:.2f}% | {results['average_inference_time']:.2f} | {len(results['false_positives'])} |\n")
        
        f.write("\n")
        
        # 性能对比
        f.write("## 性能对比\n\n")
        f.write("### CLIP模型\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 平均时间 | {performance_results['clip']['average_time']:.2f}ms |\n")
        f.write(f"| 标准差 | {performance_results['clip']['std_time']:.2f}ms |\n")
        f.write(f"| 最小时间 | {performance_results['clip']['min_time']:.2f}ms |\n")
        f.write(f"| 最大时间 | {performance_results['clip']['max_time']:.2f}ms |\n\n")
        
        f.write("### EfficientNet模型\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 平均时间 | {performance_results['efficientnet']['average_time']:.2f}ms |\n")
        f.write(f"| 标准差 | {performance_results['efficientnet']['std_time']:.2f}ms |\n")
        f.write(f"| 最小时间 | {performance_results['efficientnet']['min_time']:.2f}ms |\n")
        f.write(f"| 最大时间 | {performance_results['efficientnet']['max_time']:.2f}ms |\n\n")
        
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
        
        # 结论和建议
        f.write("## 结论和建议\n\n")
        f.write("### 模型对比总结\n")
        
        # 找出准确率最高的模型
        best_accuracy_model = max(results_list, key=lambda x: x['accuracy'])
        f.write(f"- **准确率最高的模型**: {best_accuracy_model['model_name']} ({best_accuracy_model['accuracy']:.2f}%)\n")
        
        # 找出速度最快的模型
        best_speed_model = min(results_list, key=lambda x: x['average_inference_time'])
        f.write(f"- **速度最快的模型**: {best_speed_model['model_name']} ({best_speed_model['average_inference_time']:.2f}ms)\n")
        
        f.write("\n")
        f.write("### 优化建议\n")
        f.write("1. **模型选择**: 根据应用场景选择合适的模型\n")
        f.write("   - 对于需要高精度的场景，选择准确率高的模型\n")
        f.write("   - 对于需要实时性能的场景，选择速度快的模型\n")
        f.write("2. **集成学习**: 考虑结合多个模型的预测结果，提高整体性能\n")
        f.write("3. **数据增强**: 增加更多训练数据，特别是误检率高的角色\n")
        f.write("4. **模型调优**: 继续优化模型参数和结构\n")
        f.write("5. **部署优化**: 根据部署环境选择合适的模型和优化策略\n")
    
    logger.info(f"Markdown报告已保存到: {report_md_file}")
    return report_file, report_md_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型对比测试工具')
    
    # 输入参数
    parser.add_argument('--data_dir', type=str, default='data/train', help='测试数据目录')
    parser.add_argument('--max_images_per_class', type=int, default=10, help='每个类别的最大测试图像数量')
    parser.add_argument('--iterations', type=int, default=10, help='性能测试迭代次数')
    parser.add_argument('--output_dir', type=str, default='model_comparison_reports', help='测试报告输出目录')
    
    args = parser.parse_args()
    
    # 收集测试数据
    test_data = collect_test_data(args.data_dir, args.max_images_per_class)
    
    if not test_data:
        logger.error("没有找到测试数据")
        return
    
    # 初始化分类器
    logger.info("初始化分类器...")
    classifier = get_classifier()
    classifier.initialize()
    
    # 测试结果列表
    test_results = []
    
    # 1. 测试CLIP模型
    clip_results = test_model_accuracy(classifier, test_data, use_model=False, model_name="CLIP")
    test_results.append(clip_results)
    
    # 2. 测试EfficientNet模型
    try:
        efficientnet_results = test_model_accuracy(classifier, test_data, use_model=True, model_name="EfficientNet")
        test_results.append(efficientnet_results)
    except Exception as e:
        logger.error(f"EfficientNet模型测试失败: {e}")
    
    # 3. 测试模型性能
    if test_data:
        test_image_path = test_data[0][0]
        performance_results = test_model_performance(classifier, test_image_path, args.iterations)
    else:
        performance_results = {
            'clip': {'average_time': 0, 'std_time': 0, 'min_time': 0, 'max_time': 0},
            'efficientnet': {'average_time': 0, 'std_time': 0, 'min_time': 0, 'max_time': 0}
        }
    
    # 生成模型对比报告
    report_file, report_md_file = generate_model_comparison_report(test_results, performance_results, args.output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("模型对比测试完成！")
    logger.info(f"JSON报告: {report_file}")
    logger.info(f"Markdown报告: {report_md_file}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
