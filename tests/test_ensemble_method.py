#!/usr/bin/env python3
"""
集成分类方法测试脚本
测试并比较集成方法与单个模型的性能
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
logger = logging.getLogger('test_ensemble_method')

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
logger.info(f"当前脚本路径: {current_script_path}")

# 计算项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(current_script_path), '..'))
logger.info(f"项目根目录: {project_root}")

# 添加项目根目录到Python路径
sys.path.insert(0, project_root)
logger.info(f"添加到Python路径: {project_root}")

# 打印Python路径，检查是否正确
logger.info(f"Python路径: {sys.path}")

# 导入通用分类模块
try:
    from src.core.classification.general_classification import GeneralClassification, get_classifier
    logger.info("成功导入通用分类模块")
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    # 尝试直接导入
    try:
        # 检查文件是否存在
        general_classification_path = os.path.join(project_root, 'src', 'core', 'classification', 'general_classification.py')
        logger.info(f"检查文件是否存在: {general_classification_path}")
        if os.path.exists(general_classification_path):
            logger.info("文件存在，尝试直接导入")
            # 动态导入
            import importlib.util
            spec = importlib.util.spec_from_file_location("general_classification", general_classification_path)
            general_classification = importlib.util.module_from_spec(spec)
            sys.modules["src.core.classification.general_classification"] = general_classification
            spec.loader.exec_module(general_classification)
            GeneralClassification = general_classification.GeneralClassification
            get_classifier = general_classification.get_classifier
            logger.info("成功动态导入通用分类模块")
        else:
            logger.error(f"文件不存在: {general_classification_path}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"动态导入失败: {e}")
        sys.exit(1)

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

def test_model_accuracy(classifier, test_data, method="clip", clip_weight=0.7, model_weight=0.3, confidence_threshold=0.6):
    """测试模型准确性
    
    Args:
        classifier: 分类器实例
        test_data: 测试数据列表 [(image_path, expected_role)]
        method: 分类方法 (clip, model, ensemble)
        clip_weight: CLIP模型的权重（仅用于ensemble方法）
        model_weight: MobileNetV2模型的权重（仅用于ensemble方法）
        confidence_threshold: 置信度阈值（仅用于ensemble方法）
    
    Returns:
        测试结果字典
    """
    method_names = {
        "clip": "CLIP",
        "model": "MobileNetV2",
        "ensemble": "Ensemble"
    }
    model_name = method_names.get(method, "Unknown")
    
    logger.info(f"开始测试 {model_name} 方法...")
    
    # 测试结果
    results = {
        'method': method,
        'model_name': model_name,
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
            
            # 根据方法选择分类函数
            if method == "clip":
                predicted_role, similarity, boxes = classifier.classify_image(image_path, use_model=False)
            elif method == "model":
                predicted_role, similarity, boxes = classifier.classify_image(image_path, use_model=True)
            elif method == "ensemble":
                predicted_role, similarity, boxes = classifier.classify_image_ensemble(
                    image_path, clip_weight, model_weight, confidence_threshold
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
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
    
    logger.info(f"{model_name} 方法测试完成:")
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
        classifier.classify_image_ensemble(test_image_path)
    
    # 测试CLIP模型
    logger.info(f"  测试CLIP模型 {iterations} 次...")
    clip_times = []
    for i in range(iterations):
        start_time = time.time()
        classifier.classify_image(test_image_path, use_model=False)
        end_time = time.time()
        clip_times.append(end_time - start_time)
    
    # 测试MobileNetV2模型
    logger.info(f"  测试MobileNetV2模型 {iterations} 次...")
    model_times = []
    for i in range(iterations):
        start_time = time.time()
        classifier.classify_image(test_image_path, use_model=True)
        end_time = time.time()
        model_times.append(end_time - start_time)
    
    # 测试集成方法
    logger.info(f"  测试集成方法 {iterations} 次...")
    ensemble_times = []
    for i in range(iterations):
        start_time = time.time()
        classifier.classify_image_ensemble(test_image_path)
        end_time = time.time()
        ensemble_times.append(end_time - start_time)
    
    # 计算统计信息
    clip_stats = {
        'average_time': np.mean(clip_times) * 1000,  # 转换为毫秒
        'std_time': np.std(clip_times) * 1000,
        'min_time': np.min(clip_times) * 1000,
        'max_time': np.max(clip_times) * 1000
    }
    
    model_stats = {
        'average_time': np.mean(model_times) * 1000,  # 转换为毫秒
        'std_time': np.std(model_times) * 1000,
        'min_time': np.min(model_times) * 1000,
        'max_time': np.max(model_times) * 1000
    }
    
    ensemble_stats = {
        'average_time': np.mean(ensemble_times) * 1000,  # 转换为毫秒
        'std_time': np.std(ensemble_times) * 1000,
        'min_time': np.min(ensemble_times) * 1000,
        'max_time': np.max(ensemble_times) * 1000
    }
    
    logger.info(f"CLIP模型性能:")
    logger.info(f"  平均时间: {clip_stats['average_time']:.2f}ms")
    logger.info(f"  标准差: {clip_stats['std_time']:.2f}ms")
    logger.info(f"  最小时间: {clip_stats['min_time']:.2f}ms")
    logger.info(f"  最大时间: {clip_stats['max_time']:.2f}ms")
    
    logger.info(f"MobileNetV2模型性能:")
    logger.info(f"  平均时间: {model_stats['average_time']:.2f}ms")
    logger.info(f"  标准差: {model_stats['std_time']:.2f}ms")
    logger.info(f"  最小时间: {model_stats['min_time']:.2f}ms")
    logger.info(f"  最大时间: {model_stats['max_time']:.2f}ms")
    
    logger.info(f"集成方法性能:")
    logger.info(f"  平均时间: {ensemble_stats['average_time']:.2f}ms")
    logger.info(f"  标准差: {ensemble_stats['std_time']:.2f}ms")
    logger.info(f"  最小时间: {ensemble_stats['min_time']:.2f}ms")
    logger.info(f"  最大时间: {ensemble_stats['max_time']:.2f}ms")
    
    return {
        'clip': clip_stats,
        'model': model_stats,
        'ensemble': ensemble_stats
    }

def generate_ensemble_comparison_report(results_list, performance_results, output_dir="ensemble_test_reports"):
    """生成集成方法对比报告
    
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
    report_file = os.path.join(output_dir, f"ensemble_comparison_report_{timestamp}.json")
    report_md_file = os.path.join(output_dir, f"ensemble_comparison_report_{timestamp}.md")
    
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
        f.write(f"# 集成分类方法测试报告\n")
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
        f.write("| 方法 | 测试总数 | 正确预测 | 错误预测 | 准确率 | 平均推理时间 (ms) | 误检数量 |\n")
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
        
        f.write("### MobileNetV2模型\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 平均时间 | {performance_results['model']['average_time']:.2f}ms |\n")
        f.write(f"| 标准差 | {performance_results['model']['std_time']:.2f}ms |\n")
        f.write(f"| 最小时间 | {performance_results['model']['min_time']:.2f}ms |\n")
        f.write(f"| 最大时间 | {performance_results['model']['max_time']:.2f}ms |\n\n")
        
        f.write("### 集成方法\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 平均时间 | {performance_results['ensemble']['average_time']:.2f}ms |\n")
        f.write(f"| 标准差 | {performance_results['ensemble']['std_time']:.2f}ms |\n")
        f.write(f"| 最小时间 | {performance_results['ensemble']['min_time']:.2f}ms |\n")
        f.write(f"| 最大时间 | {performance_results['ensemble']['max_time']:.2f}ms |\n\n")
        
        # 误检分析
        f.write("## 误检分析\n\n")
        
        for results in results_list:
            f.write(f"### {results['model_name']} 方法误检\n")
            
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
        f.write("### 方法对比总结\n")
        
        # 找出准确率最高的方法
        best_accuracy_method = max(results_list, key=lambda x: x['accuracy'])
        f.write(f"- **准确率最高的方法**: {best_accuracy_method['model_name']} ({best_accuracy_method['accuracy']:.2f}%)\n")
        
        # 找出速度最快的方法
        best_speed_method = min(results_list, key=lambda x: x['average_inference_time'])
        f.write(f"- **速度最快的方法**: {best_speed_method['model_name']} ({best_speed_method['average_inference_time']:.2f}ms)\n")
        
        f.write("\n")
        f.write("### 集成方法优势\n")
        f.write("1. **提高准确率**: 结合多个模型的优势，减少单一模型的误检\n")
        f.write("2. **增强鲁棒性**: 对不同类型的图像有更好的适应性\n")
        f.write("3. **置信度融合**: 通过加权融合，获得更可靠的预测结果\n")
        f.write("\n")
        f.write("### 应用建议\n")
        f.write("1. **高精度场景**: 使用集成方法，提高分类准确率\n")
        f.write("2. **实时场景**: 使用CLIP模型，保证推理速度\n")
        f.write("3. **资源受限场景**: 根据硬件条件选择合适的方法\n")
        f.write("4. **权重调整**: 根据具体应用场景调整模型权重，优化性能\n")
    
    logger.info(f"Markdown报告已保存到: {report_md_file}")
    return report_file, report_md_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='集成分类方法测试工具')
    
    # 输入参数
    parser.add_argument('--data_dir', type=str, default='data/train', help='测试数据目录')
    parser.add_argument('--max_images_per_class', type=int, default=5, help='每个类别的最大测试图像数量')
    parser.add_argument('--iterations', type=int, default=10, help='性能测试迭代次数')
    parser.add_argument('--output_dir', type=str, default='ensemble_test_reports', help='测试报告输出目录')
    parser.add_argument('--clip_weight', type=float, default=0.7, help='CLIP模型的权重')
    parser.add_argument('--model_weight', type=float, default=0.3, help='MobileNetV2模型的权重')
    parser.add_argument('--confidence_threshold', type=float, default=0.6, help='置信度阈值')
    
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
    clip_results = test_model_accuracy(classifier, test_data, method="clip")
    test_results.append(clip_results)
    
    # 2. 测试MobileNetV2模型
    try:
        model_results = test_model_accuracy(classifier, test_data, method="model")
        test_results.append(model_results)
    except Exception as e:
        logger.error(f"MobileNetV2模型测试失败: {e}")
    
    # 3. 测试集成方法
    try:
        ensemble_results = test_model_accuracy(
            classifier, 
            test_data, 
            method="ensemble",
            clip_weight=args.clip_weight,
            model_weight=args.model_weight,
            confidence_threshold=args.confidence_threshold
        )
        test_results.append(ensemble_results)
    except Exception as e:
        logger.error(f"集成方法测试失败: {e}")
    
    # 4. 测试模型性能
    if test_data:
        test_image_path = test_data[0][0]
        performance_results = test_model_performance(classifier, test_image_path, args.iterations)
    else:
        performance_results = {
            'clip': {'average_time': 0, 'std_time': 0, 'min_time': 0, 'max_time': 0},
            'model': {'average_time': 0, 'std_time': 0, 'min_time': 0, 'max_time': 0},
            'ensemble': {'average_time': 0, 'std_time': 0, 'min_time': 0, 'max_time': 0}
        }
    
    # 生成集成方法对比报告
    report_file, report_md_file = generate_ensemble_comparison_report(test_results, performance_results, args.output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("集成分类方法测试完成！")
    logger.info(f"JSON报告: {report_file}")
    logger.info(f"Markdown报告: {report_md_file}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
