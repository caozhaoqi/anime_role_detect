#!/usr/bin/env python3
"""
模型基准测试脚本
测试模型的性能、内存使用和准确性等指标
生成详细的基准测试报告
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
logger = logging.getLogger('benchmark_model')

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入通用分类模块
from src.core.general_classification import GeneralClassification, get_classifier

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

def benchmark_inference_speed(classifier, test_image_path, batch_sizes=[1], iterations=10):
    """基准测试推理速度
    
    Args:
        classifier: 分类器实例
        test_image_path: 测试图像路径
        batch_sizes: 批量大小列表
        iterations: 每个批量大小的测试迭代次数
    
    Returns:
        推理速度测试结果
    """
    logger.info("开始基准测试推理速度...")
    
    results = []
    
    for batch_size in batch_sizes:
        logger.info(f"测试批量大小: {batch_size}")
        
        # 准备批量图像
        batch_images = [test_image_path] * batch_size
        
        # 预热
        logger.info("  预热中...")
        for i in range(2):
            for image_path in batch_images:
                classifier.classify_image(image_path, use_model=False)
        
        # 测试推理时间
        logger.info(f"  执行 {iterations} 次迭代...")
        inference_times = []
        
        for i in range(iterations):
            start_time = time.time()
            
            for image_path in batch_images:
                classifier.classify_image(image_path, use_model=False)
            
            end_time = time.time()
            iteration_time = end_time - start_time
            inference_times.append(iteration_time)
            
            logger.info(f"    迭代 {i+1}/{iterations}: {iteration_time:.4f}秒")
        
        # 计算统计信息
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        # 计算每秒处理图像数 (FPS)
        fps = batch_size * iterations / sum(inference_times)
        
        results.append({
            'batch_size': batch_size,
            'iterations': iterations,
            'average_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'fps': fps,
            'per_image_time': avg_time / batch_size
        })
        
        logger.info(f"  批量大小 {batch_size} 结果:")
        logger.info(f"    平均时间: {avg_time:.4f}秒")
        logger.info(f"    标准差: {std_time:.4f}秒")
        logger.info(f"    最小时间: {min_time:.4f}秒")
        logger.info(f"    最大时间: {max_time:.4f}秒")
        logger.info(f"    FPS: {fps:.2f}")
        logger.info(f"    每张图像时间: {avg_time / batch_size:.4f}秒")
    
    return results

def benchmark_memory_usage(classifier, test_image_path, iterations=5):
    """基准测试内存使用情况
    
    Args:
        classifier: 分类器实例
        test_image_path: 测试图像路径
        iterations: 测试迭代次数
    
    Returns:
        内存使用测试结果
    """
    logger.info("开始基准测试内存使用情况...")
    
    # 初始内存使用
    initial_memory = get_memory_usage()
    logger.info(f"初始内存使用: {initial_memory['rss']:.2f} MB")
    
    # 执行分类，观察内存变化
    memory_usages = []
    
    for i in range(iterations):
        logger.info(f"内存测试迭代 {i+1}/{iterations}...")
        
        # 分类图像
        classifier.classify_image(test_image_path, use_model=False)
        
        # 记录内存使用
        mem_usage = get_memory_usage()
        memory_usages.append(mem_usage)
        logger.info(f"  内存使用: {mem_usage['rss']:.2f} MB")
    
    # 计算平均内存使用
    avg_rss = np.mean([m['rss'] for m in memory_usages])
    max_rss = np.max([m['rss'] for m in memory_usages])
    
    return {
        'initial_memory': initial_memory,
        'average_memory': avg_rss,
        'max_memory': max_rss,
        'memory_increase': avg_rss - initial_memory['rss'],
        'memory_usages': memory_usages
    }

def benchmark_accuracy(classifier, test_image_path, expected_role, iterations=5):
    """基准测试模型准确性
    
    Args:
        classifier: 分类器实例
        test_image_path: 测试图像路径
        expected_role: 期望角色
        iterations: 测试迭代次数
    
    Returns:
        准确性测试结果
    """
    logger.info("开始基准测试模型准确性...")
    
    results = {
        'expected_role': expected_role,
        'iterations': iterations,
        'correct_predictions': 0,
        'predictions': []
    }
    
    for i in range(iterations):
        logger.info(f"准确性测试迭代 {i+1}/{iterations}...")
        
        # 分类图像
        predicted_role, similarity, boxes = classifier.classify_image(test_image_path, use_model=False)
        
        # 验证预测结果
        is_correct = predicted_role == expected_role
        if is_correct:
            results['correct_predictions'] += 1
        
        results['predictions'].append({
            'iteration': i+1,
            'predicted_role': predicted_role,
            'similarity': similarity,
            'is_correct': is_correct
        })
        
        logger.info(f"  预测: {predicted_role}, 相似度: {similarity:.4f}, 正确: {is_correct}")
    
    # 计算准确率
    results['accuracy'] = results['correct_predictions'] / iterations * 100
    
    return results

def benchmark_input_sizes(classifier, test_image_path, input_sizes=[(224, 224), (256, 256), (384, 384), (512, 512)], iterations=3):
    """基准测试不同输入大小的性能
    
    Args:
        classifier: 分类器实例
        test_image_path: 测试图像路径
        input_sizes: 输入大小列表
        iterations: 每个输入大小的测试迭代次数
    
    Returns:
        输入大小测试结果
    """
    logger.info("开始基准测试不同输入大小的性能...")
    
    results = []
    
    for size in input_sizes:
        logger.info(f"测试输入大小: {size}")
        
        # 调整图像大小
        image = Image.open(test_image_path)
        resized_image = image.resize(size)
        
        # 保存调整大小后的图像
        resized_path = f"/tmp/resized_{size[0]}x{size[1]}.jpg"
        resized_image.save(resized_path)
        
        # 测试推理时间
        inference_times = []
        
        for i in range(iterations):
            start_time = time.time()
            classifier.classify_image(resized_path, use_model=False)
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            logger.info(f"    迭代 {i+1}/{iterations}: {inference_time:.4f}秒")
        
        # 计算统计信息
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        
        results.append({
            'input_size': size,
            'average_time': avg_time,
            'std_time': std_time,
            'iterations': iterations
        })
        
        logger.info(f"  输入大小 {size} 结果:")
        logger.info(f"    平均时间: {avg_time:.4f}秒")
        logger.info(f"    标准差: {std_time:.4f}秒")
    
    return results

def generate_benchmark_report(benchmark_results, output_dir="benchmark_reports"):
    """生成基准测试报告
    
    Args:
        benchmark_results: 基准测试结果字典
        output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成报告文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"benchmark_report_{timestamp}.json")
    report_md_file = os.path.join(output_dir, f"benchmark_report_{timestamp}.md")
    
    # 保存JSON报告
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"JSON基准测试报告已保存到: {report_file}")
    
    # 生成Markdown报告
    with open(report_md_file, 'w', encoding='utf-8') as f:
        f.write(f"# 模型基准测试报告\n")
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
        
        # 推理速度测试
        if 'inference_speed' in benchmark_results:
            f.write("## 推理速度测试\n\n")
            f.write("| 批量大小 | 迭代次数 | 平均时间 (秒) | 标准差 (秒) | 最小时间 (秒) | 最大时间 (秒) | FPS | 每张图像时间 (秒) |\n")
            f.write("|---------|---------|--------------|------------|--------------|--------------|-----|------------------|\n")
            
            for result in benchmark_results['inference_speed']:
                f.write(f"| {result['batch_size']} | {result['iterations']} | {result['average_time']:.4f} | {result['std_time']:.4f} | {result['min_time']:.4f} | {result['max_time']:.4f} | {result['fps']:.2f} | {result['per_image_time']:.4f} |\n")
            
            f.write("\n")
        
        # 内存使用测试
        if 'memory_usage' in benchmark_results:
            mem_result = benchmark_results['memory_usage']
            f.write("## 内存使用测试\n\n")
            f.write(f"| 指标 | 值 |\n")
            f.write(f"|------|-----|\n")
            f.write(f"| 初始内存使用 | {mem_result['initial_memory']['rss']:.2f} MB |\n")
            f.write(f"| 平均内存使用 | {mem_result['average_memory']:.2f} MB |\n")
            f.write(f"| 最大内存使用 | {mem_result['max_memory']:.2f} MB |\n")
            f.write(f"| 内存增加 | {mem_result['memory_increase']:.2f} MB |\n\n")
        
        # 准确性测试
        if 'accuracy' in benchmark_results:
            acc_result = benchmark_results['accuracy']
            f.write("## 准确性测试\n\n")
            f.write(f"| 指标 | 值 |\n")
            f.write(f"|------|-----|\n")
            f.write(f"| 期望角色 | {acc_result['expected_role']} |\n")
            f.write(f"| 测试迭代次数 | {acc_result['iterations']} |\n")
            f.write(f"| 正确预测次数 | {acc_result['correct_predictions']} |\n")
            f.write(f"| 准确率 | {acc_result['accuracy']:.2f}% |\n\n")
        
        # 输入大小测试
        if 'input_sizes' in benchmark_results:
            f.write("## 输入大小测试\n\n")
            f.write("| 输入大小 | 迭代次数 | 平均时间 (秒) | 标准差 (秒) |\n")
            f.write("|---------|---------|--------------|------------|\n")
            
            for result in benchmark_results['input_sizes']:
                f.write(f"| {result['input_size']} | {result['iterations']} | {result['average_time']:.4f} | {result['std_time']:.4f} |\n")
            
            f.write("\n")
        
        # 结论和建议
        f.write("## 结论和建议\n\n")
        f.write("### 性能评估\n")
        f.write("- **推理速度**: 模型在默认配置下的推理速度为每张图像约 X 秒\n")
        f.write("- **内存使用**: 模型运行时内存增加约 X MB\n")
        f.write("- **准确性**: 模型在测试图像上的准确率为 X%\n\n")
        
        f.write("### 优化建议\n")
        f.write("1. **批量处理**: 考虑使用批量处理提高 throughput\n")
        f.write("2. **输入大小**: 根据性能需求选择合适的输入大小\n")
        f.write("3. **内存优化**: 如果内存使用过高，考虑模型量化或裁剪\n")
        f.write("4. **并行处理**: 对于批量处理，考虑使用多线程或多进程\n")
        f.write("5. **模型选择**: 根据实际应用场景选择合适的模型\n")
    
    logger.info(f"Markdown基准测试报告已保存到: {report_md_file}")
    return report_file, report_md_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型基准测试工具')
    
    # 输入参数
    parser.add_argument('--test_image', type=str, default=None, help='测试图像路径')
    parser.add_argument('--expected_role', type=str, default='Unknown', help='期望角色')
    parser.add_argument('--batch_sizes', type=str, default='1,2,4', help='批量大小列表，逗号分隔')
    parser.add_argument('--iterations', type=int, default=5, help='测试迭代次数')
    parser.add_argument('--input_sizes', type=str, default='224x224,256x256,384x384', help='输入大小列表，逗号分隔')
    parser.add_argument('--output_dir', type=str, default='benchmark_reports', help='基准测试报告输出目录')
    
    args = parser.parse_args()
    
    # 解析批量大小
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
    
    # 解析输入大小
    input_sizes = []
    for size_str in args.input_sizes.split(','):
        width, height = map(int, size_str.split('x'))
        input_sizes.append((width, height))
    
    # 如果没有指定测试图像，使用默认图像
    if not args.test_image:
        # 查找测试图像
        test_image = None
        for root, dirs, files in os.walk('data'):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image = os.path.join(root, file)
                    break
            if test_image:
                break
        
        if not test_image:
            logger.error("没有找到测试图像")
            return
        
        args.test_image = test_image
    
    logger.info(f"使用测试图像: {args.test_image}")
    
    # 初始化分类器
    logger.info("初始化分类器...")
    classifier = get_classifier()
    classifier.initialize()
    
    # 执行基准测试
    benchmark_results = {
        'timestamp': datetime.now().isoformat(),
        'test_image': args.test_image,
        'expected_role': args.expected_role,
        'batch_sizes': batch_sizes,
        'iterations': args.iterations,
        'input_sizes': input_sizes
    }
    
    # 测试推理速度
    benchmark_results['inference_speed'] = benchmark_inference_speed(
        classifier, args.test_image, batch_sizes, args.iterations
    )
    
    # 测试内存使用
    benchmark_results['memory_usage'] = benchmark_memory_usage(
        classifier, args.test_image, args.iterations
    )
    
    # 测试准确性
    benchmark_results['accuracy'] = benchmark_accuracy(
        classifier, args.test_image, args.expected_role, args.iterations
    )
    
    # 测试不同输入大小
    benchmark_results['input_sizes'] = benchmark_input_sizes(
        classifier, args.test_image, input_sizes, args.iterations
    )
    
    # 生成基准测试报告
    report_file, report_md_file = generate_benchmark_report(benchmark_results, args.output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("基准测试完成！")
    logger.info(f"JSON基准测试报告: {report_file}")
    logger.info(f"Markdown基准测试报告: {report_md_file}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
