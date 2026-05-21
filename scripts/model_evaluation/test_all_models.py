#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测试与基准测试脚本
1. 验证所有模型是否可用
2. 对可用模型进行基准测试
3. 生成性能报告和图表
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import defaultdict
import psutil
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelTester:
    """
    模型测试器
    """
    
    def __init__(self):
        # 获取项目根目录（向上走3级）
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = os.path.join(project_root, 'models')
        self.data_dir = os.path.join(project_root, 'data', 'expanded_dataset')
        self.results = {}
        
    def find_models(self):
        """
        查找所有可用模型
        """
        model_info = []
        
        # 遍历模型目录
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            if not os.path.isdir(item_path):
                continue
                
            # 检查是否有模型文件
            model_best_path = os.path.join(item_path, 'model_best.pth')
            model_full_path = os.path.join(item_path, 'model_full.pth')
            training_results_path = os.path.join(item_path, 'training_results.json')
            
            if os.path.exists(model_best_path) or os.path.exists(model_full_path):
                model_info.append({
                    'name': item,
                    'path': item_path,
                    'model_best_exists': os.path.exists(model_best_path),
                    'model_full_exists': os.path.exists(model_full_path),
                    'training_results_exists': os.path.exists(training_results_path)
                })
        
        return model_info
    
    def load_model(self, model_info):
        """
        加载模型并验证是否可用
        """
        try:
            # 获取设备
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            # 确定模型路径
            if model_info['model_full_exists']:
                model_path = os.path.join(model_info['path'], 'model_full.pth')
            else:
                model_path = os.path.join(model_info['path'], 'model_best.pth')
            
            # 尝试加载模型
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # 初始化配置
            config = {}
            training_results_path = os.path.join(model_info['path'], 'training_results.json')
            if os.path.exists(training_results_path):
                with open(training_results_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            # 检查是完整模型还是权重
            if isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
                model_type = 'full_model'
            else:
                # 需要从training_results获取配置
                model_name = config.get('model_name', config.get('model', 'mobilenetv2'))
                num_classes = config.get('num_classes', 76)
                
                # 创建模型架构
                if 'efficientnet_b3' in model_info['name'] or model_name == 'efficientnet_b3':
                    model = models.efficientnet_b3(num_classes=num_classes)
                elif 'efficientnet_b0' in model_info['name'] or model_name == 'efficientnet_b0':
                    model = models.efficientnet_b0(num_classes=num_classes)
                elif 'resnet' in model_info['name']:
                    model = models.resnet18(num_classes=num_classes)
                else:
                    model = models.mobilenet_v2(num_classes=num_classes)
                
                # 加载权重
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                model.load_state_dict(state_dict, strict=False)
                model_type = 'state_dict'
            
            model = model.to(device)
            model.eval()
            
            return {
                'success': True,
                'device': str(device),
                'model_type': model_type,
                'config': config,
                'model': model
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_model_availability(self):
        """
        测试所有模型的可用性
        """
        print("🔍 开始测试模型可用性...")
        model_info_list = self.find_models()
        results = []
        
        for model_info in model_info_list:
            print(f"\n测试模型: {model_info['name']}")
            result = self.load_model(model_info)
            
            if result['success']:
                print(f"✅ 模型可用")
                print(f"   - 设备: {result['device']}")
                print(f"   - 类型: {result['model_type']}")
                if 'config' in result and result['config']:
                    print(f"   - 类别数: {result['config'].get('num_classes', '未知')}")
                    print(f"   - 图像大小: {result['config'].get('image_size', '未知')}")
            else:
                print(f"❌ 模型不可用: {result['error']}")
            
            results.append({
                'name': model_info['name'],
                'path': model_info['path'],
                'available': result['success'],
                'error': result.get('error'),
                'config': result.get('config', {})
            })
        
        return results


class BenchmarkEvaluator:
    """
    模型基准测试器
    """
    
    def __init__(self, model_path, config, test_dir):
        self.model_path = model_path
        self.config = config
        self.test_dir = test_dir
        self.model = None
        self.device = None
        self.class_names = None
        self.transform = None
        
    def load_model(self):
        """
        加载训练好的模型
        """
        # 获取设备
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"📦 使用设备: {self.device}")
        
        # 获取模型配置
        num_classes = self.config.get('num_classes', 76)
        model_name = self.config.get('model_name', self.config.get('model', 'mobilenetv2'))
        image_size = self.config.get('image_size', 224)
        
        # 创建模型架构
        if 'efficientnet_b3' in model_name.lower() or 'efficientnet_b3' in self.model_path.lower():
            self.model = models.efficientnet_b3(num_classes=num_classes)
        elif 'efficientnet_b0' in model_name.lower() or 'efficientnet_b0' in self.model_path.lower():
            self.model = models.efficientnet_b0(num_classes=num_classes)
        elif 'resnet' in model_name.lower():
            self.model = models.resnet18(num_classes=num_classes)
        else:
            self.model = models.mobilenet_v2(num_classes=num_classes)
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, torch.nn.Module):
            self.model = checkpoint
        else:
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载类别名称
        if 'class_names' in self.config:
            self.class_names = self.config['class_names']
        else:
            if os.path.exists(self.test_dir):
                self.class_names = sorted([d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))])
            else:
                self.class_names = [f"class_{i}" for i in range(num_classes)]
        
        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"✅ 模型加载成功")
    
    def test_accuracy(self, max_samples_per_class=20):
        """
        测试准确率
        """
        print("\n🔍 开始准确率测试...")
        
        correct = 0
        total = 0
        per_class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        with torch.no_grad():
            for class_name in os.listdir(self.test_dir):
                class_dir = os.path.join(self.test_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                
                if class_name not in self.class_names:
                    continue
                
                files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')][:max_samples_per_class]
                
                for filename in files:
                    image_path = os.path.join(class_dir, filename)
                    try:
                        image = Image.open(image_path).convert('RGB')
                        image = self.transform(image).unsqueeze(0).to(self.device)
                        
                        outputs = self.model(image)
                        _, pred = torch.max(outputs, 1)
                        
                        true_idx = self.class_names.index(class_name)
                        
                        total += 1
                        per_class_stats[class_name]['total'] += 1
                        
                        if pred.item() == true_idx:
                            correct += 1
                            per_class_stats[class_name]['correct'] += 1
                    except Exception as e:
                        print(f"⚠️ 处理图片失败: {image_path}, 错误: {e}")
                        continue
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"✅ 准确率测试完成")
        print(f"   总样本数: {total}")
        print(f"   正确数: {correct}")
        print(f"   准确率: {accuracy * 100:.2f}%")
        
        return accuracy, dict(per_class_stats)
    
    def test_inference_speed(self, num_samples=100):
        """
        测试推理速度
        """
        print("\n⚡ 开始推理速度测试...")
        
        test_images = []
        for class_name in os.listdir(self.test_dir):
            class_dir = os.path.join(self.test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            if class_name not in self.class_names:
                continue
            
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg'):
                    try:
                        image_path = os.path.join(class_dir, filename)
                        image = Image.open(image_path).convert('RGB')
                        image = self.transform(image).unsqueeze(0)
                        test_images.append(image)
                        if len(test_images) >= num_samples:
                            break
                    except:
                        continue
            if len(test_images) >= num_samples:
                break
        
        # 预热
        with torch.no_grad():
            for img in test_images[:5]:
                _ = self.model(img.to(self.device))
        
        # 测试推理时间
        start_time = time.time()
        with torch.no_grad():
            for img in test_images:
                _ = self.model(img.to(self.device))
        
        elapsed_time = time.time() - start_time
        fps = len(test_images) / elapsed_time
        
        print(f"✅ 推理速度测试完成")
        print(f"   测试样本数: {len(test_images)}")
        print(f"   总耗时: {elapsed_time:.4f}秒")
        print(f"   FPS: {fps:.2f}")
        
        return fps, elapsed_time
    
    def test_top_k_accuracy(self, k=5, max_samples_per_class=20):
        """
        测试 Top-K 准确率
        """
        print(f"\n🎯 开始 Top-{k} 准确率测试...")
        
        total = 0
        top_k_correct = 0
        
        with torch.no_grad():
            for class_name in os.listdir(self.test_dir):
                class_dir = os.path.join(self.test_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                
                if class_name not in self.class_names:
                    continue
                
                files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')][:max_samples_per_class]
                
                for filename in files:
                    try:
                        image_path = os.path.join(class_dir, filename)
                        image = Image.open(image_path).convert('RGB')
                        image = self.transform(image).unsqueeze(0).to(self.device)
                        
                        outputs = self.model(image)
                        _, top_k_preds = torch.topk(outputs, k)
                        
                        true_idx = self.class_names.index(class_name)
                        
                        total += 1
                        
                        if true_idx in top_k_preds[0].tolist():
                            top_k_correct += 1
                    except:
                        continue
        
        top_k_accuracy = top_k_correct / total if total > 0 else 0
        
        print(f"✅ Top-{k} 准确率测试完成")
        print(f"   Top-{k} 准确率: {top_k_accuracy * 100:.2f}%")
        
        return top_k_accuracy
    
    def run_benchmark(self):
        """
        运行完整的基准测试
        """
        print("🚀 开始模型基准测试")
        print("=" * 60)
        
        self.load_model()
        
        results = {}
        
        # 准确率测试
        accuracy, per_class_stats = self.test_accuracy()
        results['accuracy'] = accuracy
        results['per_class_stats'] = per_class_stats
        
        # Top-K 准确率测试
        top_1_acc = accuracy
        top_3_acc = self.test_top_k_accuracy(k=3)
        top_5_acc = self.test_top_k_accuracy(k=5)
        results['top_1_accuracy'] = top_1_acc
        results['top_3_accuracy'] = top_3_acc
        results['top_5_accuracy'] = top_5_acc
        
        # 推理速度测试
        fps, elapsed_time = self.test_inference_speed()
        results['fps'] = fps
        results['inference_time_per_image_ms'] = (elapsed_time / 100) * 1000
        
        # 添加模型信息
        results['model_name'] = self.config.get('model_name', self.config.get('model', 'unknown'))
        results['num_classes'] = self.config.get('num_classes', 76)
        results['image_size'] = self.config.get('image_size', 224)
        results['augment_level'] = self.config.get('augment_level', 'none')
        results['model_path'] = self.model_path
        
        print("\n" + "=" * 60)
        print("📊 基准测试报告")
        print("=" * 60)
        print(f"模型名称: {results['model_name']}")
        print(f"类别数量: {results['num_classes']}")
        print(f"图像大小: {results['image_size']}x{results['image_size']}")
        print("-" * 60)
        print(f"Top-1 准确率: {results['top_1_accuracy'] * 100:.2f}%")
        print(f"Top-3 准确率: {results['top_3_accuracy'] * 100:.2f}%")
        print(f"Top-5 准确率: {results['top_5_accuracy'] * 100:.2f}%")
        print(f"推理速度: {results['fps']:.2f} FPS")
        print(f"单图推理时间: {results['inference_time_per_image_ms']:.2f} ms")
        
        # 保存测试结果
        output_path = os.path.join(os.path.dirname(self.model_path), 'benchmark_results_new.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 测试结果已保存: {output_path}")
        
        return results


def generate_summary_report(availability_results, benchmark_results_list, output_path):
    """
    生成汇总报告
    """
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_models_found': len(availability_results),
        'available_models': sum(1 for r in availability_results if r['available']),
        'unavailable_models': sum(1 for r in availability_results if not r['available']),
        'availability_results': availability_results,
        'benchmark_results': benchmark_results_list
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report


def print_summary_report(report):
    """
    打印汇总报告
    """
    print("\n" + "=" * 70)
    print("📋 模型测试汇总报告")
    print("=" * 70)
    print(f"生成时间: {report['timestamp']}")
    print(f"发现模型总数: {report['total_models_found']}")
    print(f"可用模型数: {report['available_models']}")
    print(f"不可用模型数: {report['unavailable_models']}")
    
    # 列出不可用模型
    print("\n❌ 不可用模型:")
    unavailable = [r for r in report['availability_results'] if not r['available']]
    if unavailable:
        for model in unavailable:
            print(f"  - {model['name']}: {model.get('error', '未知错误')}")
    else:
        print("  无")
    
    # 列出可用模型及其性能
    print("\n✅ 可用模型及性能:")
    for bench in report['benchmark_results']:
        print(f"\n  模型: {bench['model_name']} ({os.path.basename(bench['model_path'])})")
        print(f"    - Top-1 准确率: {bench['top_1_accuracy'] * 100:.2f}%")
        print(f"    - Top-3 准确率: {bench['top_3_accuracy'] * 100:.2f}%")
        print(f"    - Top-5 准确率: {bench['top_5_accuracy'] * 100:.2f}%")
        print(f"    - 推理速度: {bench['fps']:.2f} FPS")
    
    # 推荐最佳模型
    if report['benchmark_results']:
        best_model = max(report['benchmark_results'], key=lambda x: x['top_1_accuracy'])
        print(f"\n🏆 推荐最佳模型: {best_model['model_name']}")
        print(f"   准确率: {best_model['top_1_accuracy'] * 100:.2f}%")
        print(f"   推理速度: {best_model['fps']:.2f} FPS")
    
    print("\n" + "=" * 70)


def main():
    print("🎯 模型测试与基准测试工具")
    print("=" * 70)
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. 测试模型可用性
    tester = ModelTester()
    availability_results = tester.test_model_availability()
    
    # 2. 对可用模型进行基准测试
    benchmark_results_list = []
    data_dir = os.path.join(project_root, 'data', 'expanded_dataset')
    
    for model_info in availability_results:
        if not model_info['available']:
            continue
        
        print(f"\n{'='*60}")
        print(f"开始基准测试: {model_info['name']}")
        print(f"{'='*60}")
        
        # 获取模型路径
        model_path = os.path.join(model_info['path'], 'model_full.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(model_info['path'], 'model_best.pth')
        
        # 创建评估器并运行基准测试
        evaluator = BenchmarkEvaluator(model_path, model_info['config'], data_dir)
        try:
            bench_results = evaluator.run_benchmark()
            benchmark_results_list.append(bench_results)
        except Exception as e:
            print(f"❌ 基准测试失败: {e}")
            model_info['available'] = False
            model_info['error'] = f"基准测试失败: {e}"
    
    # 3. 生成汇总报告
    output_path = os.path.join(project_root, 'benchmark_summary_report.json')
    report = generate_summary_report(availability_results, benchmark_results_list, output_path)
    
    # 4. 打印报告
    print_summary_report(report)
    
    print(f"\n📄 完整报告已保存: {output_path}")


if __name__ == "__main__":
    main()