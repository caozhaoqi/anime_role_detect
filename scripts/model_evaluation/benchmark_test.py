#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型基准测试脚本 - 评估模型性能指标
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

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BenchmarkEvaluator:
    """
    模型基准测试器
    """
    
    def __init__(self, model_path, results_path, test_dir):
        """
        初始化测试器
        
        Args:
            model_path: 模型文件路径
            results_path: 训练结果JSON路径
            test_dir: 测试数据目录
        """
        self.model_path = model_path
        self.results_path = results_path
        self.test_dir = test_dir
        self.model = None
        self.device = None
        self.class_names = None
        self.transform = None
        
    def load_model(self):
        """
        加载训练好的模型
        """
        # 读取训练结果配置
        with open(self.results_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 获取设备
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"📦 使用设备: {self.device}")
        
        # 创建模型架构
        num_classes = self.config['num_classes']
        model_name = self.config.get('model_name', 'mobilenetv2')
        
        if model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(num_classes=num_classes)
        elif model_name == 'efficientnet_b3':
            self.model = models.efficientnet_b3(num_classes=num_classes)
        elif model_name == 'resnet18':
            self.model = models.resnet18(num_classes=num_classes)
        else:
            self.model = models.mobilenet_v2(num_classes=num_classes)
        
        # 加载权重
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型加载成功: {self.model_path}")
        
        # 加载类别名称
        self._load_class_names()
        
        # 定义图像变换
        self.transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_class_names(self):
        """
        加载类别名称列表
        """
        if 'class_names' in self.config:
            self.class_names = self.config['class_names']
        else:
            # 从数据目录获取类别名称
            if os.path.exists(self.test_dir):
                self.class_names = sorted([d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))])
            else:
                self.class_names = [f"class_{i}" for i in range(self.config['num_classes'])]
    
    def test_accuracy(self):
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
                
                for filename in os.listdir(class_dir):
                    if not filename.endswith('.jpg'):
                        continue
                    
                    image_path = os.path.join(class_dir, filename)
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image).unsqueeze(0).to(self.device)
                    
                    outputs = self.model(image)
                    _, pred = torch.max(outputs, 1)
                    
                    # 获取真实标签索引
                    if class_name in self.class_names:
                        true_idx = self.class_names.index(class_name)
                    else:
                        continue
                    
                    total += 1
                    per_class_stats[class_name]['total'] += 1
                    
                    if pred.item() == true_idx:
                        correct += 1
                        per_class_stats[class_name]['correct'] += 1
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"✅ 准确率测试完成")
        print(f"   总样本数: {total}")
        print(f"   正确数: {correct}")
        print(f"   准确率: {accuracy * 100:.2f}%")
        
        return accuracy, per_class_stats
    
    def test_inference_speed(self, num_samples=100):
        """
        测试推理速度
        """
        print("\n⚡ 开始推理速度测试...")
        
        # 收集测试图片
        test_images = []
        for class_name in os.listdir(self.test_dir):
            class_dir = os.path.join(self.test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(class_dir, filename)
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image).unsqueeze(0)
                    test_images.append(image)
                    if len(test_images) >= num_samples:
                        break
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
    
    def test_memory_usage(self):
        """
        测试内存使用
        """
        print("\n💾 开始内存使用测试...")
        
        # 获取初始内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # 运行一次推理
        test_images = []
        for class_name in os.listdir(self.test_dir):
            class_dir = os.path.join(self.test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(class_dir, filename)
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image).unsqueeze(0).to(self.device)
                    test_images.append(image)
                    if len(test_images) >= 10:
                        break
            if len(test_images) >= 10:
                break
        
        with torch.no_grad():
            for img in test_images:
                _ = self.model(img)
        
        # 获取推理后内存使用
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        print(f"✅ 内存使用测试完成")
        print(f"   初始内存: {initial_memory:.2f} MB")
        print(f"   峰值内存: {peak_memory:.2f} MB")
        print(f"   模型占用: {(peak_memory - initial_memory):.2f} MB")
        
        return {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'model_memory_mb': peak_memory - initial_memory
        }
    
    def test_top_k_accuracy(self, k=5):
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
                
                for filename in os.listdir(class_dir):
                    if not filename.endswith('.jpg'):
                        continue
                    
                    image_path = os.path.join(class_dir, filename)
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image).unsqueeze(0).to(self.device)
                    
                    outputs = self.model(image)
                    _, top_k_preds = torch.topk(outputs, k)
                    
                    # 获取真实标签索引
                    if class_name in self.class_names:
                        true_idx = self.class_names.index(class_name)
                    else:
                        continue
                    
                    total += 1
                    
                    if true_idx in top_k_preds[0].tolist():
                        top_k_correct += 1
        
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
        results['per_class_stats'] = {k: dict(v) for k, v in per_class_stats.items()}
        
        # Top-K 准确率测试
        top_5_acc = self.test_top_k_accuracy(k=5)
        results['top_5_accuracy'] = top_5_acc
        
        # 推理速度测试
        fps, elapsed_time = self.test_inference_speed()
        results['fps'] = fps
        results['inference_time_per_image_ms'] = (elapsed_time / 100) * 1000
        
        # 内存使用测试
        memory_stats = self.test_memory_usage()
        results['memory'] = memory_stats
        
        # 添加模型信息
        results['model_name'] = self.config.get('model_name', 'mobilenetv2')
        results['num_classes'] = self.config['num_classes']
        results['image_size'] = self.config['image_size']
        results['augment_level'] = self.config.get('augment_level', 'none')
        
        print("\n" + "=" * 60)
        print("📊 基准测试报告")
        print("=" * 60)
        print(f"模型名称: {results['model_name']}")
        print(f"类别数量: {results['num_classes']}")
        print(f"图像大小: {results['image_size']}x{results['image_size']}")
        print(f"数据增强: {results['augment_level']}")
        print("-" * 60)
        print(f"准确率: {results['accuracy'] * 100:.2f}%")
        print(f"Top-5 准确率: {results['top_5_accuracy'] * 100:.2f}%")
        print(f"推理速度: {results['fps']:.2f} FPS")
        print(f"单图推理时间: {results['inference_time_per_image_ms']:.2f} ms")
        print(f"模型内存占用: {results['memory']['model_memory_mb']:.2f} MB")
        
        # 保存测试结果
        output_path = os.path.join(os.path.dirname(self.model_path), 'benchmark_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ 测试结果已保存: {output_path}")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="模型基准测试工具")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--results_path", type=str, required=True, help="训练结果JSON路径")
    parser.add_argument("--test_dir", type=str, default=None, help="测试数据目录（可选，默认使用训练数据目录）")
    
    args = parser.parse_args()
    
    # 如果没有指定测试目录，使用expanded_dataset
    if args.test_dir is None:
        # 获取项目根目录（向上走3级：scripts/model_evaluation/benchmark_test.py）
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.test_dir = os.path.join(project_root, 'data', 'expanded_dataset')
    
    evaluator = BenchmarkEvaluator(args.model_path, args.results_path, args.test_dir)
    evaluator.run_benchmark()


if __name__ == "__main__":
    main()