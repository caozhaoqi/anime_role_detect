#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新模型基准测试脚本
专门测试最新训练的 efficientnet_b3_loli_optimized_v2 模型
"""

import os
import sys
import json
import time
import torch
from torchvision import models, transforms
from PIL import Image
from collections import defaultdict

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

class BenchmarkEvaluator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_dir = os.path.join(project_root, 'models', model_name)
        self.data_dir = os.path.join(project_root, 'data', 'final_dataset')
        self.model = None
        self.device = None
        self.class_names = None
        self.transform = None
        self.config = {}
    
    def load_config(self):
        """加载训练配置"""
        config_path = os.path.join(self.model_dir, 'training_results.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        return self.config
    
    def load_model(self):
        """加载模型"""
        # 获取设备
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"📦 使用设备: {self.device}")
        
        # 获取配置
        num_classes = self.config.get('num_classes', 76)
        image_size = self.config.get('image_size', 224)
        
        # 创建模型
        model_path = os.path.join(self.model_dir, 'model_full.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(self.model_dir, 'model_best.pth')
        
        print(f"🔄 加载模型: {model_path}")
        
        model = models.efficientnet_b3(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
        
        self.model = model.to(self.device)
        self.model.eval()
        
        # 获取类别名称
        self.class_names = self.config.get('class_names', [f"class_{i}" for i in range(num_classes)])
        
        # 定义变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("✅ 模型加载成功")
    
    def prepare_test_data(self, max_samples_per_class=20):
        """准备测试数据"""
        test_data = []
        
        # 创建类别映射（拼音 -> 罗马音）
        pinyin_to_romaji = {
            'xiao3niao3you2xing1ye3': 'Hoshino',
            'fu2xuan2': 'Fu',
            'ji1ban3nai3ai4': 'Aris',
            'arona': 'Arona',
        }
        
        # 直接读取类别目录（支持两种结构）
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            # 检查是否是batch目录（包含子目录）
            has_subdirs = any(os.path.isdir(os.path.join(class_path, f)) for f in os.listdir(class_path))
            
            if has_subdirs:
                # 结构: data_dir/batch_dir/class_dir/image.jpg
                for sub_class_dir in os.listdir(class_path):
                    sub_class_path = os.path.join(class_path, sub_class_dir)
                    if not os.path.isdir(sub_class_path):
                        continue
                    
                    true_class = pinyin_to_romaji.get(sub_class_dir, sub_class_dir)
                    if true_class not in self.class_names:
                        continue
                    
                    files = [f for f in os.listdir(sub_class_path) if f.endswith('.jpg')][:max_samples_per_class]
                    for filename in files:
                        image_path = os.path.join(sub_class_path, filename)
                        test_data.append((image_path, true_class))
            else:
                # 结构: data_dir/class_dir/image.jpg
                true_class = pinyin_to_romaji.get(class_dir.lower(), class_dir)
                if true_class not in self.class_names:
                    continue
                
                files = [f for f in os.listdir(class_path) if f.endswith('.jpg')][:max_samples_per_class]
                for filename in files:
                    image_path = os.path.join(class_path, filename)
                    test_data.append((image_path, true_class))
        
        print(f"📊 准备测试数据: {len(test_data)} 张图片")
        return test_data
    
    def test_accuracy(self, test_data):
        """测试准确率"""
        print("\n🔍 开始准确率测试...")
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for image_path, true_class in test_data:
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image).unsqueeze(0).to(self.device)
                    
                    outputs = self.model(image)
                    _, pred = torch.max(outputs, 1)
                    pred_class = self.class_names[pred.item()]
                    
                    total += 1
                    if pred_class == true_class:
                        correct += 1
                except Exception as e:
                    print(f"⚠️ 跳过损坏图片: {os.path.basename(image_path)}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"✅ 准确率测试完成")
        print(f"   样本数: {total}, 正确: {correct}, 准确率: {accuracy*100:.2f}%")
        
        return accuracy
    
    def test_top_k_accuracy(self, test_data, k=5):
        """测试Top-K准确率"""
        print(f"\n🎯 开始 Top-{k} 准确率测试...")
        
        total = 0
        correct = 0
        
        with torch.no_grad():
            for image_path, true_class in test_data:
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image).unsqueeze(0).to(self.device)
                    
                    outputs = self.model(image)
                    _, top_k_preds = torch.topk(outputs, k)
                    top_k_classes = [self.class_names[i] for i in top_k_preds[0]]
                    
                    total += 1
                    if true_class in top_k_classes:
                        correct += 1
                except:
                    continue
        
        accuracy = correct / total if total > 0 else 0
        print(f"✅ Top-{k} 准确率: {accuracy*100:.2f}%")
        
        return accuracy
    
    def test_inference_speed(self, test_data, num_samples=100):
        """测试推理速度"""
        print("\n⚡ 开始推理速度测试...")
        
        # 准备测试张量
        test_tensors = []
        for image_path, _ in test_data[:min(num_samples, len(test_data))]:
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.transform(image).unsqueeze(0)
                test_tensors.append(image)
            except:
                continue
        
        # 预热
        with torch.no_grad():
            for img in test_tensors[:5]:
                _ = self.model(img.to(self.device))
        
        # 测试
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                idx = _ % len(test_tensors)
                _ = self.model(test_tensors[idx].to(self.device))
        
        elapsed_time = time.time() - start_time
        fps = 100 / elapsed_time
        
        print(f"✅ 推理速度测试完成")
        print(f"   测试次数: 100次")
        print(f"   耗时: {elapsed_time:.4f}秒")
        print(f"   FPS: {fps:.2f}")
        print(f"   单图耗时: {(elapsed_time/100*1000):.2f}ms")
        
        return fps
    
    def run_benchmark(self):
        """运行完整基准测试"""
        print("=" * 60)
        print(f"🚀 开始模型基准测试: {self.model_name}")
        print("=" * 60)
        
        # 加载配置
        self.load_config()
        
        # 打印配置信息
        print(f"\n📋 模型配置:")
        print(f"   模型名称: {self.config.get('model_name', 'unknown')}")
        print(f"   类别数量: {self.config.get('num_classes', 76)}")
        print(f"   图像大小: {self.config.get('image_size', 224)}")
        print(f"   训练最佳准确率: {self.config.get('best_val_acc', 0)*100:.2f}%")
        
        # 加载模型
        self.load_model()
        
        # 准备测试数据
        test_data = self.prepare_test_data()
        
        if not test_data:
            print("❌ 没有找到测试数据")
            return
        
        # 运行测试
        results = {}
        
        # 准确率测试
        results['top_1_accuracy'] = self.test_accuracy(test_data)
        
        # Top-K准确率测试
        results['top_3_accuracy'] = self.test_top_k_accuracy(test_data, k=3)
        results['top_5_accuracy'] = self.test_top_k_accuracy(test_data, k=5)
        
        # 推理速度测试
        results['fps'] = self.test_inference_speed(test_data)
        
        # 保存结果
        results['model_name'] = self.model_name
        results['config'] = self.config
        results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        output_path = os.path.join(self.model_dir, 'benchmark_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印报告
        print("\n" + "=" * 60)
        print("📊 基准测试报告")
        print("=" * 60)
        print(f"模型名称: {self.model_name}")
        print(f"训练最佳准确率: {self.config.get('best_val_acc', 0)*100:.2f}%")
        print("-" * 60)
        print(f"Top-1 准确率: {results['top_1_accuracy']*100:.2f}%")
        print(f"Top-3 准确率: {results['top_3_accuracy']*100:.2f}%")
        print(f"Top-5 准确率: {results['top_5_accuracy']*100:.2f}%")
        print(f"推理速度: {results['fps']:.2f} FPS")
        print("=" * 60)
        print(f"\n✅ 测试结果已保存: {output_path}")
        
        return results

def main():
    # 新训练的模型名称
    model_name = "efficientnet_b3_loli_optimized_v2_20260529_133654"
    
    evaluator = BenchmarkEvaluator(model_name)
    evaluator.run_benchmark()

if __name__ == "__main__":
    main()
