#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速基准测试脚本
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # 模型路径
    model_path = 'models/efficientnet_b3_loli_76_pretrained_20260520_162729/model_best.pth'
    results_path = 'models/efficientnet_b3_loli_76_pretrained_20260520_162729/training_results.json'
    data_dir = 'data/expanded_dataset'
    
    print("🚀 快速基准测试")
    print("=" * 50)
    
    # 加载配置
    with open(results_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"📦 设备: {device}")
    
    # 创建模型
    model = models.efficientnet_b3(num_classes=config['num_classes'])
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"✅ 模型加载成功")
    
    # 变换
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 快速准确率测试（采样10%数据）
    print("\n🔍 准确率测试 (采样10%)...")
    correct = 0
    total = 0
    sample_count = 0
    max_samples = 1000
    
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    with torch.no_grad():
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for filename in os.listdir(class_dir):
                if not filename.endswith('.jpg'):
                    continue
                
                image_path = os.path.join(class_dir, filename)
                image = Image.open(image_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)
                
                outputs = model(image)
                _, pred = torch.max(outputs, 1)
                
                if class_name in class_names:
                    true_idx = class_names.index(class_name)
                    
                    total += 1
                    if pred.item() == true_idx:
                        correct += 1
                
                sample_count += 1
                if sample_count >= max_samples:
                    break
            if sample_count >= max_samples:
                break
    
    accuracy = correct / total if total > 0 else 0
    print(f"   样本数: {total}, 正确: {correct}, 准确率: {accuracy * 100:.2f}%")
    
    # 推理速度测试
    print("\n⚡ 推理速度测试...")
    test_images = []
    for class_name in class_names[:3]:
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir)[:10]:
            if filename.endswith('.jpg'):
                image_path = os.path.join(class_dir, filename)
                image = Image.open(image_path).convert('RGB')
                image = transform(image).unsqueeze(0)
                test_images.append(image)
    
    # 预热
    with torch.no_grad():
        for img in test_images[:3]:
            _ = model(img.to(device))
    
    # 测试
    start_time = time.time()
    with torch.no_grad():
        for img in test_images:
            _ = model(img.to(device))
    elapsed_time = time.time() - start_time
    fps = len(test_images) / elapsed_time
    
    print(f"   测试样本: {len(test_images)}, 耗时: {elapsed_time:.4f}s, FPS: {fps:.2f}")
    
    # 内存测试
    print("\n💾 内存测试...")
    import psutil
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / (1024 * 1024)
    print(f"   当前内存: {memory:.2f} MB")
    
    print("\n" + "=" * 50)
    print("📊 基准测试报告")
    print("=" * 50)
    print(f"模型: {config['model_name']}")
    print(f"类别数: {config['num_classes']}")
    print(f"图像大小: {config['image_size']}x{config['image_size']}")
    print(f"准确率: {accuracy * 100:.2f}%")
    print(f"推理速度: {fps:.2f} FPS")
    print(f"内存占用: {memory:.2f} MB")
    print(f"训练最佳准确率: {config['best_accuracy'] * 100:.2f}%")

if __name__ == "__main__":
    main()