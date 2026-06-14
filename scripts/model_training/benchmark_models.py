#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型基准测试脚本
评估训练好的模型在测试集上的性能
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 配置参数
DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")

# 图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_device():
    """获取最佳设备"""
    if torch.backends.mps.is_available():
        print("✅ 使用 MPS 加速")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ 使用 CUDA 加速")
        return torch.device("cuda")
    else:
        print("⚠️ 使用 CPU")
        return torch.device("cpu")


def load_model(model_name, num_classes, device):
    """加载预训练模型"""
    if model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "EfficientNet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"未知模型: {model_name}")

    # 加载训练好的权重
    model_path = MODEL_DIR / f"{model_name.lower()}_best.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ 加载模型权重: {model_path}")
    else:
        print(f"❌ 模型权重文件不存在: {model_path}")
        return None

    return model.to(device)


def benchmark_model(model, dataloader, device, model_name):
    """基准测试单个模型"""
    model.eval()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    # 记录推理时间
    inference_times = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"测试 {model_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 记录开始时间
            start_time = time.time()
            
            outputs = model(inputs)
            
            # 记录推理时间
            inference_time = time.time() - start_time
            inference_times.extend([inference_time / inputs.size(0)] * inputs.size(0))
            
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            total_correct += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            total_loss += loss.item() * inputs.size(0)
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
    fps = 1.0 / np.mean(inference_times)
    
    print(f"\n{model_name} 基准测试结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  损失: {avg_loss:.4f}")
    print(f"  平均推理时间: {avg_inference_time:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'loss': avg_loss,
        'avg_inference_time_ms': avg_inference_time,
        'fps': fps
    }


def main():
    """主函数"""
    print("=" * 70)
    print("🎯 模型基准测试")
    print("=" * 70)
    
    # 获取设备
    device = get_device()
    
    # 加载数据集
    print("\n📦 加载测试数据集...")
    dataset = ImageFolder(DATA_DIR, transform=transform)
    
    # 按 80-20 分割训练集和测试集
    test_size = int(0.2 * len(dataset))
    _, test_dataset = random_split(dataset, [len(dataset) - test_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"  测试集样本数: {len(test_dataset)}")
    print(f"  类别数: {len(dataset.classes)}")
    
    # 定义模型列表
    model_names = ["MobileNetV2", "ResNet50", "EfficientNet"]
    results = []
    
    # 对每个模型进行基准测试
    for model_name in model_names:
        print(f"\n{'='*70}")
        model = load_model(model_name, len(dataset.classes), device)
        if model:
            result = benchmark_model(model, test_loader, device, model_name)
            results.append(result)
    
    # 输出综合对比
    print("\n" + "=" * 70)
    print("📊 模型性能对比")
    print("=" * 70)
    print(f"{'模型':<15} {'准确率':<10} {'损失':<10} {'推理时间(ms)':<15} {'FPS':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['model']:<15} {result['accuracy']:<10.4f} {result['loss']:<10.4f} {result['avg_inference_time_ms']:<15.2f} {result['fps']:<10.2f}")
    
    # 找出最佳模型
    if results:
        best_acc_model = max(results, key=lambda x: x['accuracy'])
        best_speed_model = min(results, key=lambda x: x['avg_inference_time_ms'])
        
        print("\n🏆 最佳模型:")
        print(f"  最高准确率: {best_acc_model['model']} ({best_acc_model['accuracy']:.4f})")
        print(f"  最快推理速度: {best_speed_model['model']} ({best_speed_model['avg_inference_time_ms']:.2f} ms)")


if __name__ == "__main__":
    from torch.utils.data import random_split
    main()
