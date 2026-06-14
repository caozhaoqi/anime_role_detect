#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 评估当前保存的模型效果
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from pathlib import Path
from tqdm import tqdm

# 配置
DATA_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")
MODEL_NAME = "MobileNetV2"

# 图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_device():
    """获取最佳设备"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model(num_classes, device):
    """加载模型"""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model_path = MODEL_DIR / f"{MODEL_NAME.lower()}_best.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ 加载模型: {model_path}")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    return model.to(device)


def test_model(model, dataloader, device, dataset_sizes):
    """测试模型"""
    model.eval()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="测试中"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            total_correct += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            total_loss += loss.item() * inputs.size(0)
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / dataset_sizes
    
    return accuracy, avg_loss


def main():
    print("=" * 70)
    print(f"🎯 快速测试: {MODEL_NAME}")
    print("=" * 70)
    
    device = get_device()
    print(f"📱 使用设备: {device}")
    
    # 加载数据集
    print("\n📦 加载数据集...")
    dataset = ImageFolder(DATA_DIR, transform=transform)
    test_size = int(0.2 * len(dataset))
    _, test_dataset = random_split(dataset, [len(dataset) - test_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"  测试集样本数: {len(test_dataset)}")
    print(f"  类别数: {len(dataset.classes)}")
    
    # 加载模型
    model = load_model(len(dataset.classes), device)
    if not model:
        return
    
    # 测试
    print("\n🔍 开始测试...")
    accuracy, loss = test_model(model, test_loader, device, len(test_dataset))
    
    # 输出结果
    print("\n" + "=" * 70)
    print("📊 测试结果")
    print("=" * 70)
    print(f"模型: {MODEL_NAME}")
    print(f"测试准确率: {accuracy * 100:.2f}%")
    print(f"测试损失: {loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
