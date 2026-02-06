#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合评估脚本

评估训练好的模型在测试集上的性能，包括准确率、推理速度、模型大小等指标。
"""

import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import logging
from tqdm import tqdm
import json
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('comprehensive_evaluation')

class CharacterDataset(Dataset):
    """角色数据集类"""
    
    def __init__(self, root_dir, transform=None):
        """初始化数据集
        
        Args:
            root_dir: 数据目录
            transform: 数据变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # 构建类别映射
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            
            # 遍历图像
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)
        
        logger.info(f"测试数据集初始化完成，包含 {len(classes)} 个类别，{len(self.images)} 张图像")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_model(model_type, num_classes):
    """获取模型
    
    Args:
        model_type: 模型类型
        num_classes: 类别数量
    
    Returns:
        模型
    """
    if model_type == 'efficientnet_b0':
        logger.info("加载模型: EfficientNet-B0")
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'mobilenet_v2':
        logger.info("加载模型: MobileNetV2")
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'shufflenet_v2':
        logger.info("加载模型: ShuffleNetV2")
        model = models.shufflenet_v2_x1_0(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'squeezenet':
        logger.info("加载模型: SqueezeNet")
        model = models.squeezenet1_0(pretrained=False)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif model_type == 'resnet18':
        logger.info("加载模型: ResNet18")
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model

def calculate_model_size(model):
    """计算模型大小
    
    Args:
        model: 模型
    
    Returns:
        模型大小（MB）, 参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    model_size = total_params * 4 / (1024 * 1024)  # 每个参数4字节
    return model_size, total_params

def evaluate_model_performance(model, data_loader, device):
    """评估模型性能
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
    
    Returns:
        准确率, 平均推理时间（毫秒）, FPS
    """
    model.eval()
    correct = 0
    total = 0
    total_inference_time = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="评估模型性能"):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播，记录时间
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            inference_time = end_time - start_time
            total_inference_time += inference_time
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (correct / total) * 100
    avg_inference_time = (total_inference_time / total) * 1000  # 转换为毫秒
    fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
    
    return accuracy, avg_inference_time, fps

def evaluate_class_performance(model, data_loader, device, class_names):
    """评估每个类别的性能
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        class_names: 类别名称列表
    
    Returns:
        每个类别的准确率字典
    """
    model.eval()
    
    # 初始化混淆矩阵
    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # 计算每个类别的准确率
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    # 计算每个类别的准确率
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            accuracy = (class_correct[i] / class_total[i]) * 100
            class_accuracies[class_name] = accuracy
        else:
            class_accuracies[class_name] = 0.0
    
    return class_accuracies

def main():
    parser = argparse.ArgumentParser(description='综合评估脚本')
    
    parser.add_argument('--test-data-dir', type=str, default='data/train',
                       help='测试数据目录')
    parser.add_argument('--model-dir', type=str, default='models/augmented_training/mobilenet_v2',
                       help='模型目录')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载器工作线程数')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据集
    test_dataset = CharacterDataset(root_dir=args.test_data_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    num_classes = len(test_dataset.class_to_idx)
    class_names = list(test_dataset.class_to_idx.keys())
    
    # 加载模型配置
    model_config_path = os.path.join(args.model_dir, 'class_to_idx.json')
    if not os.path.exists(model_config_path):
        logger.error(f"模型配置文件不存在: {model_config_path}")
        return
    
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_class_to_idx = json.load(f)
    
    # 加载最佳模型
    best_model_path = os.path.join(args.model_dir, 'model_best.pth')
    if not os.path.exists(best_model_path):
        logger.error(f"最佳模型文件不存在: {best_model_path}")
        return
    
    # 确定模型类型
    model_type = os.path.basename(args.model_dir)
    logger.info(f"评估模型: {model_type}")
    
    # 加载模型
    model = get_model(model_type, num_classes)
    model.to(device)
    
    # 加载模型权重
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"加载模型权重成功，最佳验证准确率: {checkpoint['val_accuracy']:.2f}%")
    
    # 计算模型大小
    model_size, num_params = calculate_model_size(model)
    logger.info(f"模型大小: {model_size:.2f} MB, 参数数量: {num_params:,}")
    
    # 评估模型性能
    logger.info("\n开始评估模型性能...")
    accuracy, avg_inference_time, fps = evaluate_model_performance(model, test_loader, device)
    logger.info(f"测试准确率: {accuracy:.2f}%")
    logger.info(f"平均推理时间: {avg_inference_time:.2f} 毫秒")
    logger.info(f"推理速度: {fps:.2f} FPS")
    
    # 评估每个类别的性能
    logger.info("\n开始评估每个类别的性能...")
    class_accuracies = evaluate_class_performance(model, test_loader, device, class_names)
    
    # 打印每个类别的准确率
    logger.info("\n每个类别的准确率:")
    for class_name, acc in sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{class_name}: {acc:.2f}%")
    
    # 计算平均类别准确率
    avg_class_accuracy = sum(class_accuracies.values()) / len(class_accuracies)
    logger.info(f"\n平均类别准确率: {avg_class_accuracy:.2f}%")
    
    # 保存评估结果
    evaluation_results = {
        'model_type': model_type,
        'model_size_mb': model_size,
        'num_params': num_params,
        'test_accuracy': accuracy,
        'avg_inference_time_ms': avg_inference_time,
        'fps': fps,
        'avg_class_accuracy': avg_class_accuracy,
        'class_accuracies': class_accuracies,
        'test_data_size': len(test_dataset),
        'num_classes': num_classes,
        'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = os.path.join(args.model_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n评估结果已保存到: {results_path}")
    
    # 打印综合评估报告
    logger.info("\n=== 综合评估报告 ===")
    logger.info(f"模型类型: {model_type}")
    logger.info(f"模型大小: {model_size:.2f} MB")
    logger.info(f"参数数量: {num_params:,}")
    logger.info(f"测试准确率: {accuracy:.2f}%")
    logger.info(f"平均推理时间: {avg_inference_time:.2f} 毫秒")
    logger.info(f"推理速度: {fps:.2f} FPS")
    logger.info(f"平均类别准确率: {avg_class_accuracy:.2f}%")
    logger.info(f"测试数据大小: {len(test_dataset)} 张图像")
    logger.info(f"类别数量: {num_classes}")
    logger.info("===================")

if __name__ == '__main__':
    main()
