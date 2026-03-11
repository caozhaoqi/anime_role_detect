#!/usr/bin/env python3
"""
增量训练模型基准测试
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder


def get_model(model_type, num_classes):
    """获取模型"""
    if model_type == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )
    elif model_type == 'resnet50':
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='增量训练模型基准测试')
    parser.add_argument('--model-path', type=str, default='models/incremental/model_best.pth',
                        help='模型文件路径')
    parser.add_argument('--model-type', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'efficientnet_b0', 'resnet50'],
                        help='模型类型')
    parser.add_argument('--data-dir', type=str, default='data/downloaded_images',
                        help='数据目录')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--output-dir', type=str, default='models/benchmark',
                        help='输出目录')
    return parser.parse_args()


def load_model(model_path, model_type, device):
    """加载模型"""
    print(f"加载模型: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    class_to_idx = checkpoint.get('class_to_idx', {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    model = get_model(model_type, num_classes)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"模型加载完成，类别数: {num_classes}")
    print(f"类别映射: {class_to_idx}")
    
    return model, class_to_idx, idx_to_class


def get_data_loader(data_dir, batch_size, num_workers):
    """获取数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset.classes, dataset.class_to_idx


def benchmark_model(model, dataloader, device, idx_to_class):
    """基准测试模型"""
    print("\n开始基准测试...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    total_time = 0
    num_samples = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            
            total_time += inference_time
            num_samples += images.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"进度: {batch_idx + 1}/{len(dataloader)}")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = (all_preds == all_labels).mean()
    
    num_classes = len(idx_to_class)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_scores = np.zeros(num_classes)
    
    for i in range(num_classes):
        true_positive = ((all_preds == i) & (all_labels == i)).sum()
        false_positive = ((all_preds == i) & (all_labels != i)).sum()
        false_negative = ((all_preds != i) & (all_labels == i)).sum()
        
        precision[i] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall[i] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_scores[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    avg_precision = precision.mean()
    avg_recall = recall.mean()
    avg_f1 = f1_scores.mean()
    
    fps = num_samples / total_time
    
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion_matrix[label][pred] += 1
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(avg_precision),
        'recall': float(avg_recall),
        'f1_score': float(avg_f1),
        'fps': float(fps),
        'total_time': float(total_time),
        'num_samples': int(num_samples),
        'per_class_metrics': {
            idx_to_class[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1_scores[i]),
                'support': int((all_labels == i).sum())
            }
            for i in range(num_classes)
        },
        'confusion_matrix': confusion_matrix.tolist()
    }
    
    return results


def print_results(results):
    """打印结果"""
    print("\n" + "="*60)
    print("基准测试结果")
    print("="*60)
    
    print(f"\n整体指标:")
    print(f"  准确率 (Accuracy): {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  精确率 (Precision): {results['precision']:.4f}")
    print(f"  召回率 (Recall): {results['recall']:.4f}")
    print(f"  F1分数 (F1-Score): {results['f1_score']:.4f}")
    print(f"  推理速度 (FPS): {results['fps']:.2f}")
    print(f"  总推理时间: {results['total_time']:.2f}秒")
    print(f"  测试样本数: {results['num_samples']}")
    
    print(f"\n各类别指标:")
    print("-"*60)
    print(f"{'类别':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数':<10}")
    print("-"*60)
    
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"{class_name:<10} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['support']:<10}")
    
    print("-"*60)
    
    print(f"\n混淆矩阵:")
    print(np.array(results['confusion_matrix']))


def save_results(results, output_dir):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")


def main():
    args = parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model, class_to_idx, idx_to_class = load_model(args.model_path, args.model_type, device)
    
    dataloader, classes, dataset_class_to_idx = get_data_loader(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"数据集: {len(dataloader.dataset)} 张图片, {len(classes)} 个类别")
    
    results = benchmark_model(model, dataloader, device, idx_to_class)
    
    print_results(results)
    
    save_results(results, args.output_dir)


if __name__ == '__main__':
    main()
