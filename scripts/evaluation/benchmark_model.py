#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型基准测试脚本
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
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_best_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_model(model_path, model_name, num_classes, device):
    """加载模型"""
    weights = None
    if model_name == 'mobilenetv2':
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def benchmark_model(model, dataloader, device, class_names):
    """对模型进行基准测试"""
    print("\n" + "=" * 80)
    print("🔬 模型基准测试")
    print("=" * 80)

    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []

    print("\n📊 正在进行推理测试...")

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            start_time = time.time()
            outputs = model(inputs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.time() - start_time

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            inference_times.append(inference_time / inputs.size(0) * 1000)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean() * 100

    print("\n" + "-" * 40)
    print("📈 测试结果")
    print("-" * 40)
    print(f"  ✅ 整体准确率: {accuracy:.2f}%")
    print(f"  📊 样本数量: {len(all_labels)}")
    print(f"  🏷️ 类别数量: {len(class_names)}")

    print("\n" + "-" * 40)
    print("⏱️ 推理速度")
    print("-" * 40)
    print(f"  平均推理时间: {np.mean(inference_times):.2f} ms/图片")
    print(f"  最快推理时间: {np.min(inference_times):.2f} ms/图片")
    print(f"  最慢推理时间: {np.max(inference_times):.2f} ms/图片")
    print(f"  推理速度: {1000/np.mean(inference_times):.1f} 图片/秒")

    print("\n" + "-" * 40)
    print("🏆 Top-K 准确率")
    print("-" * 40)
    for k in [1, 3, 5]:
        if k <= len(class_names):
            top_k_acc = top_k_accuracy_score(all_labels, all_probs, k=k) * 100
            print(f"  Top-{k}: {top_k_acc:.2f}%")

    print("\n" + "-" * 40)
    print("📋 分类报告")
    print("-" * 40)

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)

    report_df = []
    for class_name in class_names:
        if class_name in report:
            report_df.append({
                'class': class_name,
                'precision': report[class_name]['precision'] * 100,
                'recall': report[class_name]['recall'] * 100,
                'f1': report[class_name]['f1-score'] * 100,
                'support': int(report[class_name]['support'])
            })

    report_df = sorted(report_df, key=lambda x: x['f1'], reverse=True)

    print("\n  🥇 表现最好的前10个类别:")
    print(f"  {'类别':<20} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'样本数':>10}")
    print("  " + "-" * 60)
    for item in report_df[:10]:
        print(f"  {item['class']:<20} {item['precision']:>9.1f}% {item['recall']:>9.1f}% {item['f1']:>9.1f}% {item['support']:>10}")

    print("\n  ⚠️ 表现最差的前10个类别:")
    print(f"  {'类别':<20} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'样本数':>10}")
    print("  " + "-" * 60)
    for item in report_df[-10:]:
        print(f"  {item['class']:<20} {item['precision']:>9.1f}% {item['recall']:>9.1f}% {item['f1']:>9.1f}% {item['support']:>10}")

    print("\n  📊 整体指标:")
    print(f"    平均精确率 (macro): {report['macro avg']['precision'] * 100:.2f}%")
    print(f"    平均召回率 (macro): {report['macro avg']['recall'] * 100:.2f}%")
    print(f"    平均F1分数 (macro): {report['macro avg']['f1-score'] * 100:.2f}%")

    print("\n" + "-" * 40)
    print("📉 混淆矩阵")
    print("-" * 40)

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) * 100

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(cm, annot=False, cmap='Blues', ax=axes[0], cbar_kws={'label': '样本数量'})
    axes[0].set_title('混淆矩阵 (原始值)')
    axes[0].set_xlabel('预测标签')
    axes[0].set_ylabel('真实标签')

    sns.heatmap(cm_normalized, annot=False, cmap='Blues', ax=axes[1], cbar_kws={'label': '百分比 (%)'})
    axes[1].set_title('混淆矩阵 (百分比)')
    axes[1].set_xlabel('预测标签')
    axes[1].set_ylabel('真实标签')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ 混淆矩阵已保存: confusion_matrix.png")

    results = {
        'accuracy': accuracy,
        'sample_count': len(all_labels),
        'class_count': len(class_names),
        'avg_inference_time_ms': float(np.mean(inference_times)),
        'throughput_fps': float(1000/np.mean(inference_times)),
        'top1_accuracy': float(top_k_accuracy_score(all_labels, all_probs, k=1) * 100),
        'top3_accuracy': float(top_k_accuracy_score(all_labels, all_probs, k=3) * 100) if len(class_names) >= 3 else None,
        'top5_accuracy': float(top_k_accuracy_score(all_labels, all_probs, k=5) * 100) if len(class_names) >= 5 else None,
        'macro_precision': float(report['macro avg']['precision'] * 100),
        'macro_recall': float(report['macro avg']['recall'] * 100),
        'macro_f1': float(report['macro avg']['f1-score'] * 100),
        'per_class_report': report_df,
        'confusion_matrix': cm.tolist()
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="模型基准测试")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径 (.pth)")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--model_name", type=str, default='efficientnet_b3', help="模型名称")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--image_size", type=int, default=224, help="图像大小")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="输出结果文件")

    args = parser.parse_args()

    device = get_best_device()
    print(f"\n🖥️ 使用设备: {device}")

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        dataset = datasets.ImageFolder(train_dir, transform=transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        class_names = dataset.classes
        num_classes = len(class_names)

        val_dataset = datasets.ImageFolder(val_dir, transform=transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    else:
        full_dataset = datasets.ImageFolder(args.data_dir, transform=transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        class_names = full_dataset.classes
        num_classes = len(class_names)
        val_dataset = full_dataset

    print(f"  📊 类别数量: {num_classes}")
    print(f"  🖼️ 验证集样本: {len(val_dataset)}")

    model = load_model(args.model_path, args.model_name, num_classes, device)
    model_size_mb = os.path.getsize(args.model_path) / (1024 * 1024)
    print(f"  💾 模型大小: {model_size_mb:.2f} MB")

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    results = benchmark_model(model, val_loader, device, class_names)
    results['model_size_mb'] = model_size_mb
    results['model_path'] = args.model_path
    results['data_dir'] = args.data_dir

    output_path = args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 基准测试结果已保存: {output_path}")

    print("\n" + "=" * 80)
    print("🎉 基准测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()