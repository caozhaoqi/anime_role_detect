#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本

评估模型的性能和准确率
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model_training.train_model import CharacterDataset, CharacterClassifier

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evaluate_model')


def evaluate_model(model, data_loader, device, class_to_idx):
    """
    评估模型
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        class_to_idx: 类别到索引的映射
        
    Returns:
        预测结果和真实标签
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), idx_to_class


def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别列表
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"混淆矩阵已保存到 {save_path}")


def plot_per_class_accuracy(y_true, y_pred, classes, save_path='per_class_accuracy.png'):
    """
    绘制每个类别的准确率
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别列表
        save_path: 保存路径
    """
    per_class_acc = {}
    
    for i, cls in enumerate(classes):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).sum() / mask.sum()
            per_class_acc[cls] = acc
    
    # 排序
    per_class_acc = dict(sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True))
    
    plt.figure(figsize=(12, 6))
    plt.barh(list(per_class_acc.keys()), list(per_class_acc.values()))
    plt.xlabel('Accuracy')
    plt.ylabel('Class')
    plt.title('Per-Class Accuracy')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"每类别准确率图已保存到 {save_path}")
    
    return per_class_acc


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估模型性能')
    
    parser.add_argument('--model_path', type=str, default='models/character_classifier_best.pth', 
                       help='模型路径')
    parser.add_argument('--data_dir', type=str, default='data/split_dataset/val', 
                       help='数据目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检测设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据预处理
    val_transform = transforms.Compose([
        transforms.Resize((330, 330)),
        transforms.CenterCrop((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 加载数据集
    val_dataset = CharacterDataset(args.data_dir, transform=val_transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    logger.info(f"数据集加载完成，包含 {len(val_dataset.class_to_idx)} 个类别，{len(val_dataset)} 张图像")
    
    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    num_classes = len(checkpoint['class_to_idx'])
    
    model = CharacterClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"模型加载完成: {args.model_path}")
    
    # 评估模型
    logger.info("开始评估模型...")
    preds, labels, idx_to_class = evaluate_model(
        model, val_loader, device, checkpoint['class_to_idx']
    )
    
    # 计算准确率
    accuracy = accuracy_score(labels, preds)
    logger.info(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 生成分类报告
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(labels, preds, target_names=classes, output_dict=True)
    
    # 保存分类报告
    import json
    report_path = os.path.join(args.output_dir, 'classification_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"分类报告已保存到 {report_path}")
    
    # 打印分类报告
    logger.info("\n分类报告:")
    print(classification_report(labels, preds, target_names=classes))
    
    # 绘制混淆矩阵
    confusion_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(labels, preds, classes, confusion_matrix_path)
    
    # 绘制每类别准确率
    per_class_acc_path = os.path.join(args.output_dir, 'per_class_accuracy.png')
    per_class_acc = plot_per_class_accuracy(labels, preds, classes, per_class_acc_path)
    
    # 保存评估结果
    results = {
        'model_path': args.model_path,
        'data_dir': args.data_dir,
        'total_samples': len(val_dataset),
        'num_classes': len(classes),
        'accuracy': float(accuracy),
        'per_class_accuracy': {k: float(v) for k, v in per_class_acc.items()},
        'classification_report': report
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存到 {results_path}")
    
    logger.info("模型评估完成！")


if __name__ == '__main__':
    main()
