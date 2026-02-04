#!/usr/bin/env python3
"""
模型评估脚本
评估训练好的角色分类模型在测试集上的表现
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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evaluate_model')


class CharacterDataset(torch.utils.data.Dataset):
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
        
        logger.info(f"数据集初始化完成，包含 {len(classes)} 个类别，{len(self.images)} 张图像")
    
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


class CharacterClassifier(nn.Module):
    """角色分类器模型"""
    
    def __init__(self, num_classes):
        """初始化模型
        
        Args:
            num_classes: 类别数量
        """
        super().__init__()
        # 使用EfficientNet-B0作为基础模型
        from torchvision import models
        self.backbone = models.efficientnet_b0(pretrained=False)
        # 替换分类头
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, 
            num_classes
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def evaluate_model(args):
    """评估模型
    
    Args:
        args: 命令行参数
    """
    # 检测设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 加载验证集
    val_dataset = CharacterDataset(args.val_dir, transform=transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    num_classes = len(val_dataset.class_to_idx)
    model = CharacterClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 评估模型
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 生成分类报告
    class_names = list(val_dataset.class_to_idx.keys())
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        zero_division=0
    )
    
    logger.info("分类报告:\n" + report)
    print("分类报告:\n" + report)
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, 
                annot=False, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存混淆矩阵
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    logger.info(f"混淆矩阵已保存: {cm_path}")
    
    # 计算总体准确率
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    total = len(all_labels)
    accuracy = correct / total
    
    logger.info(f"总体准确率: {accuracy:.4f}")
    print(f"总体准确率: {accuracy:.4f}")
    
    return accuracy


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型评估工具')
    
    parser.add_argument('--model_path', type=str, default='models/character_classifier_best.pth', help='模型路径')
    parser.add_argument('--val_dir', type=str, default='data/split_dataset/val', help='验证集目录')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info('开始评估模型...')
    logger.info(f'模型路径: {args.model_path}')
    logger.info(f'验证集目录: {args.val_dir}')
    
    # 评估模型
    accuracy = evaluate_model(args)
    
    logger.info(f'模型评估完成！准确率: {accuracy:.4f}')


if __name__ == "__main__":
    main()
