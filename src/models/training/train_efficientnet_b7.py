#!/usr/bin/env python3
"""
使用EfficientNet-B7的模型训练脚本
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_efficientnet_b7')


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
        
        logger.info(f"数据集初始化完成，包含 {len(classes)} 个类别，{len(self.images)} 张图像")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"无法加载图片 {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class EfficientNetB7CharacterClassifier(nn.Module):
    """使用EfficientNet-B7的角色分类器模型"""
    
    def __init__(self, num_classes):
        """初始化模型
        
        Args:
            num_classes: 类别数量
        """
        super().__init__()
        # 使用EfficientNet-B7作为基础模型
        self.backbone = models.efficientnet_b7(pretrained=True)
        
        # 冻结前几层，只训练后面的层
        for i, param in enumerate(self.backbone.parameters()):
            if i < 80:  # 冻结前80层
                param.requires_grad = False
        
        # 替换分类头
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def train_model(args):
    """训练模型
    
    Args:
        args: 命令行参数
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomCrop((380, 380)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(45),
        transforms.ColorJitter(
            brightness=0.4, 
            contrast=0.4, 
            saturation=0.4,
            hue=0.2
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.CenterCrop((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 加载数据集
    train_dataset = CharacterDataset(args.train_dir, transform=train_transform)
    val_dataset = CharacterDataset(args.val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # 初始化模型
    model = EfficientNetB7CharacterClassifier(num_classes=len(train_dataset.class_to_idx)).to(device)
    
    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # 训练循环
    best_val_acc = 0.0
    patience = 30
    no_improve_epochs = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"开始第 {epoch+1}/{args.num_epochs} 轮训练")
        
        # 训练
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}', unit='batch') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                pbar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        logger.info(f'Epoch {epoch+1}/{args.num_epochs}')
        logger.info(f'Train Loss: {epoch_loss:.4f}')
        logger.info(f'Val Acc: {val_acc:.4f}')
        
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            model_path = os.path.join(args.output_dir, 'character_classifier_efficientnet_b7_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': train_dataset.class_to_idx,
                'val_acc': best_val_acc
            }, model_path)
            logger.info(f'最佳模型已保存: {model_path}, 验证准确率: {best_val_acc:.4f}')
        else:
            no_improve_epochs += 1
            logger.info(f'早停计数器: {no_improve_epochs}/{patience}')
            
        # 早停检查
        if no_improve_epochs >= patience:
            logger.info(f'早停: 连续 {patience} 轮验证准确率无提升')
            break
    
    return best_val_acc


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EfficientNet-B7模型训练')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default='data/augmented_dataset', help='训练集目录')
    parser.add_argument('--val_dir', type=str, default='data/split_dataset/val', help='验证集目录')
    parser.add_argument('--output_dir', type=str, default='models_efficientnet_b7', help='模型输出目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info('开始EfficientNet-B7模型训练...')
    logger.info(f'训练集目录: {args.train_dir}')
    logger.info(f'验证集目录: {args.val_dir}')
    logger.info(f'批量大小: {args.batch_size}')
    logger.info(f'训练轮数: {args.num_epochs}')
    logger.info(f'学习率: {args.learning_rate}')
    
    # 开始训练
    best_acc = train_model(args)
    
    logger.info(f'EfficientNet-B7模型训练完成！最佳验证准确率: {best_acc:.4f}')


if __name__ == "__main__":
    main()
