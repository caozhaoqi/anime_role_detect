#!/usr/bin/env python3
"""
知识蒸馏训练脚本
使用ResNet18作为教师模型，MobileNetV3作为学生模型
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
logger = logging.getLogger('knowledge_distillation')


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


class ResNetCharacterClassifier(nn.Module):
    """使用ResNet18的角色分类器模型"""
    
    def __init__(self, num_classes):
        """初始化模型
        
        Args:
            num_classes: 类别数量
        """
        super().__init__()
        # 使用ResNet18作为基础模型
        self.backbone = models.resnet18(pretrained=True)
        
        # 替换分类头
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


class SimpleCharacterClassifier(nn.Module):
    """使用MobileNetV3的简单分类器模型"""
    
    def __init__(self, num_classes):
        """初始化模型
        
        Args:
            num_classes: 类别数量
        """
        super().__init__()
        # 使用MobileNetV3 Small作为基础模型
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        
        # 替换分类头
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def distillation_loss(student_outputs, teacher_outputs, labels, temperature=2.0, alpha=0.7):
    """计算知识蒸馏损失
    
    Args:
        student_outputs: 学生模型输出
        teacher_outputs: 教师模型输出
        labels: 真实标签
        temperature: 温度参数
        alpha: 损失权重
        
    Returns:
        总损失
    """
    # 软标签损失（知识蒸馏损失）
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_outputs / temperature, dim=1),
        nn.functional.softmax(teacher_outputs / temperature, dim=1)
    ) * (temperature * temperature)
    
    # 硬标签损失（传统交叉熵损失）
    hard_loss = nn.CrossEntropyLoss()(student_outputs, labels)
    
    # 总损失
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return total_loss

def train_with_distillation(args):
    """使用知识蒸馏训练模型
    
    Args:
        args: 命令行参数
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.4, 
            contrast=0.4, 
            saturation=0.4,
            hue=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
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
    
    # 加载教师模型
    logger.info(f"加载教师模型: {args.teacher_model_path}")
    teacher_checkpoint = torch.load(args.teacher_model_path, map_location=device)
    teacher_model = ResNetCharacterClassifier(num_classes=len(train_dataset.class_to_idx)).to(device)
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model.eval()  # 教师模型只用于推理，不训练
    
    # 初始化学生模型
    student_model = SimpleCharacterClassifier(num_classes=len(train_dataset.class_to_idx)).to(device)
    
    # 损失函数与优化器
    optimizer = optim.AdamW(
        student_model.parameters(), 
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
        student_model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}', unit='batch') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 教师模型输出
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                
                # 学生模型输出
                student_outputs = student_model(inputs)
                
                # 计算知识蒸馏损失
                loss = distillation_loss(
                    student_outputs, 
                    teacher_outputs, 
                    labels, 
                    temperature=args.temperature, 
                    alpha=args.alpha
                )
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                pbar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 验证
        student_model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student_model(inputs)
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
            model_path = os.path.join(args.output_dir, 'character_classifier_distilled_best.pth')
            torch.save({
                'model_state_dict': student_model.state_dict(),
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
    parser = argparse.ArgumentParser(description='知识蒸馏训练脚本')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default='data/augmented_dataset', help='训练集目录')
    parser.add_argument('--val_dir', type=str, default='data/split_dataset/val', help='验证集目录')
    parser.add_argument('--output_dir', type=str, default='models_distilled', help='模型输出目录')
    parser.add_argument('--teacher_model_path', type=str, 
                       default='models_resnet/character_classifier_resnet_best.pth', 
                       help='教师模型路径')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # 蒸馏参数
    parser.add_argument('--temperature', type=float, default=2.0, help='温度参数')
    parser.add_argument('--alpha', type=float, default=0.7, help='蒸馏损失权重')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info('开始知识蒸馏训练...')
    logger.info(f'训练集目录: {args.train_dir}')
    logger.info(f'验证集目录: {args.val_dir}')
    logger.info(f'教师模型: {args.teacher_model_path}')
    logger.info(f'批量大小: {args.batch_size}')
    logger.info(f'训练轮数: {args.num_epochs}')
    logger.info(f'学习率: {args.learning_rate}')
    logger.info(f'温度参数: {args.temperature}')
    logger.info(f'蒸馏损失权重: {args.alpha}')
    
    # 开始训练
    best_acc = train_with_distillation(args)
    
    logger.info(f'知识蒸馏训练完成！最佳验证准确率: {best_acc:.4f}')


if __name__ == "__main__":
    main()
