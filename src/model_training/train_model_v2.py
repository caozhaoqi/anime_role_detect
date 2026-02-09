#!/usr/bin/env python3
"""
模型训练脚本 v2
使用增强后的数据集进行训练，增加训练轮数，优化训练策略
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import logging
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_model_v2')


class CharacterClassifier(nn.Module):
    """角色分类器模型"""
    
    def __init__(self, num_classes):
        """初始化模型
        
        Args:
            num_classes: 类别数量
        """
        super().__init__()
        # 使用EfficientNet-B0作为基础模型
        self.backbone = models.efficientnet_b0(pretrained=False)
        # 替换分类头
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, 
            num_classes
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def train_model(args):
    """训练模型
    
    Args:
        args: 命令行参数
    """
    # 检测设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.1
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 加载数据集
    # 首先检查并清理空目录
    def clean_empty_dirs(root_dir):
        """清理空目录"""
        for root, dirs, files in os.walk(root_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    logger.info(f"删除空目录: {dir_path}")
    
    # 清理训练和验证目录中的空目录
    clean_empty_dirs('data/split_dataset_v2/train')
    clean_empty_dirs('data/split_dataset_v2/val')
    
    # 加载数据集
    train_dataset = datasets.ImageFolder(
        root='data/split_dataset_v2/train',
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root='data/split_dataset_v2/val',
        transform=val_transform
    )
    
    # 数据加载器
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
    
    # 类别信息
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    logger.info(f"数据集加载完成！")
    logger.info(f"训练集大小: {len(train_dataset)} 张图像")
    logger.info(f"验证集大小: {len(val_dataset)} 张图像")
    logger.info(f"类别数量: {num_classes}")
    logger.info(f"类别列表: {list(class_to_idx.keys())}")
    
    # 初始化模型
    model = CharacterClassifier(num_classes=num_classes).to(device)
    
    # 加载预训练权重（如果有）
    if args.pretrained_model:
        try:
            checkpoint = torch.load(args.pretrained_model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"加载预训练模型成功: {args.pretrained_model}")
        except Exception as e:
            logger.error(f"加载预训练模型失败: {str(e)}")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    
    # 学习率调度器
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # 第一个周期的长度
        T_mult=2,  # 每个后续周期的长度倍数
        eta_min=1e-6  # 最小学习率
    )
    
    # 最佳模型参数
    best_val_acc = 0.0
    best_model_state = None
    early_stopping_counter = 0
    
    # 开始训练
    logger.info(f"开始训练模型...")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"初始学习率: {args.learning_rate}")
    
    for epoch in range(args.epochs):
        logger.info(f"开始第 {epoch+1}/{args.epochs} 轮训练")
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"训练轮次 {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 更新学习率
            scheduler.step(epoch + (correct / len(train_loader.dataset)))
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算训练指标
        train_loss = running_loss / total
        train_acc = correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"验证轮次 {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 计算验证指标
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # 当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"Current LR: {current_lr:.6f}")
        
        # 检查是否是最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            early_stopping_counter = 0
            
            # 保存最佳模型
            checkpoint = {
                'model_state_dict': best_model_state,
                'class_to_idx': class_to_idx,
                'val_acc': best_val_acc,
                'epoch': epoch
            }
            torch.save(checkpoint, 'models/character_classifier_best_v2.pth')
            logger.info(f"最佳模型已保存: models/character_classifier_best_v2.pth, 验证准确率: {best_val_acc:.4f}")
        else:
            early_stopping_counter += 1
            logger.info(f"早停计数器: {early_stopping_counter}/{args.early_stopping_patience}")
            
            # 检查是否早停
            if early_stopping_counter >= args.early_stopping_patience:
                logger.info(f"早停触发！验证准确率在 {args.early_stopping_patience} 轮内没有提升")
                break
    
    # 保存最终模型
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'val_acc': val_acc,
        'epoch': args.epochs
    }
    torch.save(final_checkpoint, 'models/character_classifier_final_v2.pth')
    logger.info(f"最终模型已保存: models/character_classifier_final_v2.pth")
    
    logger.info(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练角色分类模型 v2')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--pretrained_model', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='早停 patience')
    args = parser.parse_args()
    
    # 确保模型目录存在
    os.makedirs('models', exist_ok=True)
    
    logger.info('开始训练模型 v2...')
    logger.info(f'训练轮数: {args.epochs}')
    logger.info(f'批次大小: {args.batch_size}')
    logger.info(f'学习率: {args.learning_rate}')
    logger.info(f'早停 patience: {args.early_stopping_patience}')
    
    train_model(args)
    
    logger.info('模型训练完成！')


if __name__ == "__main__":
    main()
