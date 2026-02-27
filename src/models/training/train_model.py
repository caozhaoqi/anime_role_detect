#!/usr/bin/env python3
"""
模型训练脚本
在MacBook Air M4上训练角色分类模型，提升识别效果
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
logger = logging.getLogger('train_model')


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
            # 返回一个默认的空白图片
            image = Image.new('RGB', (224, 224), color='white')
        
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
        # 使用EfficientNet-B3作为基础模型，适合动漫角色识别任务
        # 输入分辨率300x300，能够更好地保留动漫角色的细节特征
        self.backbone = models.efficientnet_b3(pretrained=True)
        
        # 初始阶段：冻结所有骨干网络参数，只训练分类头
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        
        # 替换分类头，添加更多层以增强表达能力
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(
                self.backbone.classifier[1].in_features, 
                1024
            ),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, num_classes)
        )
    
    def unfreeze_backbone(self):
        """
        解冻骨干网络，准备进行全网微调
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def train_model(args):
    """训练模型
    
    Args:
        args: 命令行参数
    """
    # Mixup数据增强函数
    def mixup_data(x, y, alpha=0.2):
        """
        Mixup数据增强
        
        Args:
            x: 输入数据
            y: 标签
            alpha: Beta分布的alpha参数
            
        Returns:
            mixed_x: 混合后的数据
            y_a, y_b: 混合的标签
            lam: 混合比例
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """
        Mixup损失函数
        
        Args:
            criterion: 原始损失函数
            pred: 模型预测
            y_a, y_b: 混合的标签
            lam: 混合比例
            
        Returns:
            混合后的损失
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    # 检测设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据增强与预处理 - 增强版
    train_transform = transforms.Compose([
        transforms.Resize((330, 330)),
        transforms.RandomCrop((300, 300)),
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.RandomAffine(
            degrees=15, 
            translate=(0.15, 0.15), 
            scale=(0.8, 1.2),
            shear=10
        ),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
        transforms.GaussianBlur(kernel_size=(7, 11), sigma=(0.1, 6)),
        transforms.ColorJitter(
            brightness=0.5, 
            contrast=0.5, 
            saturation=0.5, 
            hue=0.2
        ),
        transforms.RandomGrayscale(p=0.4),
        transforms.RandomSolarize(threshold=192.0, p=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
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
    train_dataset = CharacterDataset(args.train_dir, transform=train_transform)
    val_dataset = CharacterDataset(args.val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    # 初始化模型
    model = CharacterClassifier(num_classes=len(train_dataset.class_to_idx)).to(device)
    
    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器 - 使用StepLR，在特定轮次降低学习率
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,
        gamma=0.1
    )
    
    # 训练循环
    best_val_acc = 0.0
    patience = 15  # 增加早停耐心值
    no_improve_epochs = 0
    unfreeze_epoch = 15  # 提前到第15轮开始解冻骨干网络
    
    for epoch in range(args.num_epochs):
        logger.info(f"开始第 {epoch+1}/{args.num_epochs} 轮训练")
        
        # 两阶段训练策略
        if epoch == unfreeze_epoch:
            logger.info(f"第 {epoch+1} 轮：解冻骨干网络，开始全网微调")
            model.unfreeze_backbone()
            # 解冻后降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate * 0.1
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}', unit='batch') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 应用Mixup数据增强
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.1)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                pbar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        val_epoch_loss = val_loss / len(val_loader.dataset)
        
        logger.info(f'Epoch {epoch+1}/{args.num_epochs}')
        logger.info(f'Train Loss: {epoch_loss:.4f}')
        logger.info(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0  # 重置早停计数器
            model_path = os.path.join(args.output_dir, f'character_classifier_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'class_to_idx': train_dataset.class_to_idx
            }, model_path)
            logger.info(f'最佳模型已保存: {model_path}, 验证准确率: {best_val_acc:.4f}')
        else:
            no_improve_epochs += 1
            logger.info(f'早停计数器: {no_improve_epochs}/{patience}')
            
        # 早停检查
        if no_improve_epochs >= patience:
            logger.info(f'早停: 连续 {patience} 轮验证准确率无提升')
            break
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'character_classifier_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx
    }, final_model_path)
    logger.info(f'最终模型已保存: {final_model_path}')
    
    return best_val_acc


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='在MacBook Air M4上训练角色分类模型')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default='data/split_dataset/train', help='训练集目录')
    parser.add_argument('--val_dir', type=str, default='data/split_dataset/val', help='验证集目录')
    parser.add_argument('--output_dir', type=str, default='models', help='模型输出目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小 - 减小以适应更大的模型')
    parser.add_argument('--num_epochs', type=int, default=80, help='训练轮数 - 增加以充分训练更复杂的模型')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='学习率 - 适当增加初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减 - 增加以减少过拟合')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数 - 增加以提高数据加载速度')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info('开始训练模型...')
    logger.info(f'训练集目录: {args.train_dir}')
    logger.info(f'验证集目录: {args.val_dir}')
    logger.info(f'批量大小: {args.batch_size}')
    logger.info(f'训练轮数: {args.num_epochs}')
    logger.info(f'学习率: {args.learning_rate}')
    
    # 开始训练
    best_acc = train_model(args)
    
    logger.info(f'训练完成！最佳验证准确率: {best_acc:.4f}')


if __name__ == "__main__":
    main()
