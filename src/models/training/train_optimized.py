#!/usr/bin/env python3
"""
优化模型训练脚本
针对小数据集进行优化
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import copy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_optimized')


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


class OptimizedCharacterClassifier(nn.Module):
    """优化的角色分类器模型 - 使用EfficientNet-B0"""
    
    def __init__(self, num_classes):
        """初始化模型
        
        Args:
            num_classes: 类别数量
        """
        super().__init__()
        # 使用EfficientNet-B0作为基础模型，更适合小数据集
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # 初始阶段：冻结所有骨干网络参数
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        
        # 替换分类头
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(
                self.backbone.classifier[1].in_features, 
                512
            ),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(256, num_classes)
        )
    
    def unfreeze_backbone(self):
        """解冻骨干网络"""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def mixup_data(x, y, alpha=0.2):
    """Mixup数据增强"""
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
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_single_fold(model, train_loader, val_loader, device, args, fold_idx):
    """训练单个fold
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        args: 训练参数
        fold_idx: fold索引
        
    Returns:
        best_val_acc: 最佳验证准确率
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    best_val_acc = 0.0
    patience = 20
    no_improve_epochs = 0
    unfreeze_epoch = 10
    
    for epoch in range(args.num_epochs):
        # 两阶段训练
        if epoch == unfreeze_epoch:
            logger.info(f"Fold {fold_idx+1}, Epoch {epoch+1}: 解冻骨干网络")
            model.unfreeze_backbone()
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate * 0.1
        
        # 训练
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
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
        
        logger.info(f"Fold {fold_idx+1}, Epoch {epoch+1}/{args.num_epochs}, "
                   f"Train Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            
            model_path = os.path.join(args.output_dir, f'fold_{fold_idx+1}_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'fold': fold_idx
            }, model_path)
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= patience:
            logger.info(f"Fold {fold_idx+1}: 早停")
            break
    
    return best_val_acc


def train_with_cross_validation(args):
    """使用交叉验证训练模型
    
    Args:
        args: 训练参数
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据增强 - 增强版
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),
        transforms.RandomAffine(
            degrees=20, 
            translate=(0.2, 0.2), 
            scale=(0.7, 1.3),
            shear=15
        ),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 8)),
        transforms.ColorJitter(
            brightness=0.6, 
            contrast=0.6, 
            saturation=0.6, 
            hue=0.3
        ),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomSolarize(threshold=160.0, p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=3, p=0.4),
        transforms.RandomAutocontrast(p=0.4),
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
    full_dataset = CharacterDataset(args.data_dir, transform=None)
    
    # K折交叉验证
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(full_dataset.images)):
        logger.info(f"\n{'='*50}")
        logger.info(f"开始训练 Fold {fold_idx+1}/{args.n_folds}")
        logger.info(f"{'='*50}")
        
        # 创建训练和验证数据集
        train_dataset = copy.deepcopy(full_dataset)
        train_dataset.images = [full_dataset.images[i] for i in train_indices]
        train_dataset.labels = [full_dataset.labels[i] for i in train_indices]
        train_dataset.transform = train_transform
        
        val_dataset = copy.deepcopy(full_dataset)
        val_dataset.images = [full_dataset.images[i] for i in val_indices]
        val_dataset.labels = [full_dataset.labels[i] for i in val_indices]
        val_dataset.transform = val_transform
        
        # 创建数据加载器
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
        model = OptimizedCharacterClassifier(num_classes=len(full_dataset.class_to_idx)).to(device)
        
        # 训练
        best_val_acc = train_single_fold(
            model, train_loader, val_loader, device, args, fold_idx
        )
        
        fold_results.append(best_val_acc)
        logger.info(f"Fold {fold_idx+1} 完成，最佳验证准确率: {best_val_acc:.4f}")
    
    # 计算平均准确率
    avg_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"交叉验证完成")
    logger.info(f"平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"各Fold准确率: {[f'{acc:.4f}' for acc in fold_results]}")
    logger.info(f"{'='*50}")
    
    # 保存结果
    results = {
        'avg_accuracy': float(avg_acc),
        'std_accuracy': float(std_acc),
        'fold_accuracies': [float(acc) for acc in fold_results],
        'n_folds': args.n_folds
    }
    
    results_path = os.path.join(args.output_dir, 'cross_validation_results.json')
    import json
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存到 {results_path}")
    
    return avg_acc


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='优化模型训练 - 针对小数据集')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/split_dataset/train', 
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, default='models_optimized', 
                       help='模型输出目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # 交叉验证参数
    parser.add_argument('--n_folds', type=int, default=5, help='交叉验证折数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info('开始优化训练...')
    logger.info(f'数据目录: {args.data_dir}')
    logger.info(f'批量大小: {args.batch_size}')
    logger.info(f'训练轮数: {args.num_epochs}')
    logger.info(f'学习率: {args.learning_rate}')
    logger.info(f'交叉验证折数: {args.n_folds}')
    
    # 开始训练
    avg_acc = train_with_cross_validation(args)
    
    logger.info(f'优化训练完成！平均准确率: {avg_acc:.4f}')


if __name__ == "__main__":
    main()
