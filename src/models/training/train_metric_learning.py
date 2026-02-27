#!/usr/bin/env python3
"""
基于度量学习的角色识别模型训练
目标：提升模型泛化能力，使其能够识别训练时没有的角色
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
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_metric_learning')


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
                    img_path = os.path.join(cls_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(idx)
        
        logger.info(f"数据集初始化完成，包含 {len(self.class_to_idx)} 个类别，{len(self.images)} 张图像")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"加载图像失败: {img_path}, 错误: {e}")
            # 返回一个空白图像作为替代
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MetricLearningModel(nn.Module):
    """度量学习模型"""
    
    def __init__(self, embedding_dim=512, num_classes=None):
        """初始化模型
        
        Args:
            embedding_dim: 特征嵌入维度
            num_classes: 类别数量（用于ArcFace Loss）
        """
        super(MetricLearningModel, self).__init__()
        
        # 使用EfficientNet作为骨干网络
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # 移除最后的分类层，使用倒数第二层的输出
        # EfficientNet的倒数第二层是一个1280维的特征向量
        self.backbone.classifier = nn.Identity()
        
        # 添加一个投影层，将特征映射到指定维度
        self.projection = nn.Sequential(
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 如果提供了类别数量，添加ArcFace分类器
        self.num_classes = num_classes
        if num_classes:
            self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入图像
            
        Returns:
            embedding: 特征嵌入
            logits: 分类logits（如果num_classes不为None）
        """
        # 提取特征
        features = self.backbone(x)
        
        # 生成嵌入
        embedding = self.projection(features)
        
        # 归一化嵌入
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        # 如果需要分类
        if self.num_classes:
            logits = self.fc(embedding)
            return embedding, logits
        
        return embedding
    
    def extract_feature(self, x):
        """提取特征
        
        Args:
            x: 输入图像
            
        Returns:
            embedding: 特征嵌入
        """
        self.eval()
        with torch.no_grad():
            embedding = self.forward(x)
        return embedding


class ArcFaceLoss(nn.Module):
    """ArcFace Loss"""
    
    def __init__(self, margin=0.5, scale=64.0):
        """初始化
        
        Args:
            margin: 角度边界
            scale: 特征缩放因子
        """
        super(ArcFaceLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, embedding, logits, labels):
        """前向传播
        
        Args:
            embedding: 特征嵌入
            logits: 分类logits
            labels: 真实标签
            
        Returns:
            loss: ArcFace损失
        """
        # 计算余弦相似度
        cos_theta = nn.functional.cosine_similarity(embedding, nn.functional.normalize(logits, dim=1), dim=1)
        
        # 计算ArcFace损失
        theta = torch.acos(cos_theta)
        target_logits = torch.cos(theta + self.margin)
        
        # 构建新的logits
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        modified_logits = self.scale * (one_hot * target_logits.unsqueeze(1) + (1 - one_hot) * logits)
        
        return self.ce_loss(modified_logits, labels)


def train_model(args):
    """训练模型
    
    Args:
        args: 训练参数
        
    Returns:
        best_acc: 最佳验证准确率
    """
    # 检测设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 增强的数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),
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
    train_dataset = CharacterDataset(args.train_dir, transform=train_transform)
    val_dataset = CharacterDataset(args.val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 初始化模型
    num_classes = len(train_dataset.class_to_idx)
    model = MetricLearningModel(
        embedding_dim=args.embedding_dim,
        num_classes=num_classes
    ).to(device)
    
    # 初始化损失函数和优化器
    arcface_loss = ArcFaceLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # 训练参数
    best_acc = 0.0
    patience = 10
    patience_counter = 0
    
    # 开始训练
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{args.num_epochs}') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # 前向传播
                embedding, logits = model(inputs)
                
                # 计算损失
                loss = arcface_loss(embedding, logits, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                embedding, logits = model(inputs)
                loss = arcface_loss(embedding, logits, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        val_epoch_loss = val_loss / len(val_loader.dataset)
        
        logger.info(f'Epoch {epoch+1}/{args.num_epochs}')
        logger.info(f'Train Loss: {epoch_loss:.4f}')
        logger.info(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_acc:.4f}')
        logger.info(f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0  # 重置早停计数器
            model_path = os.path.join(args.output_dir, f'character_metric_model_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'class_to_idx': train_dataset.class_to_idx,
                'embedding_dim': args.embedding_dim
            }, model_path)
            logger.info(f'最佳模型已保存: {model_path}, 验证准确率: {best_acc:.4f}')
        else:
            patience_counter += 1
            logger.info(f'早停计数器: {patience_counter}/{patience}')
            if patience_counter >= patience:
                logger.info('早停条件触发，停止训练')
                break
        
        # 更新学习率
        scheduler.step()
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'character_metric_model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
        'embedding_dim': args.embedding_dim
    }, final_model_path)
    logger.info(f'最终模型已保存: {final_model_path}')
    
    return best_acc


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于度量学习的角色识别模型训练')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default='data/train', help='训练集目录')
    parser.add_argument('--val_dir', type=str, default='data/split_dataset/val', help='验证集目录')
    parser.add_argument('--output_dir', type=str, default='models', help='模型输出目录')
    
    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=512, help='特征嵌入维度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info('开始训练度量学习模型...')
    logger.info(f'训练集目录: {args.train_dir}')
    logger.info(f'验证集目录: {args.val_dir}')
    logger.info(f'特征嵌入维度: {args.embedding_dim}')
    logger.info(f'批量大小: {args.batch_size}')
    logger.info(f'训练轮数: {args.num_epochs}')
    logger.info(f'学习率: {args.learning_rate}')
    
    # 开始训练
    best_acc = train_model(args)
    
    logger.info(f'训练完成！最佳验证准确率: {best_acc:.4f}')


if __name__ == "__main__":
    main()
