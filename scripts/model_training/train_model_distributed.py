#!/usr/bin/env python3
"""
分布式模型训练脚本
支持多GPU并行训练，提高训练速度
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms, models
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_model_distributed')


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
        self.backbone = models.efficientnet_b0(pretrained=True)
        # 替换分类头
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, 
            num_classes
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def setup(rank, world_size):
    """初始化分布式训练环境
    
    Args:
        rank: 当前进程的排名
        world_size: 总进程数
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def train_model(rank, world_size, args):
    """训练模型
    
    Args:
        rank: 当前进程的排名
        world_size: 总进程数
        args: 命令行参数
    """
    # 设置分布式环境
    setup(rank, world_size)
    
    # 检测设备
    device = torch.device(rank)
    logger.info(f"进程 {rank} 使用设备: {device}")
    
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
    
    # 使用分布式采样器
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 初始化模型
    model = CharacterClassifier(num_classes=len(train_dataset.class_to_idx)).to(device)
    
    # 包装模型为DDP
    model = DDP(model, device_ids=[rank])
    
    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # 早停策略
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    # 训练循环
    for epoch in range(args.num_epochs):
        # 设置采样器的epoch，确保每轮 shuffle 不同
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f"开始第 {epoch+1}/{args.num_epochs} 轮训练")
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}', unit='batch', disable=rank != 0) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item() * inputs.size(0)
                if rank == 0:
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
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        # 收集所有进程的验证结果
        val_acc = val_correct / val_total
        val_epoch_loss = val_loss / len(val_loader.dataset)
        
        # 同步所有进程的验证准确率
        val_acc_tensor = torch.tensor(val_acc, device=device)
        dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.AVG)
        avg_val_acc = val_acc_tensor.item()
        
        if rank == 0:
            logger.info(f'Epoch {epoch+1}/{args.num_epochs}')
            logger.info(f'Train Loss: {epoch_loss:.4f}')
            logger.info(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {avg_val_acc:.4f}')
            logger.info(f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                patience_counter = 0
                model_path = os.path.join(args.output_dir, f'character_classifier_best_distributed.pth')
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_acc': avg_val_acc,
                    'class_to_idx': train_dataset.class_to_idx
                }, model_path)
                logger.info(f'最佳模型已保存: {model_path}, 验证准确率: {best_val_acc:.4f}')
            else:
                patience_counter += 1
                logger.info(f'早停计数器: {patience_counter}/{patience}')
                if patience_counter >= patience:
                    logger.info('早停条件触发，停止训练')
                    break
    
    # 保存最终模型
    if rank == 0:
        final_model_path = os.path.join(args.output_dir, 'character_classifier_final_distributed.pth')
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'class_to_idx': train_dataset.class_to_idx
        }, final_model_path)
        logger.info(f'最终模型已保存: {final_model_path}')
    
    cleanup()
    return best_val_acc


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分布式角色分类模型训练')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default='data/split_dataset/train', help='训练集目录')
    parser.add_argument('--val_dir', type=str, default='data/split_dataset/val', help='验证集目录')
    parser.add_argument('--output_dir', type=str, default='models', help='模型输出目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='批量大小（每个GPU）')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检测可用的GPU数量
    world_size = torch.cuda.device_count()
    if world_size < 2:
        logger.warning(f"只检测到 {world_size} 个GPU，分布式训练需要至少2个GPU")
        logger.warning("将使用单GPU模式训练")
        
        # 如果只有一个GPU，使用单GPU模式
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        # 加载数据集
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
        model = CharacterClassifier(num_classes=len(train_dataset.class_to_idx)).to(device)
        
        # 损失函数与优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # 早停策略
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        
        # 训练循环
        for epoch in range(args.num_epochs):
            logger.info(f"开始第 {epoch+1}/{args.num_epochs} 轮训练")
            
            # 训练阶段
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
                    scheduler.step()
                    
                    running_loss += loss.item() * inputs.size(0)
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
            logger.info(f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                model_path = os.path.join(args.output_dir, f'character_classifier_best_distributed.pth')
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
                patience_counter += 1
                logger.info(f'早停计数器: {patience_counter}/{patience}')
                if patience_counter >= patience:
                    logger.info('早停条件触发，停止训练')
                    break
        
        # 保存最终模型
        final_model_path = os.path.join(args.output_dir, 'character_classifier_final_distributed.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_to_idx': train_dataset.class_to_idx
        }, final_model_path)
        logger.info(f'最终模型已保存: {final_model_path}')
        logger.info(f'训练完成！最佳验证准确率: {best_val_acc:.4f}')
    else:
        # 使用多GPU分布式训练
        logger.info(f"检测到 {world_size} 个GPU，将使用分布式训练")
        logger.info(f'批量大小: {args.batch_size} (每个GPU)')
        logger.info(f'总批量大小: {args.batch_size * world_size}')
        logger.info(f'训练轮数: {args.num_epochs}')
        logger.info(f'学习率: {args.learning_rate}')
        
        # 启动多进程分布式训练
        mp.spawn(train_model, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
