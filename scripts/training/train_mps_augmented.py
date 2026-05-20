#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版训练脚本 - 增加更多数据增强策略
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.transforms import v2

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_best_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available():
        logger.info("✅ MPS设备可用")
        return torch.device('mps')
    elif torch.cuda.is_available():
        logger.info("✅ CUDA设备可用")
        return torch.device('cuda')
    else:
        logger.info("⚠️ 仅CPU可用")
        return torch.device('cpu')


def load_notification_config():
    """加载飞书通知配置"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notification_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            notification_config = json.load(f)
        os.environ['NOTIFICATION_ENABLED'] = 'true'
        os.environ['NOTIFICATION_PLATFORM'] = notification_config['platform']
        os.environ['FEISHU_APP_ID'] = notification_config['feishu']['app_id']
        os.environ['FEISHU_APP_SECRET'] = notification_config['feishu']['app_secret']
        os.environ['FEISHU_RECEIVE_ID'] = notification_config['feishu']['receive_id']
        os.environ['FEISHU_RECEIVE_ID_TYPE'] = notification_config['feishu']['receive_id_type']
        logger.info(f"✅ 已加载通知配置: {config_path}")
    else:
        logger.warning(f"⚠️ 未找到通知配置文件: {config_path}")


def send_feishu_message(title, message):
    """发送飞书消息"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.services.notification_service import send_notification
        success = send_notification(f"**{title}**\n\n{message}", level="info")
        if success:
            logger.info("✅ 飞书通知发送成功")
            return True
        else:
            logger.warning("❌ 飞书通知发送失败（通知服务返回失败）")
            return False
    except Exception as e:
        logger.warning(f"❌ 发送飞书通知失败: {e}")
        return False


def get_augmented_transforms(image_size=224, augment_level='high'):
    """
    获取增强版数据变换
    
    Args:
        image_size: 图像大小
        augment_level: 增强级别 ('low', 'medium', 'high')
    
    Returns:
        train_transform, val_transform
    """
    # 基础变换
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 增强变换 - 根据级别选择不同的增强策略
    if augment_level == 'low':
        # 轻度增强 - 适合数据质量较高的情况
        train_transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif augment_level == 'medium':
        # 中度增强 - 平衡数据多样性和质量
        train_transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.2), int(image_size * 1.2))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif augment_level == 'high':
        # 高度增强 - 最大化数据多样性（使用 v2 API）
        train_transform = transforms.Compose([
            # 随机大小裁剪
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            
            # 几何变换
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            
            # 颜色变换
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.15
            ),
            
            # 随机高斯模糊
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            
            # 转换为张量并归一化
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # 随机擦除
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        ])
    
    else:
        train_transform = base_transform
    
    return train_transform, base_transform


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=30, patience=5):
    """训练模型"""
    best_model_wts = None
    best_acc = 0.0
    early_stop_counter = 0
    train_history = []
    best_loss = float('inf')
    
    logger.info(f"🚀 开始训练，共 {num_epochs} 轮")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info('-' * 10)
        
        # 每个 epoch 都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # 迭代数据
            batch_count = len(dataloaders[phase])
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 梯度归零
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    # 反向传播 + 优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                        # 打印进度
                        if batch_idx % 50 == 0:
                            logger.info(f"  Batch {batch_idx}/{batch_count}, Loss: {loss.item():.4f}")
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            
            logger.info(f"  {phase} Loss: {epoch_loss:.4f}, {phase} Acc: {epoch_acc:.4%}")
            
            # 记录历史
            train_history.append({
                'epoch': epoch + 1,
                'phase': phase,
                'loss': epoch_loss,
                'accuracy': epoch_acc.item()
            })
            
            # 更新学习率（基于训练轮次）
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            # 深度复制最佳模型（基于验证准确率）
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = model.state_dict().copy()
                early_stop_counter = 0
                
                # 发送进度通知
                send_feishu_message(
                    f"📈 训练进度更新 (Epoch {epoch + 1})",
                    f"训练准确率: {100 * (running_corrects.float() / len(dataloaders['train'].dataset)):.2f}%\n"
                    f"验证准确率: {epoch_acc * 100:.2f}%\n"
                    f"当前最佳: {best_acc * 100:.2f}%"
                )
            elif phase == 'val':
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logger.info(f"⚠️ 早停触发，已连续 {patience} 轮无提升")
                    break
        
        if early_stop_counter >= patience:
            break
    
    logger.info(f"\n🎉 训练完成，最佳验证准确率: {best_acc:.4%}, 最佳验证损失: {best_loss:.4f}")
    
    # 加载最佳模型权重
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    return model, best_acc.item(), best_loss, train_history


def main():
    parser = argparse.ArgumentParser(description="增强版模型训练")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--image_size", type=int, default=224, help="图像大小")
    parser.add_argument("--augment_level", type=str, default='high', 
                        choices=['low', 'medium', 'high'], help="数据增强级别")
    parser.add_argument("--patience", type=int, default=8, help="早停耐心值")
    parser.add_argument("--model_name", type=str, default='efficientnet_b0', 
                        choices=['mobilenetv2', 'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b4', 'resnet18', 'resnet50'], 
                        help="模型名称")
    parser.add_argument("--use_pretrained", action='store_true', default=True, help="使用预训练权重")
    parser.add_argument("--freeze_backbone", action='store_true', default=False, help="冻结主干网络")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="标签平滑系数")
    
    args = parser.parse_args()
    
    # 加载通知配置
    load_notification_config()
    
    # 获取设备
    device = get_best_device()
    
    # 获取数据变换
    logger.info(f"📸 使用数据增强级别: {args.augment_level}")
    train_transform, val_transform = get_augmented_transforms(args.image_size, args.augment_level)
    
    # 加载数据集
    logger.info(f"📂 加载数据: {args.data_dir}")
    
    # 检查目录结构
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        # 使用预分割的数据集
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes
    else:
        # 使用单目录数据集，自动分割
        full_dataset = datasets.ImageFolder(args.data_dir, transform=train_transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # 分割数据集
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # 应用验证集变换
        val_dataset.dataset.transform = val_transform
        
        num_classes = len(full_dataset.classes)
        class_names = full_dataset.classes
    
    logger.info(f"📋 类别数量: {num_classes}")
    logger.info(f"📊 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    }
    
    # 创建模型（带预训练权重）
    logger.info(f"🧠 使用模型: {args.model_name}, 预训练: {args.use_pretrained}")
    
    weights = 'IMAGENET1K_V1' if args.use_pretrained else None
    
    if args.model_name == 'mobilenetv2':
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif args.model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif args.model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif args.model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif args.model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model = model.to(device)
    
    # 如果冻结主干网络
    if args.freeze_backbone:
        logger.info("❄️ 冻结主干网络，仅训练分类器")
        if args.model_name.startswith('efficientnet'):
            for param in model.features.parameters():
                param.requires_grad = False
        elif args.model_name.startswith('resnet'):
            for param in model.parameters():
                param.requires_grad = False
            model.fc.requires_grad = True
        elif args.model_name == 'mobilenetv2':
            for param in model.features.parameters():
                param.requires_grad = False
    
    # 定义损失函数（带标签平滑）和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # 分层学习率
    if args.freeze_backbone:
        # 仅训练分类器
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        # 分层学习率：主干网络使用较小学习率
        if args.model_name.startswith('efficientnet'):
            optimizer = optim.Adam([
                {'params': model.features.parameters(), 'lr': args.lr * 0.1},
                {'params': model.classifier.parameters(), 'lr': args.lr}
            ], weight_decay=1e-4)
        elif args.model_name.startswith('resnet'):
            optimizer = optim.Adam([
                {'params': model.parameters()[:-1], 'lr': args.lr * 0.1},
                {'params': model.fc.parameters(), 'lr': args.lr}
            ], weight_decay=1e-4)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 训练模型
    model, best_acc, best_loss, train_history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, 
        device, args.epochs, args.patience
    )
    
    # 保存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pretrained_tag = 'pretrained' if args.use_pretrained else 'scratch'
    model_dir = f"./models/{args.model_name}_loli_{num_classes}_{pretrained_tag}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'model_best.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"✅ 模型已保存: {model_path}")
    
    # 保存完整模型（用于后续微调）
    full_model_path = os.path.join(model_dir, 'model_full.pth')
    torch.save(model, full_model_path)
    logger.info(f"✅ 完整模型已保存: {full_model_path}")
    
    # 保存训练结果
    results = {
        'model_name': args.model_name,
        'num_classes': num_classes,
        'class_names': class_names,
        'best_accuracy': best_acc,
        'best_loss': best_loss,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'image_size': args.image_size,
        'augment_level': args.augment_level,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'use_pretrained': args.use_pretrained,
        'freeze_backbone': args.freeze_backbone,
        'label_smoothing': args.label_smoothing,
        'train_history': train_history,
        'timestamp': timestamp
    }
    
    results_path = os.path.join(model_dir, 'training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✅ 训练结果已保存")
    
    # 发送完成通知
    send_feishu_message(
        f"🏆 训练完成",
        f"模型: {args.model_name}\n"
        f"类别数: {num_classes}\n"
        f"准确率: {best_acc * 100:.2f}%\n"
        f"预训练: {args.use_pretrained}\n"
        f"增强级别: {args.augment_level}\n"
        f"保存路径: {model_dir}"
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("🎬 训练任务结束")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()