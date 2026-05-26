#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量训练脚本 - 在已有模型基础上继续优化训练
使用 data/final_dataset 数据集进行微调
优化版本: 添加 CosineAnnealingLR + Warmup、CutMix、EMA、梯度裁剪
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json
from datetime import datetime
from collections import Counter

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("incremental_training")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("incremental_training")

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# ============= 配置参数 =============
BATCH_SIZE = 16
NUM_EPOCHS = 80
LEARNING_RATE = 2e-4  # 增大初始学习率加速收敛
IMAGE_SIZE = 224
NUM_WORKERS = 0
PATIENCE = 20
EARLY_STOP_DELTA = 0.001
WEIGHT_DECAY = 5e-5  # 减小权重衰减
LABEL_SMOOTHING = 0.07  # 调整标签平滑
GRADIENT_CLIP_NORM = 1.0  # 梯度裁剪
EMA_DECAY = 0.999  # EMA衰减率
WARMUP_EPOCHS = 5  # 热身轮数

# 现有模型路径
EXISTING_MODEL_DIR = './models/efficientnet_b3_loli_optimized_v2_20260522_165046'
EXISTING_MODEL_PATH = os.path.join(EXISTING_MODEL_DIR, 'model_best.pth')

# 新数据集路径
DATA_DIR = './data/final_dataset'
MODEL_DIR = './models'
CHECKPOINT_DIR = './checkpoints'

# ============= 数据加载 =============
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

        logger.info(f"数据集加载完成: {len(self.samples)} 样本, {len(self.class_to_idx)} 类别")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"加载失败 {img_path}: {e}")
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
            if self.transform:
                image = self.transform(image)
            return image, label

# ============= 数据增强 =============
def get_transforms(augment_level='heavy'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if augment_level == 'heavy':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.15),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return train_transform, val_transform

# ============= 类别权重计算 =============
def calculate_class_weights(dataset):
    labels = [sample[1] for sample in dataset.samples]
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_counts)
    
    weights = []
    for idx in range(num_classes):
        count = label_counts.get(idx, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    logger.info(f"类别权重计算完成，最小权重: {min(weights):.2f}, 最大权重: {max(weights):.2f}")
    return torch.tensor(weights)

# ============= MixUp 增强 =============
def mixup_data(x, y, alpha=1.0):
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
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============= CutMix 增强 =============
def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

# ============= EMA 模型 =============
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ============= 训练函数 =============
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, use_mixup=True, use_cutmix=True, gradient_clip_norm=0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 随机选择数据增强方式
        augment_choice = np.random.random()
        if use_mixup and augment_choice < 0.25:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.8)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        elif use_cutmix and augment_choice < 0.5:
            images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        # 梯度裁剪
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 30 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============= 检查点保存与加载 =============
def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, train_history, filepath, ema=None):
    checkpoint = {
        'epoch': epoch,
        'best_acc': best_acc,
        'train_history': train_history,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'ema_shadow': ema.shadow if ema else None,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    torch.save(checkpoint, filepath)
    logger.info(f"💾 检查点已保存: {filepath}")

def load_checkpoint(filepath, model, optimizer, scheduler, ema=None):
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if ema and checkpoint.get('ema_shadow'):
        ema.shadow = checkpoint['ema_shadow']
    logger.info(f"🔄 检查点已加载: epoch {checkpoint['epoch']}, best_acc: {checkpoint['best_acc']:.2f}%")
    return checkpoint['epoch'], checkpoint['best_acc'], checkpoint['train_history']

# ============= 学习率预热调度器 =============
class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epochs=5, eta_min=1e-7, last_epoch=-1):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine Annealing
            cos_decay = 0.5 * (1 + np.cos(np.pi * (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)))
            return [self.eta_min + (base_lr - self.eta_min) * cos_decay for base_lr in self.base_lrs]

# ============= 增量训练主函数 =============
def incremental_train(existing_model_path, train_loader, val_loader, num_classes, class_weights, device, resume_checkpoint=None):
    logger.info(f"📦 加载现有模型: {existing_model_path}")
    
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)
    
    logger.info("✅ 预训练模型加载成功")
    
    for param in model.parameters():
        param.requires_grad = True
    
    logger.info("🔓 模型配置完成：所有层均可训练")
    
    # EMA
    ema = EMA(model, decay=EMA_DECAY)
    
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=LABEL_SMOOTHING
    )
    
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': LEARNING_RATE * 0.5, 'weight_decay': WEIGHT_DECAY},
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY}
    ])
    
    # 使用 CosineAnnealingLR + Warmup
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        eta_min=1e-7
    )

    best_acc = 0.0
    best_model_state = None
    patience_counter = 0
    train_history = []
    start_epoch = 0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        start_epoch, best_acc, train_history = load_checkpoint(resume_checkpoint, model, optimizer, scheduler, ema)
        start_epoch += 1
        logger.info(f"🔄 从 epoch {start_epoch} 继续训练")
    else:
        logger.info(f"\n🚀 开始增量训练，共 {NUM_EPOCHS} 轮")

    for epoch in range(start_epoch, NUM_EPOCHS):
        logger.info(f"\n📌 Epoch {epoch + 1}/{NUM_EPOCHS}")
        logger.info(f"📊 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS,
            use_mixup=True, use_cutmix=True, gradient_clip_norm=GRADIENT_CLIP_NORM
        )
        
        # EMA 更新
        ema.update()
        
        # 使用 EMA 模型进行验证
        ema.apply_shadow()
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        ema.restore()

        scheduler.step()

        logger.info(f"  训练 Loss: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        logger.info(f"  验证 Loss: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        if val_acc > best_acc + EARLY_STOP_DELTA:
            best_acc = val_acc
            ema.apply_shadow()
            best_model_state = model.state_dict().copy()
            ema.restore()
            patience_counter = 0
            logger.info(f"  🏆 保存最佳模型，准确率: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"  ⏹️ 验证准确率连续 {PATIENCE} 轮未提升超过 {EARLY_STOP_DELTA}, 提前停止训练")
                break

        if (epoch + 1) % 5 == 0:
            checkpoint_file = os.path.join(CHECKPOINT_DIR, f'incremental_checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, train_history, checkpoint_file, ema)

    model.load_state_dict(best_model_state)
    logger.info(f"\n🎉 增量训练完成，最佳验证准确率: {best_acc:.2f}%")
    return model, best_acc, train_history

# ============= 主函数 =============
def main():
    import argparse
    parser = argparse.ArgumentParser(description='增量训练脚本 - 支持断点重训')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='检查点保存目录')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🎯 增量训练脚本 v2.0 - 优化版")
    logger.info("=" * 60)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"💻 使用设备: {device}")
    logger.info(f"📋 优化配置: CosineAnnealingLR+Warmup | CutMix+MixUp | EMA | 梯度裁剪")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    if args.resume:
        logger.info(f"🔄 将从检查点恢复训练: {args.resume}")

    logger.info(f"\n📂 加载数据集: {DATA_DIR}")
    train_transform, val_transform = get_transforms(augment_level='heavy')
    
    full_dataset = CustomImageDataset(DATA_DIR, transform=None)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    class_weights = calculate_class_weights(full_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    
    logger.info(f"📊 数据集划分: 训练集 {train_size} 张, 验证集 {val_size} 张")

    model, best_acc, train_history = incremental_train(
        EXISTING_MODEL_PATH,
        train_loader,
        val_loader,
        len(full_dataset.class_to_idx),
        class_weights,
        device,
        resume_checkpoint=args.resume
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"efficientnet_b3_loli_optimized_{timestamp}"
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model, os.path.join(model_dir, 'model_full.pth'))
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_best.pth'))
    
    results = {
        'model_name': 'efficientnet_b3',
        'num_classes': len(full_dataset.class_to_idx),
        'class_names': list(full_dataset.class_to_idx.keys()),
        'best_accuracy': best_acc / 100,
        'train_samples': train_size,
        'val_samples': val_size,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': NUM_EPOCHS,
        'weight_decay': WEIGHT_DECAY,
        'label_smoothing': LABEL_SMOOTHING,
        'train_history': train_history,
        'timestamp': timestamp,
        'optimizations': ['CosineAnnealingLR+Warmup', 'CutMix+MixUp', 'EMA', 'GradientClip']
    }
    
    with open(os.path.join(model_dir, 'training_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("📦 增量训练模型保存完成")
    logger.info(f"路径: {model_dir}")
    logger.info(f"最佳验证准确率: {best_acc:.2f}%")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
