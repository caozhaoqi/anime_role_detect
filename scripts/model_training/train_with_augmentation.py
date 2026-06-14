#!/usr/bin/env python3
"""
带数据增强的训练脚本
增强策略:
- RandomResizedCrop
- ColorJitter
- RandomErasing
- HorizontalFlip
- Mixup
- CutMix
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import json
import gc

# 数据增强配置
TRAIN_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")
LOG_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/logs")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mixup_data(x, y, alpha=1.0):
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


class CutMixCollator:
    """CutMix数据增强"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # 获取裁剪区域
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
        # 调整lambda值
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        return images, labels, labels[index], lam
    
    def rand_bbox(self, size, lam):
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


def get_transforms(augment=True):
    """获取数据变换（修正了 RandomErasing 的位置以防 PIL 类型错误）"""
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),  # RandomErasing 前必须先将图像转为 Tensor [1]
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))  # 确保在已归一化的 Tensor 上擦除
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def train_model_with_augmentation(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                                   num_epochs=30, device='cpu', mixup_alpha=0.5,
                                   start_epoch=0, best_acc=0.0, history=None):
    """带增强的训练"""
    if history is None:
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="训练"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Mixup
            if np.random.random() < 0.5 and mixup_alpha > 0:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                preds = torch.max(outputs, 1)[1]
                running_corrects += (lam * preds.eq(labels_a.data).cpu().sum().float() + 
                                    (1 - lam) * preds.eq(labels_b.data).cpu().sum().float()).item()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).float().item()
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        
        print(f"训练 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="验证"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).float().item()
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects / len(val_loader.dataset)
        
        print(f"验证 Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # 更新学习率
        scheduler.step(epoch_val_loss)
        
        # 保存最佳模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_DIR / "mobilenetv2_aug_best.pth")
            print(f"✅ 保存最佳模型 (Acc: {best_acc:.4f})")
        
        gc.collect()
    
    print(f"\n训练完成！最佳准确率: {best_acc:.4f}")
    return model, history


def main(resume=False):
    print("=" * 60)
    print("🚀 带数据增强的模型训练")
    print("=" * 60)
    
    device = get_device()
    print(f"📱 设备: {device}")
    
    # 加载数据集并配置变换
    print("\n📦 加载数据集...")
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    # 建立两个独立的 Dataset 实例以彻底实现训练集和验证集的变换隔离 [2]
    train_dataset_full = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_dataset_full = datasets.ImageFolder(str(TRAIN_DIR), transform=val_transform)
    
    # 利用 randperm 随机划分不重叠的索引，确保两个子集无数据交叉，且各自的数据变换完全独立 [2]
    num_samples = len(train_dataset_full)
    indices = torch.randperm(num_samples).tolist()
    train_size = int(0.8 * num_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"类别数: {len(train_dataset_full.classes)}")
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 创建模型
    print("\n🏗️ 创建模型...")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_dataset_full.classes))
    model = model.to(device)
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 检查是否需要恢复训练
    start_epoch = 0
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_path = MODEL_DIR / "mobilenetv2_aug_best.pth"
    history_path = LOG_DIR / "training_history_aug.json"
    
    if resume and best_model_path.exists():
        print(f"🔄 从已保存的最佳模型恢复训练...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
            start_epoch = len(history['train_acc'])
            if history['val_acc']:
                best_acc = max(history['val_acc'])
            print(f"  - 已完成 Epoch: {start_epoch}")
            print(f"  - 最佳准确率: {best_acc:.4f}")
    
    # 训练
    print("\n🔥 开始训练...")
    model, history = train_model_with_augmentation(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=30, device=device, mixup_alpha=0.5,
        start_epoch=start_epoch, best_acc=best_acc, history=history
    )
    
    # 保存历史记录
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / "training_history_aug.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ 训练完成！")
    print(f"📁 最佳模型: {MODEL_DIR / 'mobilenetv2_aug_best.pth'}")
    print(f"📊 训练历史: {LOG_DIR / 'training_history_aug.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='带数据增强的模型训练')
    parser.add_argument('--resume', action='store_true', help='从上次保存的最佳模型继续训练')
    args = parser.parse_args()
    main(resume=args.resume)