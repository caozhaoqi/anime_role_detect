#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用关键点信息训练模型
集成关键点标注到模型训练中，提高分类性能
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import logging
from tqdm import tqdm
import json
import numpy as np
import random

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from arona.training.models.keypoint_aware_model import get_keypoint_aware_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_with_keypoints')


class KeypointCharacterDataset(Dataset):
    def __init__(self, root_dir, keypoint_annotations_dir, transform=None, target_characters=None):
        self.root_dir = root_dir
        self.keypoint_annotations_dir = keypoint_annotations_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.keypoints = []
        self.class_to_idx = {}
        
        all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        if target_characters:
            classes = [c for c in all_classes if any(tc in c for tc in target_characters)]
        else:
            classes = all_classes
        
        idx = 0
        for cls in classes:
            self.class_to_idx[cls] = idx
            cls_dir = os.path.join(root_dir, cls)
            
            # 加载该类的关键点标注
            annotation_file = os.path.join(keypoint_annotations_dir, f"{cls}_keypoints.json")
            annotations = {}
            if os.path.exists(annotation_file):
                try:
                    with open(annotation_file, 'r', encoding='utf-8') as f:
                        annotations = json.load(f)
                except Exception as e:
                    logger.warning(f"加载标注文件 {annotation_file} 失败: {e}")
            
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    img_path = os.path.join(cls_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(idx)
                    
                    # 获取关键点信息
                    if img_name in annotations:
                        self.keypoints.append(annotations[img_name]['keypoints'])
                    else:
                        # 如果没有关键点信息，使用空字典
                        self.keypoints.append({})
            
            logger.info(f"角色 {cls}: {len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])} 张图像")
            idx += 1
        
        logger.info(f"数据集初始化完成，包含 {len(self.class_to_idx)} 个类别，{len(self.images)} 张图像")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        keypoint_data = self.keypoints[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # 处理关键点数据
        processed_keypoints = self._process_keypoint_data(keypoint_data)
        
        return image, label, processed_keypoints
    
    def _process_keypoint_data(self, keypoint_data):
        """处理关键点数据，转换为模型所需的格式"""
        # 提取面部关键点
        face_points = []
        if 'face' in keypoint_data and keypoint_data['face'] is not None:
            for face in keypoint_data['face']:
                if 'keypoints' in face:
                    for point in face['keypoints']:
                        try:
                            x = float(point.get('x', 0))
                            y = float(point.get('y', 0))
                            face_points.append([x, y])
                        except (ValueError, TypeError):
                            pass
        
        # 提取手部关键点
        hands_points = []
        if 'hands' in keypoint_data and keypoint_data['hands'] is not None:
            for hand in keypoint_data['hands']:
                if 'keypoints' in hand:
                    for point in hand['keypoints']:
                        try:
                            x = float(point.get('x', 0))
                            y = float(point.get('y', 0))
                            hands_points.append([x, y])
                        except (ValueError, TypeError):
                            pass
        
        # 提取姿态关键点
        pose_points = []
        if 'pose' in keypoint_data and keypoint_data['pose'] is not None:
            for point in keypoint_data['pose']:
                try:
                    # 检查point是否是字典
                    if isinstance(point, dict):
                        x = float(point.get('x', 0))
                        y = float(point.get('y', 0))
                        pose_points.append([x, y])
                except (ValueError, TypeError):
                    pass
        
        return {
            'face': face_points,
            'hands': hands_points,
            'pose': pose_points
        }


def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def collate_fn(batch):
    """自定义collate函数，处理不同长度的关键点数据"""
    images, labels, keypoints_list = zip(*batch)
    
    # 转换为张量
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    # 处理关键点数据
    # 这里我们将关键点数据转换为固定长度的张量
    # 假设每个类型的关键点最多有17个点，每个点有2个坐标
    max_points = 17
    
    face_tensors = []
    hands_tensors = []
    pose_tensors = []
    
    for keypoints in keypoints_list:
        # 处理面部关键点
        face = keypoints.get('face', [])
        face_tensor = torch.zeros(max_points * 2)
        for i, point in enumerate(face[:max_points]):
            face_tensor[i*2] = point[0]
            face_tensor[i*2+1] = point[1]
        face_tensors.append(face_tensor)
        
        # 处理手部关键点
        hands = keypoints.get('hands', [])
        hands_tensor = torch.zeros(max_points * 2)
        for i, point in enumerate(hands[:max_points]):
            hands_tensor[i*2] = point[0]
            hands_tensor[i*2+1] = point[1]
        hands_tensors.append(hands_tensor)
        
        # 处理姿态关键点
        pose = keypoints.get('pose', [])
        pose_tensor = torch.zeros(max_points * 2)
        for i, point in enumerate(pose[:max_points]):
            pose_tensor[i*2] = point[0]
            pose_tensor[i*2+1] = point[1]
        pose_tensors.append(pose_tensor)
    
    # 转换为批次张量
    face_tensors = torch.stack(face_tensors)
    hands_tensors = torch.stack(hands_tensors)
    pose_tensors = torch.stack(pose_tensors)
    
    # 组合关键点数据
    keypoints_tensor = torch.cat([face_tensors, hands_tensors, pose_tensors], dim=1)
    
    return images, labels, keypoints_tensor


def train_model(model, train_loader, val_loader, device, num_epochs=80, lr=0.0005, weight_decay=0.0002, 
                output_dir='models/keypoint_aware', class_to_idx=None, use_mixup=True, label_smoothing=0.1):
    logger.info(f"开始训练，设备: {device}")
    logger.info(f"训练轮数: {num_epochs}, 学习率: {lr}, 权重衰减: {weight_decay}")
    logger.info(f"数据增强: Mixup={use_mixup}, 标签平滑={label_smoothing}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
    )
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    
    patience = 25
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels, keypoints in pbar:
            images, labels, keypoints = images.to(device), labels.to(device), keypoints.to(device)
            
            if use_mixup and random.random() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)
                optimizer.zero_grad()
                outputs = model(images, keypoints)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(images, keypoints)
                loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
        
        scheduler.step()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, keypoints in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images, labels, keypoints = images.to(device), labels.to(device), keypoints.to(device)
                outputs = model(images, keypoints)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}: '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                   f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                   f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': class_to_idx
            }, os.path.join(output_dir, 'model_best.pth'))
            logger.info(f'保存最佳模型，验证准确率: {val_acc:.2f}%')
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            logger.info(f'早停: {patience} 个epoch没有改进')
            break
    
    results = {
        'model_type': 'keypoint_aware',
        'num_epochs': num_epochs,
        'best_val_accuracy': best_val_acc,
        'final_train_accuracy': train_accuracies[-1],
        'final_val_accuracy': val_accuracies[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'use_mixup': use_mixup,
        'label_smoothing': label_smoothing
    }
    
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f'训练完成，最佳验证准确率: {best_val_acc:.2f}%')
    return results


def main():
    parser = argparse.ArgumentParser(description='使用关键点信息训练模型')
    parser.add_argument('--data-dir', type=str, default='../../data/train', help='数据目录')
    parser.add_argument('--keypoint-annotations-dir', type=str, default='../../data/keypoint_annotations', help='关键点标注目录')
    parser.add_argument('--model-type', type=str, default='efficientnet_b3', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'efficientnet_b3', 'resnet50'],
                       help='模型类型')
    parser.add_argument('--batch-size', type=int, default=24, help='批量大小')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0002, help='权重衰减')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='标签平滑系数')
    parser.add_argument('--use-mixup', action='store_true', help='使用Mixup数据增强')
    parser.add_argument('--output-dir', type=str, default='../../models/keypoint_aware', help='输出目录')
    
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomCrop((288, 288)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.CenterCrop((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_characters = None
    
    logger.info('加载数据集...')
    full_dataset = KeypointCharacterDataset(
        args.data_dir, 
        args.keypoint_annotations_dir, 
        transform=train_transform, 
        target_characters=target_characters
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_transform
    
    logger.info(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    num_classes = len(full_dataset.class_to_idx)
    model = get_keypoint_aware_model(args.model_type, num_classes, keypoint_dim=128)
    model = model.to(device)
    
    logger.info(f'模型类别数: {num_classes}')
    logger.info(f'类别映射: {full_dataset.class_to_idx}')
    
    results = train_model(
        model, train_loader, val_loader, device,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        class_to_idx=full_dataset.class_to_idx,
        use_mixup=args.use_mixup,
        label_smoothing=args.label_smoothing
    )
    
    logger.info('训练完成！')
    logger.info(f'最佳验证准确率: {results["best_val_accuracy"]:.2f}%')


if __name__ == '__main__':
    main()