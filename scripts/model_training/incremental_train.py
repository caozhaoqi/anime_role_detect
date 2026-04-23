#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量训练脚本 - 在现有模型基础上使用新数据进行训练
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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("incremental_train")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("incremental_train")

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

BATCH_SIZE = 8
NUM_EPOCHS = 15  # 增量训练 epoch 数
LEARNING_RATE = 1e-5  # 更小的学习率
IMAGE_SIZE = 224
NUM_WORKERS = 0

MODEL_DIR = './models'


class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

        logger.info(f"数据集: {len(self.samples)} 样本, {len(self.class_to_idx)} 类别")
        logger.info(f"类别: {list(self.class_to_idx.keys())}")

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


def load_base_model(base_model_path, num_classes):
    """加载基础模型"""
    logger.info(f"加载基础模型: {base_model_path}")
    
    # 加载模型信息
    checkpoint = torch.load(base_model_path, map_location=torch.device('cpu'))
    model_type = checkpoint['model_type']
    
    logger.info(f"基础模型类型: {model_type}")
    
    # 创建模型
    if model_type == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, model_type


def freeze_layers(model, model_type):
    """冻结部分层"""
    logger.info(f"冻结 {model_type} 的部分层")
    
    # 根据模型类型冻结不同的层
    if model_type == 'resnet18':
        # 冻结除最后5层外的所有层
        for i, layer in enumerate(model.children()):
            if i < 6:  # 冻结前6个模块
                for param in layer.parameters():
                    param.requires_grad = False
    elif model_type == 'mobilenet_v2':
        # 冻结除分类器外的所有层
        for layer in model.features:
            for param in layer.parameters():
                param.requires_grad = False
    elif model_type == 'efficientnet_b0':
        # 冻结除分类器外的所有层
        for layer in model.features:
            for param in layer.parameters():
                param.requires_grad = False
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params}/{total_params} ({trainable_params/total_params*100:.2f}%)")
    
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
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


def incremental_train(base_model_path, new_data_dir, output_model_dir):
    """增量训练"""
    device = torch.device('cpu')
    logger.info(f"使用设备: {device}")

    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载新数据
    logger.info(f"加载新数据: {new_data_dir}")
    full_dataset = SimpleImageDataset(new_data_dir, transform=train_transform)
    
    num_classes = len(full_dataset.class_to_idx)
    logger.info(f"类别数: {num_classes}")

    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    logger.info(f"训练集: {len(train_dataset)}")
    logger.info(f"验证集: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 加载基础模型
    model, model_type = load_base_model(base_model_path, num_classes)
    model = model.to(device)
    
    # 冻结部分层
    model = freeze_layers(model, model_type)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 只优化可训练参数
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0.0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
            logger.info(f"  保存最佳模型，准确率: {best_acc:.2f}%")

    model.load_state_dict(best_model_state)
    logger.info(f"增量训练完成，最佳验证准确率: {best_acc:.2f}%")
    return model, best_acc, model_type, full_dataset.class_to_idx


def save_model(model, model_type, class_to_idx, accuracy, output_model_dir):
    """保存模型"""
    os.makedirs(output_model_dir, exist_ok=True)

    # 生成版本号
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{model_type}_loli_incremental_{version}"
    model_dir = os.path.join(output_model_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 保存模型状态
    model_path = os.path.join(model_dir, 'model_best.pth')
    checkpoint = {
        'model_type': model_type,
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
        'is_incremental': True
    }
    torch.save(checkpoint, model_path)

    # 保存完整模型
    full_model_path = os.path.join(model_dir, 'model_full.pth')
    torch.save(model, full_model_path)

    # 保存类别映射
    class_map_path = os.path.join(model_dir, 'class_to_idx.json')
    with open(class_map_path, 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    # 更新最佳模型链接
    best_model_link = os.path.join(output_model_dir, 'best_incremental_model.txt')
    with open(best_model_link, 'w') as f:
        f.write(model_dir)

    logger.info(f"模型已保存: {model_path}")
    logger.info(f"最佳模型链接已更新: {best_model_link}")
    return model_path, model_dir


def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    logger.info(f"模型评估准确率: {accuracy:.2f}%")
    return accuracy


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='增量训练脚本')
    parser.add_argument('--base_model', required=True, help='基础模型路径')
    parser.add_argument('--new_data', required=True, help='新数据目录')
    parser.add_argument('--output_dir', default=MODEL_DIR, help='输出目录')
    parser.add_argument('--test_data', help='测试数据目录')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("开始增量训练")
    logger.info("=" * 60)
    logger.info(f"基础模型: {args.base_model}")
    logger.info(f"新数据: {args.new_data}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 运行增量训练
    model, accuracy, model_type, class_to_idx = incremental_train(
        args.base_model, args.new_data, args.output_dir
    )
    
    # 保存模型
    model_path, model_dir = save_model(
        model, model_type, class_to_idx, accuracy, args.output_dir
    )
    
    # 评估模型（如果提供测试数据）
    if args.test_data:
        logger.info("评估模型性能...")
        test_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_dataset = SimpleImageDataset(args.test_data, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_accuracy = evaluate_model(model, test_loader, torch.device('cpu'))
        
        # 更新模型元数据
        metadata_path = os.path.join(model_dir, 'metadata.json')
        metadata = {
            'model_type': model_type,
            'accuracy': accuracy,
            'test_accuracy': test_accuracy,
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'new_data_dir': args.new_data,
                'test_data_dir': args.test_data,
                'num_classes': len(class_to_idx)
            }
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试准确率: {test_accuracy:.2f}%")
    
    logger.info("=" * 60)
    logger.info("增量训练完成")
    logger.info("=" * 60)
    logger.info(f"模型保存路径: {model_path}")
    logger.info(f"模型准确率: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
