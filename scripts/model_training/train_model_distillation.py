#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型蒸馏训练脚本

使用教师-学生架构实现知识蒸馏，减小模型体积同时保持性能
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import logging
from tqdm import tqdm
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_distillation')

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

class DistillationLoss(nn.Module):
    """蒸馏损失函数"""
    
    def __init__(self, temperature=3.0, alpha=0.7):
        """初始化蒸馏损失
        
        Args:
            temperature: 温度参数，控制软标签的平滑程度
            alpha: 蒸馏损失的权重
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_outputs, teacher_outputs, labels):
        """计算蒸馏损失
        
        Args:
            student_outputs: 学生模型的输出
            teacher_outputs: 教师模型的输出
            labels: 真实标签
        
        Returns:
            总损失
        """
        # 计算硬标签损失
        hard_loss = self.ce_loss(student_outputs, labels)
        
        # 计算软标签损失（蒸馏损失）
        student_logits = student_outputs / self.temperature
        teacher_logits = teacher_outputs / self.temperature
        
        soft_loss = nn.functional.kl_div(
            nn.functional.log_softmax(student_logits, dim=1),
            nn.functional.softmax(teacher_logits, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 总损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return total_loss

def get_teacher_model(num_classes):
    """获取教师模型
    
    Args:
        num_classes: 类别数量
    
    Returns:
        教师模型
    """
    logger.info("加载教师模型: EfficientNet-B0")
    model = models.efficientnet_b0(pretrained=True)
    # 替换分类头
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def get_student_model(num_classes, model_type='mobilenet'):
    """获取学生模型
    
    Args:
        num_classes: 类别数量
        model_type: 学生模型类型 ('mobilenet', 'shufflenet', 'squeezenet')
    
    Returns:
        学生模型
    """
    if model_type == 'mobilenet':
        logger.info("加载学生模型: MobileNetV2")
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'shufflenet':
        logger.info("加载学生模型: ShuffleNetV2")
        model = models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'squeezenet':
        logger.info("加载学生模型: SqueezeNet")
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    else:
        raise ValueError(f"不支持的学生模型类型: {model_type}")
    
    return model

def calculate_model_size(model):
    """计算模型大小
    
    Args:
        model: 模型
    
    Returns:
        模型大小（MB）
    """
    total_params = sum(p.numel() for p in model.parameters())
    model_size = total_params * 4 / (1024 * 1024)
    return model_size, total_params

def train_distillation(args):
    """训练蒸馏模型
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    # self.device = torch.device(
    #     'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    train_dataset = CharacterDataset(root_dir=args.train_dir, transform=train_transform)
    val_dataset = CharacterDataset(root_dir=args.val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    num_classes = len(train_dataset.class_to_idx)
    logger.info(f"类别数量: {num_classes}")
    
    # 保存类别映射
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'class_to_idx.json'), 'w', encoding='utf-8') as f:
        json.dump(train_dataset.class_to_idx, f, ensure_ascii=False, indent=2)
    
    # 加载教师模型
    teacher_model = get_teacher_model(num_classes)
    
    # 如果提供了教师模型权重，加载权重
    if args.teacher_model_path:
        logger.info(f"加载教师模型权重: {args.teacher_model_path}")
        try:
            state_dict = torch.load(args.teacher_model_path, map_location=device)
            if 'model_state_dict' in state_dict:
                teacher_model.load_state_dict(state_dict['model_state_dict'])
            else:
                teacher_model.load_state_dict(state_dict)
            logger.info("教师模型权重加载成功")
        except Exception as e:
            logger.error(f"教师模型权重加载失败: {e}")
    
    teacher_model.to(device)
    teacher_model.eval()  # 教师模型只用于推理
    
    # 加载学生模型
    student_model = get_student_model(num_classes, args.student_model_type)
    student_model.to(device)
    
    # 计算模型大小
    teacher_size, teacher_params = calculate_model_size(teacher_model)
    student_size, student_params = calculate_model_size(student_model)
    
    logger.info(f"教师模型大小: {teacher_size:.2f} MB, 参数数量: {teacher_params:,}")
    logger.info(f"学生模型大小: {student_size:.2f} MB, 参数数量: {student_params:,}")
    logger.info(f"模型压缩率: {teacher_size / student_size:.2f}x")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    distillation_loss = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
    
    # 最佳模型指标
    best_val_accuracy = 0.0
    
    # 训练循环
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch [{epoch+1}/{args.num_epochs}]")
        
        # 训练阶段
        student_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}")
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 教师模型推理（不计算梯度）
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            
            # 学生模型前向传播
            student_outputs = student_model(images)
            
            # 计算损失
            loss = distillation_loss(student_outputs, teacher_outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算训练指标
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(student_outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': (train_correct / train_total) * 100
            })
        
        # 计算训练 epoch 指标
        train_epoch_loss = train_loss / train_total
        train_epoch_accuracy = (train_correct / train_total) * 100
        logger.info(f"训练 Loss: {train_epoch_loss:.4f}, Accuracy: {train_epoch_accuracy:.2f}%")
        
        # 验证阶段
        student_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"验证 Epoch {epoch+1}")
            
            for images, labels in progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                
                # 过滤掉超出学生模型类别范围的样本
                valid_indices = labels < num_classes
                if not valid_indices.any():
                    continue
                
                # 使用过滤后的样本
                valid_images = images[valid_indices]
                valid_labels = labels[valid_indices]
                
                # 学生模型推理
                valid_outputs = student_model(valid_images)
                
                # 计算损失（只使用有效的标签）
                hard_loss = nn.CrossEntropyLoss()(valid_outputs, valid_labels)
                val_loss += hard_loss.item() * valid_images.size(0)
                
                # 计算验证指标
                _, predicted = torch.max(valid_outputs, 1)
                val_total += valid_labels.size(0)
                val_correct += (predicted == valid_labels).sum().item()
                
                # 更新进度条
                current_acc = (val_correct / val_total) * 100 if val_total > 0 else 0
                progress_bar.set_postfix({
                    'loss': hard_loss.item(),
                    'acc': current_acc
                })
        
        # 计算验证 epoch 指标
        if val_total > 0:
            val_epoch_loss = val_loss / val_total
            val_epoch_accuracy = (val_correct / val_total) * 100
        else:
            val_epoch_loss = 0
            val_epoch_accuracy = 0
        logger.info(f"验证 Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.2f}%")
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            best_model_path = os.path.join(args.output_dir, 'student_model_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_accuracy,
                'teacher_model_type': 'efficientnet_b0',
                'student_model_type': args.student_model_type,
                'temperature': args.temperature,
                'alpha': args.alpha
            }, best_model_path)
            logger.info(f"保存最佳模型到: {best_model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'student_model_final.pth')
    torch.save({
        'model_state_dict': student_model.state_dict(),
        'teacher_model_type': 'efficientnet_b0',
        'student_model_type': args.student_model_type,
        'temperature': args.temperature,
        'alpha': args.alpha
    }, final_model_path)
    logger.info(f"保存最终模型到: {final_model_path}")
    
    # 评估教师模型
    logger.info("\n评估教师模型性能")
    teacher_accuracy = evaluate_model(teacher_model, val_loader, device)
    logger.info(f"教师模型验证准确率: {teacher_accuracy:.2f}%")
    
    logger.info("模型蒸馏训练完成")
    logger.info(f"最佳学生模型验证准确率: {best_val_accuracy:.2f}%")
    logger.info(f"学生模型相对于教师模型的性能保持率: {best_val_accuracy / teacher_accuracy * 100:.2f}%")

def evaluate_model(model, data_loader, device):
    """评估模型性能
    
    Args:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 设备
    
    Returns:
        准确率
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (correct / total) * 100
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='模型蒸馏训练脚本')
    
    # 数据参数
    parser.add_argument('--train-dir', type=str, required=True,
                       help='训练数据目录')
    parser.add_argument('--val-dir', type=str, required=True,
                       help='验证数据目录')
    parser.add_argument('--output-dir', type=str, default='models/distillation',
                       help='输出目录')
    
    # 模型参数
    parser.add_argument('--teacher-model-path', type=str,
                       help='教师模型权重路径')
    parser.add_argument('--student-model-type', type=str, default='mobilenet',
                       choices=['mobilenet', 'shufflenet', 'squeezenet'],
                       help='学生模型类型')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批量大小')
    parser.add_argument('--num-epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载器工作线程数')
    
    # 蒸馏参数
    parser.add_argument('--temperature', type=float, default=3.0,
                       help='温度参数')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='蒸馏损失权重')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始训练
    train_distillation(args)

if __name__ == '__main__':
    main()
