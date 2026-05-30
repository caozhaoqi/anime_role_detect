#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识蒸馏训练脚本
使用 EfficientNet-B3 教师模型蒸馏到 EfficientNet-B0 学生模型
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 配置
TEACHER_MODEL = "efficientnet_b3_loli_optimized_v2_20260529_133654"
MODEL_DIR = os.path.join(project_root, "models", TEACHER_MODEL)
OUTPUT_DIR = os.path.join(project_root, "models", "student_efficientnet_b0")
DATA_DIR = os.path.join(project_root, "data", "final_dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""

    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        # 硬损失 (交叉熵)
        hard_loss = F.cross_entropy(student_logits, labels)

        # 软损失 (KL散度)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (
            self.temperature**2
        )

        # 加权组合
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


class AverageMeter:
    """计算并存储平均值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_teacher_model():
    """加载教师模型"""
    print("=" * 60)
    print("🔄 加载教师模型 (EfficientNet-B3)...")
    print("=" * 60)

    model_path = os.path.join(MODEL_DIR, "model_full.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "model_best.pth")

    with open(os.path.join(MODEL_DIR, "training_results.json"), "r") as f:
        config = json.load(f)

    num_classes = config.get("num_classes", 74)

    # 创建教师模型
    teacher = models.efficientnet_b3(num_classes=num_classes)

    # 加载权重
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        teacher = checkpoint
    elif "model_state_dict" in checkpoint:
        teacher.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        teacher.load_state_dict(checkpoint, strict=False)

    teacher.eval()
    print("✅ 教师模型加载成功")
    return teacher, num_classes


def create_student_model(num_classes):
    """创建学生模型 (EfficientNet-B0)"""
    print("\n📝 创建学生模型 (EfficientNet-B0)...")

    student = models.efficientnet_b0(num_classes=num_classes)

    print("✅ 学生模型创建成功")
    return student


def get_data_loaders(batch_size=32, image_size=224):
    """获取数据加载器"""
    print("\n📂 准备数据集...")

    # 图像预处理
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 使用 ImageFolder 风格的数据集
    from torchvision.datasets import ImageFolder

    # 获取类别名
    class_names = []
    if os.path.exists(DATA_DIR):
        class_names = sorted(
            [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        )
        print(f"   找到 {len(class_names)} 个类别")

    # 训练数据集
    train_dataset = ImageFolder(DATA_DIR, transform=train_transform)

    # 划分训练/验证集 (90/10)
    total_size = len(train_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"   训练样本: {len(train_dataset)}")
    print(f"   验证样本: {len(val_dataset)}")

    return train_loader, val_loader, class_names


def train_epoch(teacher, student, train_loader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    teacher.eval()
    student.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 教师预测 (不计算梯度)
        with torch.no_grad():
            teacher_logits = teacher(images)

        # 学生预测
        student_logits = student(images)

        # 计算损失
        loss = criterion(student_logits, teacher_logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = student_logits.max(1)
        acc = (predicted == labels).float().mean().item()

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.4f}"})

    return loss_meter.avg, acc_meter.avg


def validate(teacher, student, val_loader, device):
    """验证模型"""
    teacher.eval()
    student.eval()

    correct = 0
    total = 0
    top3_correct = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            # 教师预测
            teacher_logits = teacher(images)

            # 学生预测
            student_logits = student(images)

            # Top-1 准确率
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Top-3 准确率
            _, top3_pred = student_logits.topk(3, 1, True, True)
            top3_correct += (top3_pred == labels.view(-1, 1).expand_as(top3_pred)).sum().item()

    top1_acc = 100.0 * correct / total
    top3_acc = 100.0 * top3_correct / total

    return top1_acc, top3_acc


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.7, help="Hard loss weight")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 知识蒸馏训练 - EfficientNet-B0 学生模型")
    print("=" * 60)
    print(f"\n配置:")
    print(f"   教师模型: EfficientNet-B3")
    print(f"   学生模型: EfficientNet-B0")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Alpha: {args.alpha}")

    # 设备
    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
    print(f"\n设备: {device}")

    # 加载模型
    teacher, num_classes = load_teacher_model()
    teacher = teacher.to(device)

    student = create_student_model(num_classes)
    student = student.to(device)

    # 数据加载器
    train_loader, val_loader, class_names = get_data_loaders(
        batch_size=args.batch_size, image_size=args.image_size
    )

    # 损失函数
    criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)

    # 优化器
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0001)

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    print("\n" + "=" * 60)
    print("📝 开始训练...")
    print("=" * 60)

    best_top1 = 0
    best_top3 = 0

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            teacher, student, train_loader, criterion, optimizer, device, epoch
        )

        # 验证
        top1_acc, top3_acc = validate(teacher, student, val_loader, device)

        # 学习率调整
        scheduler.step()

        # 保存最佳模型
        is_best = top1_acc > best_top1
        if is_best:
            best_top1 = top1_acc
            best_top3 = top3_acc

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "top1_acc": top1_acc,
                    "top3_acc": top3_acc,
                },
                os.path.join(OUTPUT_DIR, "student_best.pth"),
            )

        print(
            f"\nEpoch {epoch}: Top-1={top1_acc:.2f}%, Top-3={top3_acc:.2f}% | Best: Top-1={best_top1:.2f}%, Top-3={best_top3:.2f}%"
        )

    # 保存训练结果
    results = {
        "model_name": "efficientnet_b0",
        "teacher_model": "efficientnet_b3",
        "num_classes": num_classes,
        "best_top1_accuracy": best_top1 / 100,
        "best_top3_accuracy": best_top3 / 100,
        "class_names": class_names if class_names else None,
        "training_config": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "image_size": args.image_size,
        },
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    with open(os.path.join(OUTPUT_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"\n最佳学生模型:")
    print(f"   Top-1 准确率: {best_top1:.2f}%")
    print(f"   Top-3 准确率: {best_top3:.2f}%")
    print(f"\n模型保存位置: {OUTPUT_DIR}")
    print(f"   - student_best.pth")
    print(f"   - training_results.json")


if __name__ == "__main__":
    main()
