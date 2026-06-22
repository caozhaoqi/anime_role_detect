#!/usr/bin/env python3
"""带数据增强的训练脚本"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import gc
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRAIN_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/training_dataset")
MODEL_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/models")
LOG_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/logs")


class SafeImageFolder(datasets.ImageFolder):
    """安全的图片文件夹数据集，自动跳过损坏的图片"""
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except Exception as e:
            logger.warning(f"跳过损坏图片: {path} - {e}")
            sample = Image.new('RGB', (224, 224), color=0)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def get_transforms(augment=True):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            return True
        return False


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=50, device='cpu', mixup_alpha=0.5,
                start_epoch=0, best_acc=0.0, history=None):
    history = history or {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    early_stopping = EarlyStopping(patience=5)

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*60}")

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc="训练"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

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
        logger.info(f"训练 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

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
        logger.info(f"验证 Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }, MODEL_DIR / "mobilenetv2_aug_best.pth")
            logger.info(f"保存最佳模型 (Acc: {best_acc:.4f})")

        if early_stopping(epoch_val_loss):
            logger.info(f"早停触发，停止训练")
            break

        gc.collect()

    logger.info(f"\n训练完成！最佳准确率: {best_acc:.4f}")
    return model, history


def main(resume=False, model_type='mobilenet'):
    logger.info("=" * 60)
    logger.info("带数据增强的模型训练")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"设备: {device}")

    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)

    full_dataset = SafeImageFolder(str(TRAIN_DIR), transform=None)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset_raw, val_dataset_raw = random_split(full_dataset, [train_size, val_size])

    train_dataset = TransformDataset(train_dataset_raw, train_transform)
    val_dataset = TransformDataset(val_dataset_raw, val_transform)

    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    logger.info(f"类别数: {len(full_dataset.classes)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    num_classes = len(full_dataset.classes)
    if model_type == 'efficientnet':
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_type == 'resnet':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    start_epoch = 0
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_path = MODEL_DIR / "mobilenetv2_aug_best.pth"
    history_path = LOG_DIR / "training_history_aug.json"

    if resume and best_model_path.exists():
        logger.info("从已保存的最佳模型恢复训练...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
        logger.info(f"已完成 Epoch: {start_epoch}, 最佳准确率: {best_acc:.4f}")

    logger.info("开始训练...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=50, device=device, mixup_alpha=0.5,
        start_epoch=start_epoch, best_acc=best_acc, history=history
    )

    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / "training_history_aug.json", 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n训练完成！")
    logger.info(f"最佳模型: {MODEL_DIR / 'mobilenetv2_aug_best.pth'}")
    logger.info(f"训练历史: {LOG_DIR / 'training_history_aug.json'}")


class TransformDataset(Dataset):
    """包装数据集并应用变换"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='带数据增强的模型训练')
    parser.add_argument('--resume', action='store_true', help='从上次保存的最佳模型继续训练')
    parser.add_argument('--model', type=str, default='mobilenet', choices=['mobilenet', 'efficientnet', 'resnet'])
    args = parser.parse_args()
    main(resume=args.resume, model_type=args.model)
