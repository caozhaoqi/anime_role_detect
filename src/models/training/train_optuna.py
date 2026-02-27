#!/usr/bin/env python3
"""
使用Optuna进行超参数调优的脚本
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
import numpy as np
import optuna
from optuna.trial import TrialState

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hyperparameter_tuning')


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
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"无法加载图片 {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class SimpleCharacterClassifier(nn.Module):
    """使用MobileNetV3的简单分类器模型"""
    
    def __init__(self, num_classes):
        """初始化模型
        
        Args:
            num_classes: 类别数量
        """
        super().__init__()
        # 使用MobileNetV3 Small作为基础模型
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        
        # 替换分类头
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def objective(trial, args):
    """Optuna目标函数
    
    Args:
        trial: Optuna trial对象
        args: 命令行参数
        
    Returns:
        验证准确率
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 超参数搜索空间
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    temperature = trial.suggest_uniform('temperature', 0.5, 3.0)
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.4, 
            contrast=0.4, 
            saturation=0.4,
            hue=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
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
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # 初始化模型
    model = SimpleCharacterClassifier(num_classes=len(train_dataset.class_to_idx)).to(device)
    
    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # 训练循环
    best_val_acc = 0.0
    patience = 20
    no_improve_epochs = 0
    
    for epoch in range(args.num_epochs):
        # 训练
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                break
        
        scheduler.step()
    
    return best_val_acc

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用Optuna进行超参数调优')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default='data/augmented_dataset', help='训练集目录')
    parser.add_argument('--val_dir', type=str, default='data/split_dataset/val', help='验证集目录')
    parser.add_argument('--output_dir', type=str, default='models_optuna', help='模型输出目录')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--n_trials', type=int, default=50, help='Optuna试验次数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info('开始超参数调优...')
    logger.info(f'训练集目录: {args.train_dir}')
    logger.info(f'验证集目录: {args.val_dir}')
    logger.info(f'试验次数: {args.n_trials}')
    
    # 创建Optuna研究
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler()
    )
    
    # 运行优化
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=3600  # 1小时超时
    )
    
    # 打印结果
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    logger.info(f'试验总数: {len(study.trials)}')
    logger.info(f'剪枝试验数: {len(pruned_trials)}')
    logger.info(f'完成试验数: {len(complete_trials)}')
    
    logger.info('最佳试验:')
    best_trial = study.best_trial
    logger.info(f'  验证准确率: {best_trial.value:.4f}')
    logger.info('  超参数:')
    for key, value in best_trial.params.items():
        logger.info(f'    {key}: {value}')
    
    # 保存最佳超参数
    import json
    best_params_path = os.path.join(args.output_dir, 'best_hyperparameters.json')
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_val_acc': best_trial.value,
            'params': best_trial.params
        }, f, indent=2)
    
    logger.info(f'最佳超参数已保存到: {best_params_path}')
    
    # 使用最佳超参数重新训练模型
    logger.info('使用最佳超参数重新训练模型...')
    
    # 重新构建数据加载器
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.4, 
            contrast=0.4, 
            saturation=0.4,
            hue=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
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
        batch_size=best_trial.params['batch_size'], 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=best_trial.params['batch_size'], 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # 初始化模型
    model = SimpleCharacterClassifier(num_classes=len(train_dataset.class_to_idx)).to(device)
    
    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=best_trial.params['learning_rate'],
        weight_decay=best_trial.params['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * 2,  # 增加训练轮数
        eta_min=1e-6
    )
    
    # 训练循环
    best_val_acc = 0.0
    patience = 30
    no_improve_epochs = 0
    
    for epoch in range(args.num_epochs * 2):
        logger.info(f"开始第 {epoch+1}/{args.num_epochs*2} 轮训练")
        
        # 训练
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        logger.info(f'Epoch {epoch+1}/{args.num_epochs*2}')
        logger.info(f'Train Loss: {epoch_loss:.4f}')
        logger.info(f'Val Acc: {val_acc:.4f}')
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            # 保存最佳模型
            model_path = os.path.join(args.output_dir, 'character_classifier_optuna_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': train_dataset.class_to_idx,
                'val_acc': best_val_acc,
                'hyperparameters': best_trial.params
            }, model_path)
            logger.info(f'最佳模型已保存: {model_path}, 验证准确率: {best_val_acc:.4f}')
        else:
            no_improve_epochs += 1
            logger.info(f'早停计数器: {no_improve_epochs}/{patience}')
            if no_improve_epochs >= patience:
                logger.info(f'早停: 连续 {patience} 轮验证准确率无提升')
                break
        
        scheduler.step()
    
    logger.info(f'超参数调优完成！最佳验证准确率: {best_val_acc:.4f}')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用Optuna进行超参数调优')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default='data/augmented_dataset', help='训练集目录')
    parser.add_argument('--val_dir', type=str, default='data/split_dataset/val', help='验证集目录')
    parser.add_argument('--output_dir', type=str, default='models_optuna', help='模型输出目录')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--n_trials', type=int, default=50, help='Optuna试验次数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info('开始超参数调优...')
    logger.info(f'训练集目录: {args.train_dir}')
    logger.info(f'验证集目录: {args.val_dir}')
    logger.info(f'试验次数: {args.n_trials}')
    
    # 创建Optuna研究
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler()
    )
    
    # 运行优化
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=3600  # 1小时超时
    )
    
    # 打印结果
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    logger.info(f'试验总数: {len(study.trials)}')
    logger.info(f'剪枝试验数: {len(pruned_trials)}')
    logger.info(f'完成试验数: {len(complete_trials)}')
    
    logger.info('最佳试验:')
    best_trial = study.best_trial
    logger.info(f'  验证准确率: {best_trial.value:.4f}')
    logger.info('  超参数:')
    for key, value in best_trial.params.items():
        logger.info(f'    {key}: {value}')
    
    # 保存最佳超参数
    import json
    best_params_path = os.path.join(args.output_dir, 'best_hyperparameters.json')
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_val_acc': best_trial.value,
            'params': best_trial.params
        }, f, indent=2)
    
    logger.info(f'最佳超参数已保存到: {best_params_path}')
    
    # 使用最佳超参数重新训练模型
    logger.info('使用最佳超参数重新训练模型...')
    
    # 重新构建数据加载器
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.4, 
            contrast=0.4, 
            saturation=0.4,
            hue=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
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
        batch_size=best_trial.params['batch_size'], 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=best_trial.params['batch_size'], 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # 初始化模型
    model = SimpleCharacterClassifier(num_classes=len(train_dataset.class_to_idx)).to(device)
    
    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=best_trial.params['learning_rate'],
        weight_decay=best_trial.params['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * 2,  # 增加训练轮数
        eta_min=1e-6
    )
    
    # 训练循环
    best_val_acc = 0.0
    patience = 30
    no_improve_epochs = 0
    
    for epoch in range(args.num_epochs * 2):
        logger.info(f"开始第 {epoch+1}/{args.num_epochs*2} 轮训练")
        
        # 训练
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        logger.info(f'Epoch {epoch+1}/{args.num_epochs*2}')
        logger.info(f'Train Loss: {epoch_loss:.4f}')
        logger.info(f'Val Acc: {val_acc:.4f}')
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            # 保存最佳模型
            model_path = os.path.join(args.output_dir, 'character_classifier_optuna_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': train_dataset.class_to_idx,
                'val_acc': best_val_acc,
                'hyperparameters': best_trial.params
            }, model_path)
            logger.info(f'最佳模型已保存: {model_path}, 验证准确率: {best_val_acc:.4f}')
        else:
            no_improve_epochs += 1
            logger.info(f'早停计数器: {no_improve_epochs}/{patience}')
            if no_improve_epochs >= patience:
                logger.info(f'早停: 连续 {patience} 轮验证准确率无提升')
                break
        
        scheduler.step()
    
    logger.info(f'超参数调优完成！最佳验证准确率: {best_val_acc:.4f}')


if __name__ == "__main__":
    main()
