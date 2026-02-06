#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合系统脚本

实现图像和文本的多模态融合，提高角色识别准确性。
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
from transformers import BertTokenizer, BertModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('multimodal_fusion_system')

class MultimodalDataset(Dataset):
    """多模态数据集类"""
    
    def __init__(self, root_dir, transform=None):
        """初始化多模态数据集
        
        Args:
            root_dir: 数据目录
            transform: 数据变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.texts = []
        self.class_to_idx = {}
        
        # 构建类别映射
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            
            # 生成类别对应的文本描述
            text_description = self._generate_text_description(cls)
            
            # 遍历图像
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)
                    self.texts.append(text_description)
        
        logger.info(f"多模态数据集初始化完成，包含 {len(classes)} 个类别，{len(self.images)} 张图像")
    
    def _generate_text_description(self, class_name):
        """生成类别对应的文本描述
        
        Args:
            class_name: 类别名称
        
        Returns:
            文本描述
        """
        # 解析类别名称，提取角色信息
        parts = class_name.split('_')
        if len(parts) >= 3:
            series = parts[0] + '_' + parts[1]
            character = ' '.join(parts[2:])
            return f"Anime character {character} from {series}"
        elif len(parts) == 2:
            series = parts[0]
            character = parts[1]
            return f"Anime character {character} from {series}"
        else:
            return f"Anime character {class_name}"
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        text = self.texts[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, text, label

class MultimodalFusionModel(nn.Module):
    """多模态融合模型类"""
    
    def __init__(self, image_model_type, num_classes, text_feature_dim=768, fusion_method='concat'):
        """初始化多模态融合模型
        
        Args:
            image_model_type: 图像模型类型
            num_classes: 类别数量
            text_feature_dim: 文本特征维度
            fusion_method: 融合方法 (concat, add, multiply)
        """
        super(MultimodalFusionModel, self).__init__()
        
        self.fusion_method = fusion_method
        
        # 图像编码器
        if image_model_type == 'efficientnet_b0':
            self.image_encoder = models.efficientnet_b0(pretrained=True)
            self.image_feature_dim = self.image_encoder.classifier[1].in_features
            self.image_encoder.classifier = nn.Identity()
        elif image_model_type == 'mobilenet_v2':
            self.image_encoder = models.mobilenet_v2(pretrained=True)
            self.image_feature_dim = self.image_encoder.classifier[1].in_features
            self.image_encoder.classifier = nn.Identity()
        elif image_model_type == 'resnet18':
            self.image_encoder = models.resnet18(pretrained=True)
            self.image_feature_dim = self.image_encoder.fc.in_features
            self.image_encoder.fc = nn.Identity()
        else:
            raise ValueError(f"不支持的图像模型类型: {image_model_type}")
        
        # 文本编码器（BERT）
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_feature_dim = text_feature_dim
        
        # 融合层
        if fusion_method == 'concat':
            self.fusion_dim = self.image_feature_dim + self.text_feature_dim
            self.fusion_layer = nn.Linear(self.fusion_dim, self.fusion_dim // 2)
        elif fusion_method == 'add' or fusion_method == 'multiply':
            # 对齐特征维度
            self.image_proj = nn.Linear(self.image_feature_dim, self.text_feature_dim)
            self.fusion_dim = self.text_feature_dim
            self.fusion_layer = nn.Linear(self.fusion_dim, self.fusion_dim // 2)
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fusion_dim // 2, num_classes)
        )
        
        logger.info(f"多模态融合模型初始化完成，融合方法: {fusion_method}")
    
    def encode_text(self, texts):
        """编码文本
        
        Args:
            texts: 文本列表
        
        Returns:
            文本特征
        """
        # 标记化文本
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 移动到设备
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        
        # 编码文本
        outputs = self.text_encoder(**inputs)
        # 使用CLS token的表示作为文本特征
        text_features = outputs.pooler_output
        
        return text_features
    
    def forward(self, images, texts):
        """前向传播
        
        Args:
            images: 图像批次
            texts: 文本列表
        
        Returns:
            分类输出
        """
        # 编码图像
        image_features = self.image_encoder(images)
        if image_features.dim() == 4:  # 对于某些模型，可能需要平均池化
            image_features = torch.mean(image_features, dim=[2, 3])
        
        # 编码文本
        text_features = self.encode_text(texts)
        
        # 融合特征
        if self.fusion_method == 'concat':
            fused_features = torch.cat([image_features, text_features], dim=1)
        elif self.fusion_method == 'add':
            image_features = self.image_proj(image_features)
            fused_features = image_features + text_features
        elif self.fusion_method == 'multiply':
            image_features = self.image_proj(image_features)
            fused_features = image_features * text_features
        
        # 融合层
        fused_features = self.fusion_layer(fused_features)
        
        # 分类
        outputs = self.classifier(fused_features)
        
        return outputs

def train_multimodal_model(args):
    """训练多模态融合模型
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载多模态数据集
    dataset = MultimodalDataset(root_dir=args.data_dir, transform=transform)
    
    # 分割数据集为训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    num_classes = len(dataset.class_to_idx)
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, f'{args.image_model_type}_{args.fusion_method}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存类别映射
    with open(os.path.join(output_dir, 'class_to_idx.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset.class_to_idx, f, ensure_ascii=False, indent=2)
    
    # 创建多模态融合模型
    model = MultimodalFusionModel(
        image_model_type=args.image_model_type,
        num_classes=num_classes,
        fusion_method=args.fusion_method
    )
    model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # 训练指标
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # 最佳模型指标
    best_val_accuracy = 0.0
    
    # 训练循环
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch [{epoch+1}/{args.num_epochs}]")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}")
        
        for images, texts, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算训练指标
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
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
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        logger.info(f"训练 Loss: {train_epoch_loss:.4f}, Accuracy: {train_epoch_accuracy:.2f}%")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"验证 Epoch {epoch+1}")
            
            for images, texts, labels in progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                
                # 前向传播
                outputs = model(images, texts)
                loss = criterion(outputs, labels)
                
                # 计算验证指标
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 更新进度条
                current_acc = (val_correct / val_total) * 100 if val_total > 0 else 0
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': current_acc
                })
        
        # 计算验证 epoch 指标
        if val_total > 0:
            val_epoch_loss = val_loss / val_total
            val_epoch_accuracy = (val_correct / val_total) * 100
        else:
            val_epoch_loss = 0
            val_epoch_accuracy = 0
        
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        logger.info(f"验证 Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.2f}%")
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            best_model_path = os.path.join(output_dir, 'model_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_accuracy,
                'fusion_method': args.fusion_method,
                'class_to_idx': dataset.class_to_idx
            }, best_model_path)
            logger.info(f"保存最佳模型到: {best_model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'fusion_method': args.fusion_method,
        'class_to_idx': dataset.class_to_idx
    }, final_model_path)
    logger.info(f"保存最终模型到: {final_model_path}")
    
    # 保存训练结果
    training_results = {
        'image_model_type': args.image_model_type,
        'fusion_method': args.fusion_method,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'best_val_accuracy': best_val_accuracy,
        'final_train_accuracy': train_accuracies[-1],
        'final_val_accuracy': val_accuracies[-1],
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(training_results, f, ensure_ascii=False, indent=2)
    logger.info(f"训练结果已保存到: {results_path}")
    
    logger.info("多模态融合模型训练完成")
    logger.info(f"最佳验证准确率: {best_val_accuracy:.2f}%")
    
    return training_results

def main():
    parser = argparse.ArgumentParser(description='多模态融合系统脚本')
    
    parser.add_argument('--data-dir', type=str, default='data/train',
                       help='训练数据目录')
    parser.add_argument('--output-dir', type=str, default='models/multimodal_fusion',
                       help='输出目录')
    parser.add_argument('--image-model-type', type=str, default='mobilenet_v2',
                       choices=['efficientnet_b0', 'mobilenet_v2', 'resnet18'],
                       help='图像模型类型')
    parser.add_argument('--fusion-method', type=str, default='concat',
                       choices=['concat', 'add', 'multiply'],
                       help='融合方法')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批量大小')
    parser.add_argument('--num-epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载器工作线程数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始训练
    train_multimodal_model(args)

if __name__ == '__main__':
    main()
