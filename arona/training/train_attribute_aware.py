#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
属性辅助分类模型
结合图像特征和属性信息进行分类
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('attribute_aware_model')


class AttributeAwareClassifier(nn.Module):
    """属性感知分类器
    
    结合CNN图像特征和属性嵌入向量进行分类
    """
    
    def __init__(self, num_classes, attribute_dims, embed_dim=128, dropout=0.3):
        super(AttributeAwareClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.attribute_dims = attribute_dims
        self.embed_dim = embed_dim
        
        # 加载预训练的CNN骨干网络
        self.backbone = models.mobilenet_v2(pretrained=True)
        
        # 冻结骨干网络的部分层
        for param in self.backbone.features[:14].parameters():
            param.requires_grad = False
        
        # 获取特征维度
        backbone_output_dim = self.backbone.classifier[1].in_features
        
        # 图像特征投影层
        self.image_projection = nn.Sequential(
            nn.Linear(backbone_output_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout)
        )
        
        # 属性嵌入层
        self.attribute_embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim // 4) for dim in attribute_dims
        ])
        
        # 属性特征融合层
        total_attr_dim = embed_dim // 4 * len(attribute_dims)
        self.attr_projection = nn.Sequential(
            nn.Linear(total_attr_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout)
        )
        
        # 融合分类器
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # 替换原始分类器
        self.backbone.classifier = nn.Identity()
    
    def forward(self, x, attributes):
        batch_size = x.size(0)
        
        # 提取图像特征
        image_features = self.backbone(x)
        image_features = image_features.view(batch_size, -1)
        image_embed = self.image_projection(image_features)
        
        # 处理属性嵌入
        attr_embeds = []
        for i, (attr_emb_layer, attr_value) in enumerate(zip(self.attribute_embeddings, attributes.T)):
            attr_embed = attr_emb_layer(attr_value)
            attr_embeds.append(attr_embed)
        
        # 拼接所有属性嵌入
        attr_features = torch.cat(attr_embeds, dim=1)
        attr_embed = self.attr_projection(attr_features)
        
        # 融合图像和属性特征
        combined = torch.cat([image_embed, attr_embed], dim=1)
        logits = self.fusion(combined)
        
        return logits


class AttributeDataset(torch.utils.data.Dataset):
    """带属性的数据集"""
    
    def __init__(self, data_dir, annotations_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        logger.info(f"加载 {len(self.annotations)} 条标注")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        img_path = os.path.join(self.data_dir, ann['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 获取属性标签
        attributes = torch.tensor(ann['attribute_labels'], dtype=torch.long)
        
        # 获取角色标签
        character = ann['character']
        
        return image, attributes, character


def train_attribute_aware_model(data_dir, annotations_file, output_dir, num_epochs=50, batch_size=32, lr=0.001):
    """训练属性感知模型"""
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    full_dataset = AttributeDataset(data_dir, annotations_file, train_transform)
    
    # 获取类别映射
    characters = sorted(list(set([ann['character'] for ann in full_dataset.annotations])))
    class_to_idx = {char: idx for idx, char in enumerate(characters)}
    num_classes = len(class_to_idx)
    
    logger.info(f"类别数: {num_classes}")
    logger.info(f"类别映射: {class_to_idx}")
    
    # 获取属性维度
    attribute_dims = []
    for i in range(len(full_dataset.annotations[0]['attribute_labels'])):
        unique_values = set()
        for ann in full_dataset.annotations:
            unique_values.add(ann['attribute_labels'][i])
        attribute_dims.append(len(unique_values))
    
    logger.info(f"属性维度: {attribute_dims}")
    
    # 分割数据集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_transform
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 创建模型
    model = AttributeAwareClassifier(num_classes, attribute_dims, embed_dim=128, dropout=0.3)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练循环
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, attributes, characters in train_loader:
            images = images.to(device)
            attributes = attributes.to(device)
            labels = torch.tensor([class_to_idx[c] for c in characters], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(images, attributes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, attributes, characters in val_loader:
                images = images.to(device)
                attributes = attributes.to(device)
                labels = torch.tensor([class_to_idx[c] for c in characters], dtype=torch.long).to(device)
                
                outputs = model(images, attributes)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}: '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                   f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': class_to_idx,
                'attribute_dims': attribute_dims
            }, os.path.join(output_dir, 'model_best.pth'))
            logger.info(f'保存最佳模型，验证准确率: {val_acc:.2f}%')
    
    logger.info(f'训练完成，最佳验证准确率: {best_val_acc:.2f}%')


def main():
    parser = argparse.ArgumentParser(description='训练属性感知分类模型')
    parser.add_argument('--data-dir', type=str, default='../../data/train', help='数据目录')
    parser.add_argument('--annotations-file', type=str, default='../../config/attribute_annotations.json', help='标注文件')
    parser.add_argument('--output-dir', type=str, default='../../models/attribute_aware', help='输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    
    args = parser.parse_args()
    
    train_attribute_aware_model(
        args.data_dir,
        args.annotations_file,
        args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == '__main__':
    main()
