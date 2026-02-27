#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合系统

结合文本描述等信息，提升角色识别的准确性。
"""

import os
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('multimodal_fusion')

class MultimodalDataset(Dataset):
    """多模态数据集"""
    def __init__(self, data_dir, text_annotations=None, transform=None, tokenizer=None, max_length=128):
        self.data_dir = data_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_paths = []
        self.labels = []
        self.texts = []
        self.class_to_idx = {}
        
        # 遍历目录结构
        for idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(idx)
                        
                        # 生成文本描述
                        text = self._generate_text_description(class_name, img_name)
                        if text_annotations and class_name in text_annotations:
                            text = text_annotations[class_name]
                        self.texts.append(text)
        
        logger.info(f"数据集初始化完成，包含 {len(self.class_to_idx)} 个类别，{len(self.image_paths)} 张图像")
    
    def _generate_text_description(self, class_name, img_name):
        """生成文本描述"""
        # 从类别名中提取信息
        parts = class_name.split('_')
        if len(parts) >= 2:
            series = parts[0]
            character = '_'.join(parts[1:])
            return f"This is {character} from {series} anime/manga series."
        return f"This is a character from anime/manga series."
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        text = self.texts[idx]
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # 处理文本
            if self.tokenizer:
                text_inputs = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                text_input_ids = text_inputs['input_ids'].squeeze()
                text_attention_mask = text_inputs['attention_mask'].squeeze()
                return image, text_input_ids, text_attention_mask, label
            else:
                return image, label
        except Exception as e:
            logger.error(f"加载数据 {img_path} 失败: {e}")
            # 返回占位符
            if self.tokenizer:
                return torch.zeros(3, 224, 224), torch.zeros(self.max_length, dtype=torch.long), torch.zeros(self.max_length, dtype=torch.long), label
            else:
                return torch.zeros(3, 224, 224), label

class ImageEncoder(nn.Module):
    """图像编码器"""
    def __init__(self, embedding_size=512):
        super(ImageEncoder, self).__init__()
        from torchvision import models
        self.backbone = models.efficientnet_b0(pretrained=False)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.projection = nn.Linear(self.feature_dim, embedding_size)
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        embeddings = nn.functional.normalize(embeddings, dim=1)
        return embeddings

class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, embedding_size=512):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.projection = nn.Linear(self.bert.config.hidden_size, embedding_size)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用CLS token的输出
        cls_output = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(cls_output)
        embeddings = nn.functional.normalize(embeddings, dim=1)
        return embeddings

class MultimodalFusionModel(nn.Module):
    """多模态融合模型"""
    def __init__(self, num_classes=1000, embedding_size=512, fusion_method='concat'):
        super(MultimodalFusionModel, self).__init__()
        self.image_encoder = ImageEncoder(embedding_size=embedding_size)
        self.text_encoder = TextEncoder(embedding_size=embedding_size)
        self.fusion_method = fusion_method
        
        # 根据融合方法确定融合维度
        if fusion_method == 'concat':
            fusion_dim = embedding_size * 2
        elif fusion_method == 'add':
            fusion_dim = embedding_size
        elif fusion_method == 'multiply':
            fusion_dim = embedding_size
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim // 2, num_classes)
        )
    
    def forward(self, image, text_input_ids, text_attention_mask):
        # 提取图像特征
        image_embeddings = self.image_encoder(image)
        
        # 提取文本特征
        text_embeddings = self.text_encoder(text_input_ids, text_attention_mask)
        
        # 融合特征
        if self.fusion_method == 'concat':
            fused_features = torch.cat([image_embeddings, text_embeddings], dim=1)
        elif self.fusion_method == 'add':
            fused_features = image_embeddings + text_embeddings
        elif self.fusion_method == 'multiply':
            fused_features = image_embeddings * text_embeddings
        else:
            fused_features = image_embeddings
        
        # 分类
        logits = self.classifier(fused_features)
        return logits, image_embeddings, text_embeddings, fused_features

class MultimodalFusionSystem:
    """多模态融合系统"""
    def __init__(self, model_path=None, num_classes=1000, embedding_size=512, fusion_method='concat', device='cpu'):
        self.device = device
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.fusion_method = fusion_method
        self.model = self._load_model(model_path)
        self.tokenizer = self._load_tokenizer()
        self.class_to_idx = {}
        self.idx_to_class = {}
    
    def _load_model(self, model_path):
        """加载模型"""
        model = MultimodalFusionModel(
            num_classes=self.num_classes,
            embedding_size=self.embedding_size,
            fusion_method=self.fusion_method
        )
        if model_path and os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"成功加载模型: {model_path}")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                logger.info("使用随机初始化的模型")
        model = model.to(self.device)
        return model
    
    def _load_tokenizer(self):
        """加载分词器"""
        try:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            logger.info("成功加载BERT分词器")
            return tokenizer
        except Exception as e:
            logger.error(f"加载分词器失败: {e}")
            return None
    
    def train(self, data_dir, text_annotations=None, epochs=10, batch_size=32, learning_rate=1e-4):
        """训练模型"""
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载数据集
        dataset = MultimodalDataset(
            data_dir=data_dir,
            text_annotations=text_annotations,
            transform=transform,
            tokenizer=self.tokenizer
        )
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 更新类别数量
        self.num_classes = len(self.class_to_idx)
        # 重新初始化分类器
        self._update_classifier(self.num_classes)
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # 训练模型
        self.model.train()
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = tqdm(dataloader, desc=f"训练 Epoch {epoch+1}")
            for images, text_input_ids, text_attention_mask, labels in progress_bar:
                images = images.to(self.device)
                text_input_ids = text_input_ids.to(self.device)
                text_attention_mask = text_attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs, _, _, _ = self.model(images, text_input_ids, text_attention_mask)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计指标
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': (correct / total) * 100
                })
            
            # 计算epoch指标
            epoch_loss = running_loss / total
            epoch_accuracy = (correct / total) * 100
            logger.info(f"训练 Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            
            # 更新学习率
            scheduler.step()
            
            # 保存最佳模型
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                logger.info(f"保存最佳模型，准确率: {best_accuracy:.2f}%")
        
        return best_accuracy
    
    def _update_classifier(self, num_classes):
        """更新分类器"""
        # 根据融合方法确定融合维度
        if self.fusion_method == 'concat':
            fusion_dim = self.embedding_size * 2
        elif self.fusion_method == 'add':
            fusion_dim = self.embedding_size
        elif self.fusion_method == 'multiply':
            fusion_dim = self.embedding_size
        else:
            fusion_dim = self.embedding_size
        
        # 替换分类器
        self.model.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        logger.info(f"分类器已更新，支持 {num_classes} 个类别")
    
    def evaluate(self, data_dir, text_annotations=None, batch_size=32):
        """评估模型"""
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载数据集
        dataset = MultimodalDataset(
            data_dir=data_dir,
            text_annotations=text_annotations,
            transform=transform,
            tokenizer=self.tokenizer
        )
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 评估模型
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="评估")
            for images, text_input_ids, text_attention_mask, labels in progress_bar:
                images = images.to(self.device)
                text_input_ids = text_input_ids.to(self.device)
                text_attention_mask = text_attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs, _, _, _ = self.model(images, text_input_ids, text_attention_mask)
                
                # 计算损失
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                
                # 统计指标
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': (correct / total) * 100
                })
        
        # 计算指标
        eval_loss = running_loss / total
        eval_accuracy = (correct / total) * 100
        logger.info(f"评估 Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.2f}%")
        
        return eval_accuracy
    
    def predict(self, image_path, text_description=None):
        """预测角色"""
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        
        # 处理文本
        if not text_description:
            text_description = "This is a character from anime/manga series."
        
        text_inputs = self.tokenizer(
            text_description,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_input_ids = text_inputs['input_ids'].to(self.device)
        text_attention_mask = text_inputs['attention_mask'].to(self.device)
        
        # 模型推理
        self.model.eval()
        with torch.no_grad():
            outputs, _, _, _ = self.model(image, text_input_ids, text_attention_mask)
            _, predicted = torch.max(outputs, 1)
            predicted_idx = predicted.item()
            predicted_character = self.idx_to_class.get(predicted_idx, f"unknown_{predicted_idx}")
            
            # 计算置信度
            probabilities = nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0, predicted_idx].item()
        
        return {
            'character': predicted_character,
            'confidence': confidence,
            'text_description': text_description
        }
    
    def save_model(self, output_path):
        """保存模型"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(self.model.state_dict(), output_path)
        logger.info(f"模型已保存到: {output_path}")
    
    def save_config(self, output_path):
        """保存配置"""
        config = {
            'num_classes': self.num_classes,
            'embedding_size': self.embedding_size,
            'fusion_method': self.fusion_method,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info(f"配置已保存到: {output_path}")
    
    def load_config(self, config_path):
        """加载配置"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                self.num_classes = config.get('num_classes', self.num_classes)
                self.embedding_size = config.get('embedding_size', self.embedding_size)
                self.fusion_method = config.get('fusion_method', self.fusion_method)
                self.class_to_idx = config.get('class_to_idx', {})
                self.idx_to_class = config.get('idx_to_class', {})
                
                logger.info(f"配置已加载，包含 {len(self.class_to_idx)} 个类别")
                return True
            except Exception as e:
                logger.error(f"加载配置失败: {e}")
                return False
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多模态融合系统')
    parser.add_argument('--model-path', type=str, default='models/multimodal_model.pth', help='模型路径')
    parser.add_argument('--config-path', type=str, default='models/multimodal_config.json', help='配置路径')
    parser.add_argument('--train-dir', type=str, default='data/train', help='训练数据目录')
    parser.add_argument('--val-dir', type=str, default='data/split_dataset/val', help='验证数据目录')
    parser.add_argument('--output-model', type=str, default='models/multimodal_model_trained.pth', help='输出模型路径')
    parser.add_argument('--output-config', type=str, default='models/multimodal_config_trained.json', help='输出配置路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--fusion-method', type=str, default='concat', choices=['concat', 'add', 'multiply'], help='融合方法')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    args = parser.parse_args()
    
    # 初始化多模态融合系统
    system = MultimodalFusionSystem(
        model_path=args.model_path,
        fusion_method=args.fusion_method,
        device=args.device
    )
    
    # 加载配置
    system.load_config(args.config_path)
    
    # 训练模型
    logger.info("开始训练多模态融合模型...")
    train_accuracy = system.train(
        data_dir=args.train_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    logger.info(f"模型训练完成，训练准确率: {train_accuracy:.2f}%")
    
    # 评估模型
    logger.info("开始评估多模态融合模型...")
    val_accuracy = system.evaluate(
        data_dir=args.val_dir,
        batch_size=args.batch_size
    )
    logger.info(f"模型评估完成，验证准确率: {val_accuracy:.2f}%")
    
    # 保存模型和配置
    system.save_model(args.output_model)
    system.save_config(args.output_config)
    
    logger.info("多模态融合系统运行完成")

if __name__ == '__main__':
    main()
