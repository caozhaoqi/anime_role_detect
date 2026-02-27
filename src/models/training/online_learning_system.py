#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线学习系统

实现模型的在线学习能力，持续适应新角色。
支持增量学习和模型更新，同时保持对现有角色的识别能力。
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
import faiss

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('online_learning')

class CharacterDataset(Dataset):
    """角色数据集"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
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
        
        logger.info(f"数据集初始化完成，包含 {len(self.class_to_idx)} 个类别，{len(self.image_paths)} 张图像")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"加载图像 {img_path} 失败: {e}")
            # 返回一个随机图像和标签作为占位符
            return torch.zeros(3, 224, 224), label

class ArcFaceModel(nn.Module):
    """ArcFace模型用于特征提取"""
    def __init__(self, num_classes=1000, embedding_size=512):
        super(ArcFaceModel, self).__init__()
        # 使用EfficientNet-B0作为基础模型
        from torchvision import models
        self.backbone = models.efficientnet_b0(pretrained=False)
        # 获取特征维度
        self.feature_dim = self.backbone.classifier[1].in_features
        # 替换分类头为特征提取层
        self.backbone.classifier = nn.Identity()
        # 添加特征投影层
        self.projection = nn.Linear(self.feature_dim, embedding_size)
    
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        # 投影到目标维度
        embeddings = self.projection(features)
        # L2归一化
        embeddings = nn.functional.normalize(embeddings, dim=1)
        return embeddings

class CharacterClassifier(nn.Module):
    """角色分类器"""
    def __init__(self, num_classes=1000, embedding_size=512):
        super(CharacterClassifier, self).__init__()
        self.arcface = ArcFaceModel(num_classes=num_classes, embedding_size=embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)
    
    def forward(self, x):
        embeddings = self.arcface(x)
        logits = self.classifier(embeddings)
        return logits, embeddings

class OnlineLearningSystem:
    """在线学习系统"""
    def __init__(self, model_path=None, num_classes=1000, embedding_size=512, device='cpu'):
        self.device = device
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.model = self._load_model(model_path)
        self.character_database = self._initialize_database()
        self.class_to_idx = {}
        self.idx_to_class = {}
    
    def _load_model(self, model_path):
        """加载模型"""
        model = CharacterClassifier(num_classes=self.num_classes, embedding_size=self.embedding_size)
        if model_path and os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"成功加载模型: {model_path}")
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                logger.info("使用随机初始化的模型")
        model = model.to(self.device)
        return model
    
    def _initialize_database(self):
        """初始化特征数据库"""
        # 创建FAISS索引
        index = faiss.IndexFlatL2(self.embedding_size)
        return {
            'index': index,
            'embeddings': [],
            'labels': [],
            'character_names': []
        }
    
    def update_model(self, new_data_dir, epochs=10, batch_size=32, learning_rate=1e-4):
        """更新模型"""
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载新数据
        dataset = CharacterDataset(new_data_dir, transform=transform)
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 检查类别数量是否变化
        new_num_classes = len(self.class_to_idx)
        if new_num_classes != self.num_classes:
            logger.info(f"类别数量变化: {self.num_classes} -> {new_num_classes}")
            self.num_classes = new_num_classes
            # 扩展分类器层
            self._expand_classifier(new_num_classes)
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # 训练模型
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = tqdm(dataloader, desc=f"训练 Epoch {epoch+1}")
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs, embeddings = self.model(images)
                
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
        
        # 更新特征数据库
        self._update_database(dataset)
        
        return epoch_accuracy
    
    def _expand_classifier(self, new_num_classes):
        """扩展分类器层以支持新类别"""
        old_classifier = self.model.classifier
        old_embedding_size = old_classifier.in_features
        
        # 创建新的分类器
        new_classifier = nn.Linear(old_embedding_size, new_num_classes)
        
        # 复制旧权重
        with torch.no_grad():
            min_classes = min(old_classifier.out_features, new_num_classes)
            new_classifier.weight[:min_classes] = old_classifier.weight[:min_classes]
            if old_classifier.bias is not None:
                new_classifier.bias[:min_classes] = old_classifier.bias[:min_classes]
        
        # 替换分类器
        self.model.classifier = new_classifier
        logger.info(f"分类器已扩展到 {new_num_classes} 个类别")
    
    def _update_database(self, dataset):
        """更新特征数据库"""
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="更新特征数据库"):
                images = images.to(self.device)
                _, embeddings = self.model(images)
                
                # 将特征添加到数据库
                embeddings_np = embeddings.cpu().numpy()
                self.character_database['embeddings'].extend(embeddings_np.tolist())
                self.character_database['labels'].extend(labels.tolist())
                
                # 添加角色名称
                for label in labels:
                    character_name = self.idx_to_class.get(int(label), f"unknown_{int(label)}")
                    self.character_database['character_names'].append(character_name)
        
        # 更新FAISS索引
        if self.character_database['embeddings']:
            embeddings_np = np.array(self.character_database['embeddings']).astype('float32')
            self.character_database['index'].reset()
            self.character_database['index'].add(embeddings_np)
            logger.info(f"特征数据库已更新，包含 {len(self.character_database['embeddings'])} 个特征")
    
    def add_new_character(self, character_name, image_paths):
        """添加新角色"""
        # 检查角色是否已存在
        if character_name in self.class_to_idx:
            logger.info(f"角色 {character_name} 已存在")
            return False
        
        # 为新角色分配ID
        new_idx = len(self.class_to_idx)
        self.class_to_idx[character_name] = new_idx
        self.idx_to_class[new_idx] = character_name
        
        # 扩展分类器
        self._expand_classifier(len(self.class_to_idx))
        
        # 创建临时数据目录
        temp_dir = os.path.join('data', 'temp_new_character')
        os.makedirs(os.path.join(temp_dir, character_name), exist_ok=True)
        
        # 复制图像到临时目录
        for i, img_path in enumerate(image_paths):
            if os.path.exists(img_path):
                img_name = f"{character_name}_{i}.jpg"
                import shutil
                shutil.copy(img_path, os.path.join(temp_dir, character_name, img_name))
        
        # 更新模型
        accuracy = self.update_model(temp_dir, epochs=5)
        
        # 清理临时目录
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        logger.info(f"成功添加新角色: {character_name}, 训练准确率: {accuracy:.2f}%")
        return True
    
    def recognize_character(self, image_path, top_k=5):
        """识别角色"""
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        
        # 模型推理
        self.model.eval()
        with torch.no_grad():
            logits, embedding = self.model(image)
            
            # 分类结果
            _, predicted = torch.max(logits, 1)
            predicted_idx = predicted.item()
            predicted_character = self.idx_to_class.get(predicted_idx, f"unknown_{predicted_idx}")
            
            # 特征检索
            if self.character_database['index'].ntotal > 0:
                embedding_np = embedding.cpu().numpy().astype('float32')
                distances, indices = self.character_database['index'].search(embedding_np, top_k)
                
                retrieval_results = []
                for i in range(top_k):
                    if indices[0][i] < len(self.character_database['character_names']):
                        character_name = self.character_database['character_names'][indices[0][i]]
                        retrieval_results.append({
                            'character': character_name,
                            'distance': distances[0][i]
                        })
            else:
                retrieval_results = []
        
        return {
            'classification': predicted_character,
            'retrieval': retrieval_results
        }
    
    def save_model(self, output_path):
        """保存模型"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(self.model.state_dict(), output_path)
        logger.info(f"模型已保存到: {output_path}")
    
    def save_database(self, output_path):
        """保存特征数据库"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存数据库
        database_data = {
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'embeddings': self.character_database['embeddings'],
            'labels': self.character_database['labels'],
            'character_names': self.character_database['character_names']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(database_data, f, ensure_ascii=False, indent=2)
        logger.info(f"特征数据库已保存到: {output_path}")
    
    def load_database(self, database_path):
        """加载特征数据库"""
        if os.path.exists(database_path):
            try:
                with open(database_path, 'r', encoding='utf-8') as f:
                    database_data = json.load(f)
                
                self.class_to_idx = database_data.get('class_to_idx', {})
                self.idx_to_class = database_data.get('idx_to_class', {})
                self.character_database['embeddings'] = database_data.get('embeddings', [])
                self.character_database['labels'] = database_data.get('labels', [])
                self.character_database['character_names'] = database_data.get('character_names', [])
                
                # 更新FAISS索引
                if self.character_database['embeddings']:
                    embeddings_np = np.array(self.character_database['embeddings']).astype('float32')
                    self.character_database['index'].reset()
                    self.character_database['index'].add(embeddings_np)
                
                logger.info(f"特征数据库已加载，包含 {len(self.character_database['embeddings'])} 个特征")
                return True
            except Exception as e:
                logger.error(f"加载特征数据库失败: {e}")
                return False
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='在线学习系统')
    parser.add_argument('--model-path', type=str, default='models/character_classifier.pth', help='模型路径')
    parser.add_argument('--database-path', type=str, default='models/character_database.json', help='特征数据库路径')
    parser.add_argument('--new-data-dir', type=str, default='data/new_characters', help='新数据目录')
    parser.add_argument('--output-model', type=str, default='models/character_classifier_updated.pth', help='更新后模型路径')
    parser.add_argument('--output-database', type=str, default='models/character_database_updated.json', help='更新后特征数据库路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    args = parser.parse_args()
    
    # 初始化在线学习系统
    system = OnlineLearningSystem(
        model_path=args.model_path,
        device=args.device
    )
    
    # 加载特征数据库
    system.load_database(args.database_path)
    
    # 更新模型
    logger.info("开始更新模型...")
    accuracy = system.update_model(
        new_data_dir=args.new_data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    logger.info(f"模型更新完成，准确率: {accuracy:.2f}%")
    
    # 保存模型和数据库
    system.save_model(args.output_model)
    system.save_database(args.output_database)
    
    logger.info("在线学习系统运行完成")

if __name__ == '__main__':
    main()
