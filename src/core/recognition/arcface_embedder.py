#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArcFace特征提取器
基于ArcFace损失函数的特征学习，提升角色区分能力

ArcFace特点：
- 在特征和权重之间添加角度间隔
- 增强类内紧凑性
- 增强类间分离性
- 更适合细粒度识别任务
"""

import os
import math
from pathlib import Path
from typing import List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ArcMarginProduct:
    """
    ArcFace的加性角度间隔损失
    
    公式：
    L = -log(e^(s*(cos(theta_yi + m))) / (e^(s*(cos(theta_yi + m))) + sum(e^(s*cos(theta_j)))))
    
    其中：
    - s: 缩放因子
    - m: 角度间隔
    - theta: 特征与权重的角度
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False,
    ):
        """
        初始化
        
        Args:
            in_features: 输入特征维度
            out_features: 输出类别数
            s: 缩放因子
            m: 角度间隔
            easy_margin: 是否使用简单间隔
        """
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        # 初始化权重
        import torch
        import torch.nn as nn
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # 预计算
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, input: "torch.Tensor", label: "torch.Tensor") -> "torch.Tensor":
        """
        前向传播
        
        Args:
            input: 输入特征 (batch_size, in_features)
            label: 标签 (batch_size,)
            
        Returns:
            输出logits
        """
        import torch
        import torch.nn.functional as F
        
        # 归一化特征和权重
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 计算sin和cos
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # 计算phi = cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # one-hot编码
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # 输出
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


class ArcFaceEmbedder:
    """
    ArcFace特征提取器
    
    使用ArcFace训练的特征提取模型，提取 discriminative 特征
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        backbone: str = "resnet50",
        embedding_dim: int = 512,
        device: Optional[str] = None,
    ):
        """
        初始化
        
        Args:
            model_path: 预训练模型路径
            backbone: 骨干网络
            embedding_dim: 嵌入维度
            device: 运行设备
        """
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.device = device or self._select_device()
        self.model = None
        self.model_path = model_path
        
        logger.info(f"ArcFaceEmbedder 初始化: backbone={backbone}, dim={embedding_dim}")
    
    def _select_device(self) -> str:
        """选择设备"""
        import platform
        if platform.system() == "Darwin":
            return "cpu"
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except:
            return "cpu"
    
    def _build_model(self):
        """构建模型"""
        import torch
        import torch.nn as nn
        import torchvision.models as models
        
        # 使用ResNet作为骨干
        if self.backbone == "resnet50":
            base_model = models.resnet50(pretrained=True)
            base_model.fc = nn.Identity()
            feature_dim = 2048
        elif self.backbone == "resnet18":
            base_model = models.resnet18(pretrained=True)
            base_model.fc = nn.Identity()
            feature_dim = 512
        else:
            raise ValueError(f"不支持的骨干网络: {self.backbone}")
        
        # 特征投影层
        self.model = nn.Sequential(
            base_model,
            nn.Linear(feature_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 加载预训练权重
        if self.model_path and os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"加载模型权重: {self.model_path}")
    
    def embed_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Optional[np.ndarray]:
        """
        提取图片特征
        
        Args:
            image_input: 图片输入
            
        Returns:
            归一化特征向量
        """
        if self.model is None:
            self._build_model()
        
        import torch
        import torchvision.transforms as transforms
        
        # 图片预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input).convert("RGB")
            else:
                raise ValueError(f"不支持的输入类型: {type(image_input)}")
            
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = self.model(image_tensor)
            
            # 归一化
            embedding = feature.cpu().numpy().flatten()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"ArcFace特征提取失败: {e}")
            return None
    
    def embed_images(
        self,
        image_inputs: List[Union[str, Image.Image, np.ndarray]],
        batch_size: int = 16,
    ) -> List[Optional[np.ndarray]]:
        """
        批量提取特征
        
        Args:
            image_inputs: 图片列表
            batch_size: 批大小
            
        Returns:
            特征列表
        """
        if self.model is None:
            self._build_model()
        
        import torch
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        results = []
        
        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i:i+batch_size]
            
            try:
                images = []
                for img_input in batch:
                    if isinstance(img_input, str):
                        img = Image.open(img_input).convert("RGB")
                    elif isinstance(img_input, Image.Image):
                        img = img_input.convert("RGB")
                    elif isinstance(img_input, np.ndarray):
                        img = Image.fromarray(img_input).convert("RGB")
                    else:
                        raise ValueError(f"不支持的输入类型: {type(img_input)}")
                    images.append(transform(img))
                
                image_batch = torch.stack(images).to(self.device)
                
                with torch.no_grad():
                    features = self.model(image_batch)
                
                features_np = features.cpu().numpy()
                
                for vec in features_np:
                    vec = vec / (np.linalg.norm(vec) + 1e-8)
                    results.append(vec.astype(np.float32))
                    
            except Exception as e:
                logger.error(f"批量特征提取失败: {e}")
                results.extend([None] * len(batch))
        
        return results


class ArcFaceTrainer:
    """
    ArcFace训练器
    训练ArcFace特征提取模型
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        backbone: str = "resnet50",
        s: float = 30.0,
        m: float = 0.50,
        device: Optional[str] = None,
    ):
        """
        初始化训练器
        
        Args:
            num_classes: 类别数
            embedding_dim: 嵌入维度
            backbone: 骨干网络
            s: ArcFace缩放因子
            m: ArcFace角度间隔
            device: 运行设备
        """
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.device = device or self._select_device()
        
        import torch
        import torch.nn as nn
        import torchvision.models as models
        
        # 构建骨干网络
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Identity()
            feature_dim = 2048
        elif backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Identity()
            feature_dim = 512
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        self.backbone = self.backbone.to(self.device)
        
        # 特征投影层
        self.embedding_layer = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        ).to(self.device)
        
        # ArcFace损失层
        self.arc_margin = ArcMarginProduct(
            embedding_dim, num_classes, s=s, m=m
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam([
            {"params": self.backbone.parameters(), "lr": 1e-4},
            {"params": self.embedding_layer.parameters(), "lr": 1e-3},
            {"params": self.arc_margin.parameters(), "lr": 1e-3},
        ])
        
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"ArcFaceTrainer 初始化完成: {num_classes}类, {embedding_dim}维")
    
    def _select_device(self) -> str:
        """选择设备"""
        import platform
        if platform.system() == "Darwin":
            return "cpu"
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except:
            return "cpu"
    
    def train_step(
        self,
        images: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> float:
        """
        单步训练
        
        Args:
            images: 图片张量 (batch_size, 3, 224, 224)
            labels: 标签 (batch_size,)
            
        Returns:
            损失值
        """
        import torch
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        
        # 前向传播
        features = self.backbone(images)
        embeddings = self.embedding_layer(features)
        logits = self.arc_margin(embeddings, labels)
        
        # 计算损失
        loss = self.criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def extract_features(self, images: "torch.Tensor") -> np.ndarray:
        """
        提取特征（推理模式）
        
        Args:
            images: 图片张量
            
        Returns:
            特征数组
        """
        import torch
        
        images = images.to(self.device)
        
        with torch.no_grad():
            features = self.backbone(images)
            embeddings = self.embedding_layer(features)
        
        embeddings = embeddings.cpu().numpy()
        
        # 归一化
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        return embeddings.astype(np.float32)
    
    def save_model(self, path: str):
        """保存模型"""
        import torch
        
        torch.save({
            "backbone": self.backbone.state_dict(),
            "embedding": self.embedding_layer.state_dict(),
            "arc_margin": self.arc_margin.state_dict(),
        }, path)
        
        logger.info(f"模型已保存: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        import torch
        
        checkpoint = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint["backbone"])
        self.embedding_layer.load_state_dict(checkpoint["embedding"])
        self.arc_margin.load_state_dict(checkpoint["arc_margin"])
        
        logger.info(f"模型已加载: {path}")


if __name__ == "__main__":
    # 测试
    embedder = ArcFaceEmbedder()
    
    if os.path.exists("test.jpg"):
        feature = embedder.embed_image("test.jpg")
        print(f"特征维度: {feature.shape}")
        print(f"特征范数: {np.linalg.norm(feature)}")
