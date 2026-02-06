#!/usr/bin/env python3
"""
ArcFace损失函数实现
用于度量学习，增强模型的特征提取能力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    """
    ArcFace损失函数
    参考论文: Additive Angular Margin Loss for Deep Face Recognition
    """
    
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        """
        初始化ArcFace损失函数
        
        Args:
            in_features: 输入特征维度
            out_features: 输出类别数
            s: 特征缩放因子
            m: 角度边际
        """
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # 创建权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, labels):
        """
        前向传播
        
        Args:
            features: 输入特征，形状为 [batch_size, in_features]
            labels: 标签，形状为 [batch_size]
            
        Returns:
            loss: ArcFace损失
        """
        # 归一化特征和权重
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        # 计算余弦相似度
        cos_theta = F.linear(features, weight)
        
        # 计算角度边际
        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        
        # 为目标类别添加角度边际
        target_margin = torch.cos(theta + self.m)
        
        # 创建掩码
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        # 组合损失
        logits = one_hot * target_margin + (1.0 - one_hot) * cos_theta
        logits *= self.s
        
        # 使用交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss

class CosFaceLoss(nn.Module):
    """
    CosFace损失函数
    参考论文: CosFace: Large Margin Cosine Loss for Deep Face Recognition
    """
    
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        """
        初始化CosFace损失函数
        
        Args:
            in_features: 输入特征维度
            out_features: 输出类别数
            s: 特征缩放因子
            m: 余弦边际
        """
        super(CosFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # 创建权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, labels):
        """
        前向传播
        
        Args:
            features: 输入特征，形状为 [batch_size, in_features]
            labels: 标签，形状为 [batch_size]
            
        Returns:
            loss: CosFace损失
        """
        # 归一化特征和权重
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=1)
        
        # 计算余弦相似度
        cos_theta = F.linear(features, weight)
        
        # 为目标类别添加余弦边际
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        # 组合损失
        logits = one_hot * (cos_theta - self.m) + (1.0 - one_hot) * cos_theta
        logits *= self.s
        
        # 使用交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss

class SoftmaxLoss(nn.Module):
    """
    标准Softmax损失函数
    作为对比基准
    """
    
    def __init__(self, in_features, out_features):
        """
        初始化Softmax损失函数
        
        Args:
            in_features: 输入特征维度
            out_features: 输出类别数
        """
        super(SoftmaxLoss, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, features, labels):
        """
        前向传播
        
        Args:
            features: 输入特征，形状为 [batch_size, in_features]
            labels: 标签，形状为 [batch_size]
            
        Returns:
            loss: Softmax损失
        """
        logits = self.fc(features)
        loss = F.cross_entropy(logits, labels)
        return loss
