#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型定义
"""

import torch
import torch.nn as nn
from torchvision import models


class CharacterAttributeModel(nn.Module):
    """带有属性预测分支的角色分类模型"""
    
    def __init__(self, base_model_type='mobilenet_v2', num_classes=5, num_attributes=6):
        super().__init__()
        
        # 加载基础模型
        if base_model_type == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(pretrained=True)
            self.feature_dim = self.base_model.classifier[1].in_features
            # 移除原始分类器
            self.base_model.classifier = nn.Identity()
        elif base_model_type == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=True)
            self.feature_dim = self.base_model.classifier[1].in_features
            # 移除原始分类器
            self.base_model.classifier = nn.Identity()
        elif base_model_type == 'resnet18':
            self.base_model = models.resnet18(pretrained=True)
            self.feature_dim = self.base_model.fc.in_features
            # 移除原始分类器
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f"不支持的模型类型: {base_model_type}")
        
        # 分类分支
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        # 属性预测分支
        self.attribute_classifier = nn.Linear(self.feature_dim, num_attributes)
    
    def forward(self, x):
        # 获取特征
        if hasattr(self.base_model, 'features'):
            # MobileNetV2 和 EfficientNet
            features = self.base_model.features(x)
            features = features.mean([2, 3])  # 全局平均池化
        else:
            # ResNet
            features = self.base_model(x)
        
        # 分类预测
        class_output = self.classifier(features)
        # 属性预测
        attribute_output = self.attribute_classifier(features)
        
        return class_output, attribute_output


def get_model_with_attributes(model_type, num_classes, num_attributes):
    """获取带有属性预测分支的模型"""
    return CharacterAttributeModel(model_type, num_classes, num_attributes)