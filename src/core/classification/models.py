#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色分类模型定义
"""

import torch
import torch.nn as nn
from torchvision import models


def get_base_model(model_type):
    """获取基础模型
    
    Args:
        model_type: 模型类型
    
    Returns:
        tuple: (基础模型, 特征维度)
    """
    if model_type == 'mobilenet_v2':
        base_model = models.mobilenet_v2(pretrained=True)
        feature_dim = base_model.classifier[1].in_features
        # 移除分类层
        base_model.classifier = nn.Identity()
    elif model_type == 'efficientnet_b0':
        base_model = models.efficientnet_b0(pretrained=True)
        feature_dim = base_model.classifier[1].in_features
        # 移除分类层
        base_model.classifier = nn.Identity()
    elif model_type == 'resnet18':
        base_model = models.resnet18(pretrained=True)
        feature_dim = base_model.fc.in_features
        # 移除分类层
        base_model.fc = nn.Identity()
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return base_model, feature_dim


class CharacterClassifier(nn.Module):
    """角色分类模型"""
    def __init__(self, model_type, num_classes):
        super(CharacterClassifier, self).__init__()
        self.base_model, self.feature_dim = get_base_model(model_type)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
    
    def forward(self, x):
        features = self.base_model(x)
        logits = self.classifier(features)
        return logits


class CharacterAttributeModel(nn.Module):
    """带有属性预测的角色分类模型"""
    def __init__(self, model_type, num_classes, num_attributes):
        super(CharacterAttributeModel, self).__init__()
        self.base_model, self.feature_dim = get_base_model(model_type)
        
        # 分类头
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # 属性预测头
        self.attribute_predictor = nn.Linear(self.feature_dim, num_attributes)
    
    def forward(self, x):
        features = self.base_model(x)
        class_logits = self.classifier(features)
        attribute_logits = self.attribute_predictor(features)
        return class_logits, attribute_logits


def get_model(model_type, num_classes):
    """获取分类模型
    
    Args:
        model_type: 模型类型
        num_classes: 类别数
    
    Returns:
        CharacterClassifier: 分类模型
    """
    return CharacterClassifier(model_type, num_classes)


def get_model_with_attributes(model_type, num_classes, num_attributes):
    """获取带有属性预测的模型
    
    Args:
        model_type: 模型类型
        num_classes: 类别数
        num_attributes: 属性数
    
    Returns:
        CharacterAttributeModel: 带属性预测的模型
    """
    return CharacterAttributeModel(model_type, num_classes, num_attributes)