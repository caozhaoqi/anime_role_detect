#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn as nn
from torchvision import models


class CharacterAttributeModel(nn.Module):
    """"""
    
    def __init__(self, base_model_type='mobilenet_v2', num_classes=5, num_attributes=6):
        super().__init__()
        
        # 
        if base_model_type == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(pretrained=True)
            self.feature_dim = self.base_model.classifier[1].in_features
            # 
            self.base_model.classifier = nn.Identity()
        elif base_model_type == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=True)
            self.feature_dim = self.base_model.classifier[1].in_features
            # 
            self.base_model.classifier = nn.Identity()
        elif base_model_type == 'resnet18':
            self.base_model = models.resnet18(pretrained=True)
            self.feature_dim = self.base_model.fc.in_features
            # 
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f": {base_model_type}")
        
        # 
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        # 
        self.attribute_classifier = nn.Linear(self.feature_dim, num_attributes)
    
    def forward(self, x):
        # 
        if hasattr(self.base_model, 'features'):
            # MobileNetV2  EfficientNet
            features = self.base_model.features(x)
            features = features.mean([2, 3])  # 
        else:
            # ResNet
            features = self.base_model(x)
        
        # 
        class_output = self.classifier(features)
        # 
        attribute_output = self.attribute_classifier(features)
        
        return class_output, attribute_output


def get_model_with_attributes(model_type, num_classes, num_attributes):
    """"""
    return CharacterAttributeModel(model_type, num_classes, num_attributes)