#!/usr/bin/env python3
"""
创建NSFW模型权重文件
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models

# 标签顺序
LABELS = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

# 创建模型
def create_nsfw_model():
    """创建NSFW检测模型"""
    # 加载预训练的MobileNet V2模型
    model = models.mobilenet_v2(pretrained=True)
    
    # 修改分类器以适应5个类别
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(LABELS))
    
    return model

# 保存模型权重
def save_model_weights():
    """保存模型权重"""
    # 创建模型
    model = create_nsfw_model()
    
    # 定义保存路径
    model_path = os.path.join('models', 'nsfw_model', 'nsfw_model.pth')
    model_path = os.path.normpath(model_path)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 保存权重
    torch.save(model.state_dict(), model_path)
    print(f"模型权重已保存到: {model_path}")

if __name__ == "__main__":
    save_model_weights()
