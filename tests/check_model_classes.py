#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查模型文件中的类别信息
"""

import os
import torch

# 检查模型文件
model_paths = [
    'models/arona_plana/model_best.pth',
    'models/arona_plana_resnet18/model_best.pth',
    'models/augmented_training/mobilenet_v2/model_best.pth'
]

for model_path in model_paths:
    if os.path.exists(model_path):
        print(f"\n检查模型: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'class_to_idx' in checkpoint:
                class_to_idx = checkpoint['class_to_idx']
                print(f"类别数量: {len(class_to_idx)}")
                print("类别映射:")
                for cls, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
                    print(f"  {cls}: {idx}")
            else:
                print("模型中没有 class_to_idx 信息")
            
            if 'model_type' in checkpoint:
                print(f"模型类型: {checkpoint['model_type']}")
        except Exception as e:
            print(f"读取模型失败: {e}")
    else:
        print(f"模型文件不存在: {model_path}")
