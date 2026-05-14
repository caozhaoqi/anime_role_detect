#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试PyTorch在macOS上的初始化
"""

import os
import sys

# 设置环境变量
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPS_HIGH_WATERMARK_RATIO'] = '0.0'

print("测试PyTorch初始化...")

try:
    import torch
    print(f"✓ PyTorch导入成功，版本: {torch.__version__}")
    
    # 检查设备
    if torch.backends.mps.is_available():
        print("✓ MPS设备可用")
        device = torch.device("mps")
        print(f"✓ 当前设备: {device}")
    elif torch.cuda.is_available():
        print("✓ CUDA设备可用")
        device = torch.device("cuda")
        print(f"✓ 当前设备: {device}")
    else:
        print("✓ 使用CPU设备")
        device = torch.device("cpu")
    
    # 测试简单操作
    x = torch.randn(3, 3).to(device)
    print(f"✓ 张量操作成功: {x.shape}")
    
    # 测试CLIP模型导入
    try:
        from transformers import CLIPProcessor, CLIPModel
        print("✓ CLIP模型导入成功")
        
        # 加载模型
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✓ CLIP模型加载成功")
        
        # 测试推理
        from PIL import Image
        import io
        
        # 创建一个简单的测试图像
        img = Image.new('RGB', (224, 224), color='red')
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model.get_image_features(**inputs)
        print(f"✓ CLIP推理成功，特征维度: {outputs.shape}")
        
    except Exception as e:
        print(f"✗ CLIP模型测试失败: {e}")
        
except Exception as e:
    print(f"✗ PyTorch初始化失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成")