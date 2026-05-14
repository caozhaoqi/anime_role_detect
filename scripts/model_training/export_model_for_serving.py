#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 H5 模型转换为 SavedModel 格式，用于 TensorFlow Serving
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model

# 模型路径
model_path = os.path.join(os.path.dirname(__file__), '..', 'nsfw_model_img', 'src', 'main', 'resources', 'mobilenet_v2_140_224', 'saved_model.h5')
model_path = os.path.normpath(model_path)

# 输出路径
export_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'nsfw_model', '1')
export_path = os.path.normpath(export_path)

print(f"加载模型: {model_path}")
print(f"导出路径: {export_path}")

# 确保输出目录存在
os.makedirs(export_path, exist_ok=True)

# 加载模型
model = load_model(model_path)
print("模型加载成功")

# 查看模型输入和输出
print("\n模型输入:")
for input_layer in model.inputs:
    print(f"  - {input_layer.name}: {input_layer.shape}")

print("\n模型输出:")
for output_layer in model.outputs:
    print(f"  - {output_layer.name}: {output_layer.shape}")

# 导出为 SavedModel
tf.saved_model.save(model, export_path)
print("\n模型导出成功!")

# 验证导出结果
print("\n验证导出结果:")
try:
    loaded_model = tf.saved_model.load(export_path)
    print("模型加载成功")
    
    # 检查签名
    signatures = list(loaded_model.signatures.keys())
    print(f"可用签名: {signatures}")
    
    if 'serving_default' in signatures:
        print("serving_default 签名存在")
        
        # 查看输入输出信息
        serving_default = loaded_model.signatures['serving_default']
        print("\n输入信息:")
        for name, tensor in serving_default.inputs.items():
            print(f"  - {name}: {tensor.shape}")
        
        print("\n输出信息:")
        for name, tensor in serving_default.outputs.items():
            print(f"  - {name}: {tensor.shape}")
    else:
        print("serving_default 签名不存在")
except Exception as e:
    print(f"验证失败: {e}")
