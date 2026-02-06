
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边缘设备推理脚本
"""

import os
import json
import numpy as np
import time
from PIL import Image

class AnimeRoleDetector:
    """动漫角色检测器"""
    
    def __init__(self, model_path, config_path):
        """初始化检测器
        
        Args:
            model_path: 模型路径
            config_path: 配置路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.model_path = model_path
        self.class_to_idx = self.config.get('class_to_idx', {})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 加载模型
        self.model = self._load_model()
        
        print(f"初始化完成，支持 {len(self.class_to_idx)} 个角色")
    
    def _load_model(self):
        """加载模型"""
        # 根据模型类型选择加载方式
        model_type = self.config.get('model_type', 'mobilenet_v2')
        
        if self.model_path.endswith('.onnx'):
            # 使用ONNX Runtime
            import onnxruntime
            return onnxruntime.InferenceSession(self.model_path)
        elif self.model_path.endswith('.tflite'):
            # 使用TFLite
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter
        else:
            # 使用PyTorch
            import torch
            import torch.nn as nn
            from torchvision import models
            
            # 获取模型
            if model_type == 'efficientnet_b0':
                model = models.efficientnet_b0(pretrained=False)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.class_to_idx))
            elif model_type == 'mobilenet_v2':
                model = models.mobilenet_v2(pretrained=False)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(self.class_to_idx))
            elif model_type == 'resnet18':
                model = models.resnet18(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, len(self.class_to_idx))
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 加载权重
            device = torch.device('cpu')
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            model.eval()
            return model
    
    def preprocess(self, image):
        """预处理图像
        
        Args:
            image: PIL图像
        
        Returns:
            预处理后的图像
        """
        # 调整大小
        image = image.resize((self.config['preprocessing']['resize'], self.config['preprocessing']['resize']))
        
        # 中心裁剪
        width, height = image.size
        crop_size = self.config['preprocessing']['crop']
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        image = image.crop((left, top, right, bottom))
        
        # 转换为数组
        image = np.array(image).astype(np.float32)
        
        # 归一化
        mean = self.config['preprocessing']['mean']
        std = self.config['preprocessing']['std']
        image = (image / 255.0 - mean) / std
        
        # 调整维度
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = np.expand_dims(image, axis=0)  # 添加批次维度
        
        return image
    
    def predict(self, image_path):
        """预测图像
        
        Args:
            image_path: 图像路径
        
        Returns:
            预测结果
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 预处理
        preprocessed = self.preprocess(image)
        
        # 推理
        start_time = time.time()
        
        try:
            # 尝试ONNX Runtime
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            outputs = self.model.run([output_name], {input_name: preprocessed.astype(np.float32)})
            predictions = outputs[0]
        except AttributeError:
            try:
                # 尝试TFLite
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()
                self.model.set_tensor(input_details[0]['index'], preprocessed.astype(np.float32))
                self.model.invoke()
                predictions = self.model.get_tensor(output_details[0]['index'])
            except AttributeError:
                # 尝试PyTorch
                import torch
                input_tensor = torch.from_numpy(preprocessed).float()
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                predictions = outputs.numpy()
        
        inference_time = (time.time() - start_time) * 1000
        
        # 后处理
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        class_name = self.idx_to_class.get(predicted_class, '未知')
        
        return {
            'class_name': class_name,
            'confidence': float(confidence),
            'inference_time_ms': inference_time,
            'fps': 1000 / inference_time
        }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='边缘设备推理脚本')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--model', type=str, default='models/deployment/mobilenet_v2.onnx', help='模型路径')
    parser.add_argument('--config', type=str, default='models/deployment/config.json', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 初始化检测器
    detector = AnimeRoleDetector(args.model, args.config)
    
    # 预测
    result = detector.predict(args.image)
    
    # 打印结果
    print(f"预测角色: {result['class_name']}")
    print(f"置信度: {result['confidence']:.4f}")
    print(f"推理时间: {result['inference_time_ms']:.2f} ms")
    print(f"推理速度: {result['fps']:.2f} FPS")
