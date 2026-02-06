#!/usr/bin/env python3
"""
Grad-CAM可视化实现
用于分析模型在做决策时关注的图像区域
"""
import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

class GradCAM:
    """Grad-CAM实现"""
    
    def __init__(self, model, target_layer):
        """初始化Grad-CAM
        
        Args:
            model: 要分析的模型
            target_layer: 目标卷积层
        """
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向传播钩子"""
        
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        # 为目标层注册钩子
        target_found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                target_found = True
                break
        
        if not target_found:
            raise ValueError(f"目标层 {self.target_layer} 未找到")
    
    def generate_cam(self, image, target_class=None):
        """生成CAM热力图
        
        Args:
            image: 输入图像
            target_class: 目标类别索引，如果为None则使用模型预测的类别
            
        Returns:
            cam: CAM热力图
            predicted_class: 预测的类别索引
            confidence: 预测的置信度
        """
        # 前向传播
        output = self.model(image)
        
        # 获取预测类别
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        confidence = F.softmax(output, dim=1)[0, target_class].item()
        
        # 清零梯度
        self.model.zero_grad()
        
        # 反向传播到目标类别
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算梯度权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 生成CAM
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze()
        cam = F.relu(cam)
        
        # 归一化
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        return cam.cpu().numpy(), target_class, confidence

def visualize_cam(image_path, cam, predicted_class, confidence, class_names=None, output_path=None):
    """可视化CAM热力图
    
    Args:
        image_path: 原始图像路径
        cam: CAM热力图
        predicted_class: 预测的类别索引
        confidence: 预测的置信度
        class_names: 类别名称列表
        output_path: 输出图像路径
    """
    # 加载原始图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 调整CAM大小以匹配原始图像
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    
    # 转换为热力图
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    
    # 叠加热力图到原始图像
    overlay = cv2.addWeighted(image, 0.6, cam, 0.4, 0)
    
    # 添加预测信息
    if class_names and predicted_class < len(class_names):
        class_name = class_names[predicted_class]
    else:
        class_name = f"Class {predicted_class}"
    
    # 在图像上添加文本
    text = f"Prediction: {class_name} ({confidence:.2f})"
    overlay = cv2.putText(
        overlay, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    
    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, overlay)
        print(f"Grad-CAM可视化结果已保存到: {output_path}")
    
    return overlay

def load_model(model_path):
    """加载模型
    
    Args:
        model_path: 模型权重路径
        
    Returns:
        model: 加载好的模型
        class_names: 类别名称列表
    """
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 获取类别信息
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        class_names = None
    
    # 创建模型
    model = efficientnet_b0(pretrained=False)
    
    # 获取分类器输入特征数
    num_ftrs = model.classifier[1].in_features
    
    # 调整分类器
    if class_names:
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 处理键名不匹配
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            name = k[9:]  # 移除 'backbone.'
        else:
            name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model, class_names

def preprocess_image(image_path):
    """预处理图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        image_tensor: 预处理后的图像张量
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Grad-CAM可视化工具')
    
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--model', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--target-layer', type=str, default='features.8.0',
                       help='目标卷积层')
    parser.add_argument('--output', type=str, default='output/grad_cam_result.jpg',
                       help='输出图像路径')
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model, class_names = load_model(args.model)
    
    # 预处理图像
    print(f"预处理图像: {args.image}")
    image_tensor = preprocess_image(args.image)
    
    # 初始化Grad-CAM
    print(f"初始化Grad-CAM，目标层: {args.target_layer}")
    grad_cam = GradCAM(model, args.target_layer)
    
    # 生成CAM
    print("生成CAM热力图...")
    cam, predicted_class, confidence = grad_cam.generate_cam(image_tensor)
    
    # 可视化
    print("可视化结果...")
    visualize_cam(
        args.image, cam, predicted_class, confidence,
        class_names=class_names, output_path=args.output
    )
    
    print("\n分析完成!")
    if class_names:
        print(f"预测类别: {class_names[predicted_class]}")
    else:
        print(f"预测类别索引: {predicted_class}")
    print(f"置信度: {confidence:.4f}")
    print(f"结果保存到: {args.output}")

if __name__ == '__main__':
    main()
