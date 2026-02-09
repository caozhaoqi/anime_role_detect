#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型转换脚本

将PyTorch模型转换为ONNX格式，便于部署
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
from torchvision import models
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_converter')


class ModelConverter:
    def __init__(self, model_path, output_dir='models/onnx'):
        """
        初始化模型转换器
        
        Args:
            model_path: 模型路径
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型
        self.checkpoint = torch.load(model_path, map_location='cpu')
        self.class_to_idx = self.checkpoint.get('class_to_idx', {})
        
        # 获取类别数量
        self.num_classes = len(self.class_to_idx) if self.class_to_idx else 47
        
        logger.info(f"加载模型: {model_path}")
        logger.info(f"类别数量: {self.num_classes}")
    
    def create_model(self):
        """
        创建模型结构
        
        Returns:
            模型
        """
        # 检测模型结构类型
        classifier_keys = [k for k in self.checkpoint['model_state_dict'].keys() if 'classifier' in k]
        
        if len(classifier_keys) == 2:  # 简单结构: Linear(1280 -> 47)
            class SimpleEfficientNet(nn.Module):
                def __init__(self, num_classes=47):
                    super().__init__()
                    self.backbone = models.efficientnet_b0(pretrained=False)
                    self.backbone.classifier = nn.Linear(1280, num_classes)
                
                def forward(self, x):
                    return self.backbone(x)
            
            model = SimpleEfficientNet(num_classes=self.num_classes)
            
        else:  # 复杂结构: Linear(1280 -> 512) -> ReLU -> Dropout -> Linear(512 -> 256) -> ReLU -> Dropout -> Linear(256 -> 47)
            class ComplexEfficientNet(nn.Module):
                def __init__(self, num_classes=47, dropout_rate=0.3):
                    super().__init__()
                    self.backbone = models.efficientnet_b0(pretrained=False)
                    self.backbone.classifier = nn.Sequential(
                        nn.Linear(1280, 512),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(256, num_classes)
                    )
                
                def forward(self, x):
                    return self.backbone(x)
            
            model = ComplexEfficientNet(num_classes=self.num_classes)
        
        # 加载权重
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def convert_to_onnx(self, model, output_name, input_size=(1, 3, 224, 224)):
        """
        将模型转换为ONNX格式
        
        Args:
            model: PyTorch模型
            output_name: 输出文件名
            input_size: 输入尺寸
        """
        logger.info(f"开始转换模型到ONNX格式...")
        
        # 创建示例输入
        dummy_input = torch.randn(input_size)
        
        # 定义输出路径
        onnx_path = os.path.join(self.output_dir, output_name)
        
        # 转换模型
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"模型已成功转换为ONNX格式: {onnx_path}")
            
            # 验证ONNX模型
            self.validate_onnx_model(onnx_path, dummy_input)
            
            return onnx_path
            
        except Exception as e:
            logger.error(f"转换失败: {str(e)}")
            raise
    
    def validate_onnx_model(self, onnx_path, dummy_input):
        """
        验证ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
            dummy_input: 示例输入
        """
        try:
            import onnx
            import onnxruntime as ort
            
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            
            # 检查模型
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX模型验证通过")
            
            # 使用ONNX Runtime进行推理测试
            ort_session = ort.InferenceSession(onnx_path)
            
            # 获取输入输出名称
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            
            # 进行推理
            ort_inputs = {input_name: dummy_input.numpy()}
            ort_outputs = ort_session.run([output_name], ort_inputs)
            
            logger.info(f"ONNX Runtime推理测试成功，输出形状: {ort_outputs[0].shape}")
            
        except ImportError:
            logger.warning("未安装onnx或onnxruntime，跳过验证")
        except Exception as e:
            logger.warning(f"ONNX模型验证失败: {str(e)}")
    
    def save_class_mapping(self, output_name='class_mapping.json'):
        """
        保存类别映射
        
        Args:
            output_name: 输出文件名
        """
        mapping_path = os.path.join(self.output_dir, output_name)
        
        # 创建索引到类别的映射
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        mapping_data = {
            'class_to_idx': self.class_to_idx,
            'idx_to_class': idx_to_class,
            'num_classes': self.num_classes
        }
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"类别映射已保存: {mapping_path}")
    
    def convert_model(self, output_name=None):
        """
        完整的模型转换流程
        
        Args:
            output_name: 输出文件名（可选）
        """
        # 创建模型
        model = self.create_model()
        
        # 生成输出文件名
        if output_name is None:
            base_name = os.path.splitext(os.path.basename(self.model_path))[0]
            output_name = f"{base_name}.onnx"
        
        # 转换为ONNX
        onnx_path = self.convert_to_onnx(model, output_name)
        
        # 保存类别映射
        self.save_class_mapping()
        
        logger.info("模型转换完成！")
        
        return onnx_path


def main():
    parser = argparse.ArgumentParser(description='模型转换脚本')
    
    parser.add_argument('--model-path', type=str, 
                       default='models/character_classifier_optimized_best.pth',
                       help='PyTorch模型路径')
    parser.add_argument('--output-dir', type=str, 
                       default='models/onnx',
                       help='ONNX模型输出目录')
    parser.add_argument('--output-name', type=str, 
                       default=None,
                       help='ONNX模型输出文件名')
    parser.add_argument('--input-size', type=int, nargs=4,
                       default=[1, 3, 224, 224],
                       help='输入尺寸 (batch, channels, height, width)')
    
    args = parser.parse_args()
    
    # 初始化转换器
    converter = ModelConverter(args.model_path, args.output_dir)
    
    # 转换模型
    input_size = tuple(args.input_size)
    onnx_path = converter.convert_model(args.output_name)
    
    logger.info(f"ONNX模型已保存到: {onnx_path}")


if __name__ == '__main__':
    main()