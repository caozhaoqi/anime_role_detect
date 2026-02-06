#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型导出为ONNX格式脚本

将训练好的角色分类模型导出为ONNX格式，以优化推理速度和跨平台部署
"""

import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_export')

# 导入torchvision模型
import torchvision.models as models

# 定义模型架构
class CharacterClassifier(nn.Module):
    def __init__(self, num_classes=131):
        super(CharacterClassifier, self).__init__()
        # 使用EfficientNet-B0作为基础模型
        self.backbone = models.efficientnet_b0(pretrained=False)
        # 替换分类头
        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features, 
            num_classes
        )
    
    def forward(self, x):
        return self.backbone(x)

class ModelExporter:
    def __init__(self, model_path, num_classes=131):
        """
        初始化模型导出器
        
        Args:
            model_path: 模型权重文件路径
            num_classes: 类别数量
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
    def _load_model(self):
        """
        加载模型
        """
        logger.info(f"加载模型: {self.model_path}")
        model = CharacterClassifier(num_classes=self.num_classes)
        
        # 加载权重
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            # 处理不同格式的权重文件
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            elif 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
            logger.info("模型权重加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
        
        # 设置为评估模式
        model.eval()
        return model
    
    def export_to_onnx(self, output_path, input_shape=(1, 3, 224, 224)):
        """
        导出模型为ONNX格式
        
        Args:
            output_path: 输出ONNX文件路径
            input_shape: 输入张量形状
        """
        logger.info(f"导出模型为ONNX格式，输入形状: {input_shape}")
        
        # 创建示例输入
        dummy_input = torch.randn(input_shape, device=self.device)
        
        # 导出模型
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
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
            logger.info(f"模型导出成功: {output_path}")
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            raise
    
    def validate_onnx(self, onnx_path, test_image_path=None):
        """
        验证ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
            test_image_path: 测试图像路径
        """
        logger.info(f"验证ONNX模型: {onnx_path}")
        
        try:
            # 安装必要的库
            import onnx
            import onnxruntime
            
            # 检查ONNX模型
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            logger.info("ONNX模型验证通过")
            
            # 创建ONNX运行时会话
            session = onnxruntime.InferenceSession(onnx_path)
            
            # 获取输入和输出名称
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            logger.info(f"输入名称: {input_name}")
            logger.info(f"输出名称: {output_name}")
            
            # 如果提供了测试图像，进行推理测试
            if test_image_path and os.path.exists(test_image_path):
                logger.info(f"使用测试图像: {test_image_path}")
                
                # 图像预处理
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                image = Image.open(test_image_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).numpy()
                
                # 运行推理
                import time
                start_time = time.time()
                outputs = session.run([output_name], {input_name: image_tensor})
                inference_time = time.time() - start_time
                
                logger.info(f"ONNX推理时间: {inference_time:.4f}秒")
                logger.info(f"输出形状: {outputs[0].shape}")
                
                # 与PyTorch模型比较
                with torch.no_grad():
                    torch_input = torch.tensor(image_tensor).to(self.device)
                    torch_start = time.time()
                    torch_outputs = self.model(torch_input)
                    torch_time = time.time() - torch_start
                    
                logger.info(f"PyTorch推理时间: {torch_time:.4f}秒")
                
                # 比较结果
                onnx_output = outputs[0][0]
                torch_output = torch_outputs.cpu().numpy()[0]
                
                # 计算差异
                diff = np.abs(onnx_output - torch_output).max()
                logger.info(f"ONNX与PyTorch输出最大差异: {diff:.6f}")
                
                if diff < 1e-5:
                    logger.info("ONNX模型与PyTorch模型输出一致")
                else:
                    logger.warning("ONNX模型与PyTorch模型输出存在差异")
                    
        except Exception as e:
            logger.error(f"ONNX模型验证失败: {e}")
            raise
    
    def benchmark_inference(self, onnx_path, iterations=100):
        """
        基准测试推理速度
        
        Args:
            onnx_path: ONNX模型路径
            iterations: 测试迭代次数
        """
        logger.info(f"基准测试推理速度，迭代次数: {iterations}")
        
        try:
            import onnxruntime
            
            # 创建ONNX运行时会话
            session = onnxruntime.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            
            # 创建随机输入
            dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
            
            # 预热
            for _ in range(10):
                session.run(None, {input_name: dummy_input})
            
            # 测试
            import time
            start_time = time.time()
            for _ in range(iterations):
                session.run(None, {input_name: dummy_input})
            total_time = time.time() - start_time
            
            avg_time = total_time / iterations
            fps = 1.0 / avg_time
            
            logger.info(f"基准测试完成")
            logger.info(f"总时间: {total_time:.4f}秒")
            logger.info(f"平均推理时间: {avg_time:.4f}秒")
            logger.info(f"帧率: {fps:.2f} FPS")
            
            return {
                'total_time': total_time,
                'avg_time': avg_time,
                'fps': fps
            }
            
        except Exception as e:
            logger.error(f"基准测试失败: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='模型导出为ONNX格式脚本')
    
    parser.add_argument('--model-path', type=str, 
                       default='models/character_classifier_best_improved.pth',
                       help='模型权重文件路径')
    parser.add_argument('--output-path', type=str, 
                       default='models/character_classifier.onnx',
                       help='输出ONNX文件路径')
    parser.add_argument('--num-classes', type=int, default=131,
                       help='类别数量')
    parser.add_argument('--test-image', type=str,
                       default='data/validation_data/honkai_star_rail/火花/honkai_star_rail_火花_0000.jpg',
                       help='测试图像路径')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行基准测试')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 初始化导出器
    exporter = ModelExporter(
        model_path=args.model_path,
        num_classes=args.num_classes
    )
    
    # 导出模型
    exporter.export_to_onnx(args.output_path)
    
    # 验证模型
    exporter.validate_onnx(args.output_path, args.test_image)
    
    # 运行基准测试
    if args.benchmark:
        exporter.benchmark_inference(args.output_path)
    
    logger.info("模型导出和验证完成")

if __name__ == '__main__':
    main()
