#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型转换脚本：将PyTorch模型转换为Core ML格式
"""

import os
import sys

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import CLIPModel, CLIPProcessor
import coremltools as ct
from core.logging.global_logger import get_logger

logger = get_logger("coreml_converter")


def convert_clip_to_coreml(model_name="openai/clip-vit-base-patch32", output_dir="./coreml_models"):
    """将CLIP模型转换为Core ML格式
    
    Args:
        model_name: CLIP模型名称
        output_dir: 输出目录
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"加载CLIP模型: {model_name}")
        # 加载模型和处理器
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        # 设置模型为评估模式
        model.eval()
        
        # 创建示例输入
        # CLIP模型的get_image_features方法只需要图像输入
        example_input = torch.randn(1, 3, 224, 224)
        
        logger.info("转换模型为Core ML格式...")
        # 转换为Core ML模型
        # 创建一个包装类来处理单个方法
        class ClipImageFeatureWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, image):
                return self.model.get_image_features(image)
        
        # 创建包装模型
        wrapper = ClipImageFeatureWrapper(model)
        
        # 追踪包装模型
        traced_model = torch.jit.trace(wrapper, example_input)
        
        # 创建Core ML模型
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="image", shape=example_input.shape)],
            outputs=[ct.TensorType(name="features")]
        )
        
        # 保存模型
        output_path = os.path.join(output_dir, "clip_model.mlpackage")
        coreml_model.save(output_path)
        logger.info(f"模型保存到: {output_path}")
        
        # 保存处理器配置
        processor_config = {
            "model_name": model_name,
            "image_size": 224  # CLIP模型默认图像大小
        }
        
        import json
        config_path = os.path.join(output_dir, "clip_processor_config.json")
        with open(config_path, 'w') as f:
            json.dump(processor_config, f, indent=2)
        logger.info(f"处理器配置保存到: {config_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"转换模型失败: {e}")
        raise


def main():
    """主函数"""
    try:
        logger.info("开始转换模型...")
        output_path = convert_clip_to_coreml()
        logger.info(f"模型转换完成: {output_path}")
    except Exception as e:
        logger.error(f"转换过程中发生错误: {e}")


if __name__ == "__main__":
    main()
