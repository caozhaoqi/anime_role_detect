#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WD Vit Tagger 模型转换脚本：将 PyTorch 模型转换为 Core ML 格式
"""

import os
import sys
import json
import torch
from transformers import AutoModelForImageClassification, AutoProcessor
import coremltools as ct

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 使用全局日志系统
from core.logging.global_logger import get_logger
logger = get_logger("wd_tagger_coreml_converter")


def convert_wd_tagger_to_coreml(model_name="SmilingWolf/wd-vit-tagger-v3", output_dir="./coreml_models"):
    """将 WD Vit Tagger v3 模型转换为 Core ML 格式
    
    Args:
        model_name: WD Vit Tagger 模型名称
        output_dir: 输出目录
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"加载 WD Vit Tagger 模型: {model_name}")
        # 加载模型和处理器
        model = AutoModelForImageClassification.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        # 设置模型为评估模式
        model.eval()
        
        # 创建包装类来处理模型的 forward 方法
        class WDTaggerWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, image):
                output = self.model(image)
                return output.logits
        
        # 创建包装模型
        wrapper = WDTaggerWrapper(model)
        
        # 创建示例输入
        # WD Vit Tagger 模型输入为图像，形状为 [batch_size, channels, height, width]
        example_input = torch.randn(1, 3, 448, 448)
        
        logger.info("转换模型为 Core ML 格式...")
        # 转换为 Core ML 模型
        traced_model = torch.jit.trace(wrapper, example_input)
        
        # 创建 Core ML 模型
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="image", shape=example_input.shape)],
            outputs=[ct.TensorType(name="logits")]
        )
        
        # 保存模型
        output_path = os.path.join(output_dir, "wd_tagger.mlpackage")
        coreml_model.save(output_path)
        logger.info(f"模型保存到: {output_path}")
        
        # 保存处理器配置
        processor_config = {
            "model_name": model_name,
            "image_size": 448  # WD Vit Tagger 默认图像大小
        }
        
        config_path = os.path.join(output_dir, "wd_tagger_config.json")
        with open(config_path, 'w') as f:
            json.dump(processor_config, f, indent=2)
        logger.info(f"处理器配置保存到: {config_path}")
        
        # 保存标签映射
        id2label = model.config.id2label
        label_path = os.path.join(output_dir, "wd_tagger_labels.json")
        with open(label_path, 'w') as f:
            json.dump(id2label, f, indent=2)
        logger.info(f"标签映射保存到: {label_path}")
        
        return output_path
    except Exception as e:
        logger.error(f"转换模型失败: {e}")
        raise


def main():
    """主函数"""
    try:
        logger.info("开始转换 WD Vit Tagger 模型...")
        output_path = convert_wd_tagger_to_coreml()
        logger.info(f"模型转换完成: {output_path}")
    except Exception as e:
        logger.error(f"转换过程中发生错误: {e}")


if __name__ == "__main__":
    main()
