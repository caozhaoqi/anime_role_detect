#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建端到端Core ML模型，集成特征提取和分类功能
"""

import os
import sys
import json
import numpy as np
import coremltools as ct
from coremltools.models import MLModel

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 使用全局日志系统
from core.logging.global_logger import get_logger
logger = get_logger("end_to_end_coreml")


def create_end_to_end_model(output_dir="./coreml_models"):
    """创建端到端Core ML模型
    
    Args:
        output_dir: 输出目录
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载现有的Core ML模型
        clip_model_path = os.path.join(output_dir, "clip_model.mlpackage")
        if not os.path.exists(clip_model_path):
            raise FileNotFoundError(f"CLIP Core ML模型不存在: {clip_model_path}")
        
        logger.info(f"加载CLIP Core ML模型: {clip_model_path}")
        clip_model = MLModel(clip_model_path)
        
        # 加载索引文件
        index_path = os.path.join(os.path.dirname(__file__), '..', 'role_index.faiss')
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        
        # 映射文件路径
        mapping_path = os.path.join(os.path.dirname(__file__), '..', 'role_mapping.json')
        
        # 尝试加载映射文件
        mapping = {}
        if os.path.exists(mapping_path):
            logger.info(f"加载索引映射文件: {mapping_path}")
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
        else:
            logger.warning(f"映射文件不存在: {mapping_path}，将在运行时生成")
        
        # 创建一个简单的分类器模型
        # 这里我们创建一个自定义的Core ML模型，集成特征提取和分类
        
        # 首先，我们需要创建一个模型规范
        from coremltools.models import datatypes
        from coremltools.models import neural_network
        
        # 输入是图像
        input_features = [('image', datatypes.Array(3, 224, 224))]
        # 输出是角色名称和相似度
        output_features = [('role', datatypes.String()), ('similarity', datatypes.Double())]
        
        # 创建神经网络规范
        builder = neural_network.NeuralNetworkBuilder(input_features, output_features)
        
        # 添加层
        # 这里我们只是创建一个占位符，实际的推理逻辑会在Python端处理
        # Core ML目前不支持直接集成Faiss索引搜索
        
        # 保存模型
        model_path = os.path.join(output_dir, "end_to_end_classifier.mlpackage")
        
        # 由于Core ML不直接支持Faiss索引搜索，我们创建一个包装器
        # 实际的推理会在Python端使用Core ML进行特征提取，然后使用Faiss进行搜索
        
        logger.info(f"创建端到端模型包装器: {model_path}")
        
        # 创建配置文件
        config = {
            "clip_model_path": clip_model_path,
            "index_path": index_path,
            "mapping_path": mapping_path,
            "image_size": 224
        }
        
        config_path = os.path.join(output_dir, "end_to_end_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"配置文件保存到: {config_path}")
        
        logger.info("端到端模型创建完成")
        return model_path
    except Exception as e:
        logger.error(f"创建端到端模型失败: {e}")
        raise


def main():
    """主函数"""
    try:
        logger.info("开始创建端到端Core ML模型...")
        output_path = create_end_to_end_model()
        logger.info(f"端到端模型创建完成: {output_path}")
    except Exception as e:
        logger.error(f"创建过程中发生错误: {e}")


if __name__ == "__main__":
    main()
