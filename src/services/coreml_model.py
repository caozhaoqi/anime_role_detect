#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreML 模型包装器
提供 CoreML 模型的加载和推理功能
"""

import os
import sys
import platform
import numpy as np
from PIL import Image

# 使用全局日志系统
from src.core.logging.global_logger import get_logger
logger = get_logger("coreml_model")

# CoreML 模型实例
coreml_model = None

# 检查是否在 Mac 平台
IS_MAC = platform.system() == 'Darwin'

def initialize_coreml_async():
    """异步初始化 CoreML 模型"""
    if not IS_MAC:
        logger.info("非 Mac 平台，跳过 CoreML 模型初始化")
        return False

    try:
        import asyncio
        import coremltools as ct

        # 查找 CoreML 模型文件
        model_paths = [
            "./coreml_models/character_classifier_best_improved.mlpackage",
            "./models/character_classifier_best_improved.mlpackage",
            "./coreml_models/end_to_end_model.mlpackage",
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if not model_path:
            logger.warning("未找到 CoreML 模型文件，CoreML 功能将不可用")
            return False

        logger.info(f"加载 CoreML 模型: {model_path}")
        coreml_model = ct.models.MLModel(model_path)
        logger.info("CoreML 模型加载成功")
        return True

    except ImportError:
        logger.warning("coremltools 未安装，CoreML 功能将不可用")
        return False
    except Exception as e:
        logger.error(f"CoreML 模型初始化失败: {e}")
        return False

def initialize_coreml_model():
    """初始化 CoreML 模型（同步版本，仅用于非异步环境）"""
    global coreml_model

    if not IS_MAC:
        logger.info("非 Mac 平台，跳过 CoreML 模型初始化")
        return False

    try:
        import coremltools as ct

        # 查找 CoreML 模型文件
        model_paths = [
            "./coreml_models/character_classifier_best_improved.mlpackage",
            "./models/character_classifier_best_improved.mlpackage",
            "./coreml_models/end_to_end_model.mlpackage",
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if not model_path:
            logger.warning("未找到 CoreML 模型文件，CoreML 功能将不可用")
            return False

        logger.info(f"加载 CoreML 模型: {model_path}")
        coreml_model = ct.models.MLModel(model_path)
        logger.info("CoreML 模型加载成功")
        return True

    except ImportError:
        logger.warning("coremltools 未安装，CoreML 功能将不可用")
        return False
    except Exception as e:
        logger.error(f"CoreML 模型初始化失败: {e}")
        return False

def classify_with_coreml(image_path):
    """使用 CoreML 模型分类图像
    
    Args:
        image_path: 图像路径
    
    Returns:
        (role, similarity, boxes): 角色名称、相似度、边界框
    """
    global coreml_model
    
    if coreml_model is None:
        logger.warning("CoreML 模型未初始化")
        raise Exception("CoreML 模型未初始化")
    
    try:
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        
        # 预处理图像
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = (img_array / 255.0 - 0.5) * 2.0
        
        # 使用 CoreML 模型推理
        logger.info("使用 CoreML 模型推理")
        output = coreml_model.predict({'input': img_array})
        
        # 获取预测结果
        if 'var_874' in output:
            predictions = output['var_874']
        elif 'output' in output:
            predictions = output['output']
        else:
            # 尝试找到输出键
            output_keys = [k for k in output.keys() if k != 'input']
            if output_keys:
                predictions = output[output_keys[0]]
            else:
                logger.error("无法找到模型输出")
                raise Exception("无法找到模型输出")
        
        # 获取最高概率的类别
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        
        # 加载类别映射
        mapping_path = "./models/character_classifier_best_improved_class_mapping.json"
        if os.path.exists(mapping_path):
            import json
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            role = mapping.get(str(predicted_idx), f"角色_{predicted_idx}")
        else:
            role = f"角色_{predicted_idx}"
        
        logger.info(f"CoreML 分类结果: {role}, 置信度: {confidence:.4f}")
        
        return role, confidence, []
        
    except Exception as e:
        logger.error(f"CoreML 分类失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise
