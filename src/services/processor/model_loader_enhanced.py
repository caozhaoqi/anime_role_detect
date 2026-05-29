#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的模型加载器 - 修复模型加载问题

修复的问题：
1. 模型加载失败时提供清晰的错误提示
2. 验证模型架构与类别映射的匹配
3. 移除硬编码的 3 个类别
4. 添加模型完整性检查
"""

import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from src.core.logging.global_logger import get_logger

logger = get_logger("model_loader")

# 全局模型缓存
_model_cache: Dict[str, Tuple[Any, Dict]] = {}


class ModelLoadingError(Exception):
    """模型加载异常"""
    pass


def validate_model_architecture(model: torch.nn.Module, expected_num_classes: int, model_name: str) -> bool:
    """
    验证模型架构是否正确
    
    Args:
        model: 模型实例
        expected_num_classes: 期望的类别数量
        model_name: 模型名称
        
    Returns:
        bool: 架构是否匹配
        
    Raises:
        ModelLoadingError: 如果架构不匹配
    """
    try:
        # 获取模型的最后一个层
        if hasattr(model, 'classifier'):
            # MobileNet, EfficientNet 等
            if isinstance(model.classifier, nn.Sequential):
                last_layer = model.classifier[-1]
                if isinstance(last_layer, nn.Linear):
                    actual_num_classes = last_layer.out_features
                    if actual_num_classes != expected_num_classes:
                        raise ModelLoadingError(
                            f"模型架构不匹配：期望{expected_num_classes}个类别，"
                            f"但模型最后一层输出{actual_num_classes}个类别"
                        )
        elif hasattr(model, 'fc'):
            # ResNet 等
            if isinstance(model.fc, nn.Sequential):
                last_layer = model.fc[-1]
            else:
                last_layer = model.fc
            
            if isinstance(last_layer, nn.Linear):
                actual_num_classes = last_layer.out_features
                if actual_num_classes != expected_num_classes:
                    raise ModelLoadingError(
                        f"模型架构不匹配：期望{expected_num_classes}个类别，"
                        f"但模型最后一层输出{actual_num_classes}个类别"
                    )
        
        return True
    except ModelLoadingError:
        raise
    except Exception as e:
        logger.warning(f"模型架构验证失败：{e}")
        return True  # 验证失败但不阻止加载


def get_model_class_to_idx_from_training_config(model_dir: str) -> Optional[Dict[str, int]]:
    """
    从训练配置文件中加载类别映射
    
    Args:
        model_dir: 模型目录
        
    Returns:
        类别映射字典或 None
    """
    # 尝试从 training_results.json 加载
    training_results_path = os.path.join(model_dir, "training_results.json")
    if os.path.exists(training_results_path):
        try:
            with open(training_results_path, 'r', encoding='utf-8') as f:
                training_results = json.load(f)
                class_to_idx = training_results.get('class_to_idx', {})
                if class_to_idx:
                    logger.info(f"从 training_results.json 加载类别映射：{len(class_to_idx)} 个类别")
                    return class_to_idx
        except Exception as e:
            logger.warning(f"从 training_results.json 加载失败：{e}")
    
    return None


def load_trained_model(model_name: str, force_reload: bool = False) -> Optional[Tuple[Any, Dict[str, int]]]:
    """
    加载训练好的模型（增强版）
    
    Args:
        model_name: 模型名称
        force_reload: 是否强制重新加载
        
    Returns:
        tuple: (模型，类别映射) 或 None
        
    Raises:
        ModelLoadingError: 如果模型加载失败
    """
    try:
        # 检查缓存
        if model_name in _model_cache and not force_reload:
            logger.info(f"使用缓存的模型：{model_name}")
            return _model_cache[model_name]
        
        # 处理默认模型
        if model_name == "default":
            model_name = "efficientnet_b0"
        
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent.parent
        model_dir = project_root / "models" / model_name
        
        # 优先使用 model_full.pth
        model_path = model_dir / "model_full.pth"
        if not model_path.exists():
            model_path = model_dir / "model_best.pth"
        
        class_map_path = model_dir / "class_to_idx.json"
        
        # 检查模型文件是否存在
        if not model_path.exists():
            error_msg = f"模型文件不存在：{model_path}"
            logger.error(error_msg)
            raise ModelLoadingError(error_msg)
        
        logger.info(f"加载模型：{model_path}")
        
        # 加载模型
        try:
            checkpoint = torch.load(
                model_path,
                map_location=torch.device('cpu'),
                weights_only=False
            )
        except Exception as e:
            error_msg = f"模型文件损坏或格式错误：{e}"
            logger.error(error_msg)
            raise ModelLoadingError(error_msg)
        
        # 加载类别映射（优先级：training_results.json > class_to_idx.json > 模型文件内嵌）
        class_to_idx = {}
        
        # 1. 从 training_results.json 加载（最可靠）
        class_to_idx = get_model_class_to_idx_from_training_config(str(model_dir))
        
        # 2. 从 class_to_idx.json 加载
        if not class_to_idx and class_map_path.exists():
            try:
                with open(class_map_path, 'r', encoding='utf-8') as f:
                    class_to_idx = json.load(f)
                logger.info(f"从 class_to_idx.json 加载类别映射：{len(class_to_idx)} 个类别")
            except Exception as e:
                logger.warning(f"从 class_to_idx.json 加载失败：{e}")
        
        # 3. 从模型文件中加载
        if not class_to_idx and isinstance(checkpoint, dict) and 'class_to_idx' in checkpoint:
            class_to_idx = checkpoint['class_to_idx']
            logger.info(f"从模型文件加载类别映射：{len(class_to_idx)} 个类别")
        
        # 4. 如果都失败，抛出错误（不再使用硬编码的 3 个类别）
        if not class_to_idx:
            error_msg = (
                f"无法加载类别映射！请确保模型目录包含以下文件之一：\n"
                f"  - training_results.json\n"
                f"  - class_to_idx.json\n"
                f"模型目录：{model_dir}"
            )
            logger.error(error_msg)
            raise ModelLoadingError(error_msg)
        
        num_classes = len(class_to_idx)
        logger.info(f"类别数量：{num_classes}")
        
        # 创建或加载模型
        if isinstance(checkpoint, torch.nn.Module):
            # 完整模型文件
            logger.info(f"加载完整模型：{model_name}")
            model = checkpoint
            model.eval()
        else:
            # 需要创建模型架构
            logger.info(f"创建模型架构：{model_name}")
            model = create_model_from_name(model_name, num_classes)
            
            # 加载权重
            state_dict = None
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                error_msg = f"模型文件中没有找到权重数据"
                logger.error(error_msg)
                raise ModelLoadingError(error_msg)
            
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        
        # 验证模型架构
        validate_model_architecture(model, num_classes, model_name)
        
        # 缓存模型
        _model_cache[model_name] = (model, class_to_idx)
        logger.info(f"成功加载模型：{model_name} ({num_classes}个类别)")
        
        return model, class_to_idx
        
    except ModelLoadingError:
        raise
    except Exception as e:
        error_msg = f"加载模型失败：{e}"
        logger.error(error_msg)
        raise ModelLoadingError(error_msg)


def create_model_from_name(model_name: str, num_classes: int) -> torch.nn.Module:
    """
    根据模型名称创建模型架构
    
    Args:
        model_name: 模型名称
        num_classes: 类别数量
        
    Returns:
        模型实例
    """
    if 'resnet18' in model_name:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )
    elif 'resnet50' in model_name:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )
    elif 'mobilenet_v2' in model_name:
        model = models.mobilenet_v2(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )
    elif 'efficientnet_b0' in model_name:
        model = models.efficientnet_b0(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )
    elif 'efficientnet_b3' in model_name:
        model = models.efficientnet_b3(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 768),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(768),
            nn.Dropout(p=0.15),
            nn.Linear(768, num_classes)
        )
    else:
        # 默认使用 EfficientNet-B0
        logger.warning(f"未知模型名称 {model_name}，使用 EfficientNet-B0 作为默认架构")
        model = models.efficientnet_b0(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )
    
    return model


def clear_model_cache():
    """清空模型缓存"""
    global _model_cache
    _model_cache.clear()
    logger.info("模型缓存已清空")
