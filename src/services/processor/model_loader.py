#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型加载器

负责加载训练好的模型
"""

import os
import torch
from src.core.logging.global_logger import get_logger

logger = get_logger("image_processor")

# 全局模型缓存
_model_cache = {}


def load_trained_model(model_name):
    """
    加载训练好的模型
    
    Args:
        model_name: 模型名称
    
    Returns:
        tuple: (模型, 类别映射) 或 None
    """
    try:
        # 处理默认模型
        if model_name == "default":
            model_name = "efficientnet_b0"
        
        # 获取项目根目录（相对于当前文件向上走3级）
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 模型文件路径（使用绝对路径）
        model_dir = os.path.join(project_root, "models", model_name)
        # 优先使用 model_full.pth 文件，因为它包含完整的权重数据
        model_path = os.path.join(model_dir, "model_full.pth")
        # 如果 model_full.pth 不存在，尝试使用 model_best.pth
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model_best.pth")
        class_map_path = os.path.join(model_dir, "class_to_idx.json")
        
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            logger.warning(f"项目根目录: {project_root}")
            return None
        
        # 加载模型
        # 只使用 weights_only=False 加载完整模型，因为模型文件包含完整的模型结构
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # 加载类别映射
        class_to_idx = {}
        
        # 1. 首先尝试从模型文件中加载class_to_idx
        try:
            if isinstance(checkpoint, dict) and 'class_to_idx' in checkpoint:
                class_to_idx = checkpoint['class_to_idx']
                logger.info(f"从模型文件加载了类别映射: {len(class_to_idx)} 个类别")
        except Exception as e:
            logger.warning(f"从模型文件加载类别映射失败: {e}")
        
        # 2. 尝试从class_to_idx.json文件加载
        if not class_to_idx and os.path.exists(class_map_path):
            try:
                import json
                with open(class_map_path, 'r', encoding='utf-8') as f:
                    class_to_idx = json.load(f)
                logger.info(f"从class_to_idx.json文件加载了类别映射: {len(class_to_idx)} 个类别")
            except Exception as e:
                logger.warning(f"从class_to_idx.json文件加载类别映射失败: {e}")
        
        # 3. 如果都失败，使用默认类别映射
        if not class_to_idx:
            class_to_idx = {"unknown": 0, "plana": 1, "other": 2}
            logger.warning(f"类别映射文件不存在: {class_map_path}，使用默认类别映射")
        
        # 检查是否是完整模型文件（包含模型结构）
        if isinstance(checkpoint, torch.nn.Module):
            logger.info(f"加载完整模型: {model_name}")
            model = checkpoint
            model.eval()
        else:
            # 创建模型实例
            import torchvision.models as models
            import torch.nn as nn
            
            # 根据模型名称选择合适的模型结构
            if 'resnet18' in model_name:
                model = models.resnet18(pretrained=False)
                model.fc = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(model.fc.in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(p=0.1),
                    nn.Linear(512, len(class_to_idx))
                )
            elif 'mobilenet_v2' in model_name:
                model = models.mobilenet_v2(pretrained=False)
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(model.classifier[1].in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(p=0.1),
                    nn.Linear(512, len(class_to_idx))
                )
            else:  # 默认使用EfficientNetB0模型结构
                model = models.efficientnet_b0(pretrained=False)
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(model.classifier[1].in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(p=0.1),
                    nn.Linear(512, len(class_to_idx))
                )
            
            # 加载权重
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                logger.warning(f"模型文件中没有找到权重数据: {model_path}")
                return None
            
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        
        logger.info(f"成功加载模型: {model_name}")
        return model, class_to_idx
        
    except Exception as e:
        logger.error(f"加载训练模型失败: {e}")
        return None


def load_models():
    """
    加载所有模型
    
    Returns:
        bool: 是否加载成功
    """
    try:
        logger.info("加载模型...")
        # 这里可以添加加载多个模型的逻辑
        # 目前只需要返回成功即可
        logger.info("模型加载完成")
        return True
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return False


def get_preprocessor():
    """
    获取预处理器
    
    Returns:
        callable: 预处理器函数
    """
    try:
        from src.core.preprocessing.preprocessing import Preprocessing
        preprocessor = Preprocessing()
        return preprocessor.preprocess
    except Exception as e:
        logger.error(f"获取预处理器失败: {e}")
        return None


def get_keypoint_detector():
    """
    获取关键点检测器
    
    Returns:
        callable: 关键点检测器函数
    """
    try:
        from src.core.keypoint.mediapipe_keypoint_detector import MediaPipeKeypointDetector
        detector = MediaPipeKeypointDetector()
        return detector.detect
    except Exception as e:
        logger.error(f"获取关键点检测器失败: {e}")
        return None


def get_tagger():
    """
    获取标签器
    
    Returns:
        callable: 标签器函数
    """
    try:
        from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
        tagger = WDViTV3Tagger()
        tagger.load_model()
        return tagger.generate_tags
    except Exception as e:
        logger.error(f"获取标签器失败: {e}")
        return None


def get_role_predictor():
    """
    获取角色预测器
    
    Returns:
        callable: 角色预测器函数
    """
    try:
        from src.core.classification.classification import Classification
        classifier = Classification()
        
        # 检查分类器是否有索引
        if classifier.index is not None:
            return classifier.classify
        else:
            # 如果没有索引，使用搜索服务来获取角色名
            logger.warning("分类器索引不存在，尝试使用搜索服务获取角色名")
            return _search_based_role_predictor
    except Exception as e:
        logger.error(f"获取角色预测器失败: {e}")
        return _search_based_role_predictor


def _search_based_role_predictor(image_path=None, image_bytes=None, tags=None):
    """
    基于搜索服务的角色预测器 - 通过以图搜图获取最相似的角色名
    
    Args:
        image_path: 图像路径或BytesIO对象
        image_bytes: 图像字节数据
        tags: 标签列表（备用）
        
    Returns:
        str: 预测的角色名
    """
    try:
        # 首先尝试使用搜索服务
        from src.services.search_service.search_client import get_search_client
        
        client = get_search_client()
        
        # 检查搜索服务是否可用
        if not client.health_check():
            logger.warning("搜索服务不可用，使用标签预测")
            return _simple_role_predictor(tags)
        
        # 如果有图像数据，使用搜索服务获取角色名
        if image_path:
            # 检查image_path是否是字符串（文件路径）
            if isinstance(image_path, str):
                # 文件路径
                result = client.search_image(image_path, top_k=1)
                if result.get('success') and result.get('results'):
                    role_name = result['results'][0].get('role')
                    if role_name:
                        logger.info(f"通过搜索服务识别角色: {role_name}")
                        return role_name
            elif hasattr(image_path, 'read'):
                # BytesIO对象
                image_data = image_path.read()
                result = client.search_image_bytes(image_data, "temp.jpg", top_k=1)
                if result.get('success') and result.get('results'):
                    role_name = result['results'][0].get('role')
                    if role_name:
                        logger.info(f"通过搜索服务识别角色: {role_name}")
                        return role_name
        
        # 如果没有图像数据或搜索失败，使用标签预测
        return _simple_role_predictor(tags)
        
    except Exception as e:
        logger.error(f"搜索服务角色预测器失败: {e}")
        return _simple_role_predictor(tags)


def _simple_role_predictor(tags):
    """
    简单的基于规则的角色预测器
    
    Args:
        tags: 标签列表
        
    Returns:
        str: 预测的角色名
    """
    try:
        if tags is None:
            return '未知角色'
            
        # 转换为小写的标签列表
        tags_lower = [str(t).lower() for t in tags]
        
        # 基于标签进行简单的角色预测
        if any(keyword in tags_lower for keyword in ['honkai', '崩坏', 'star', 'rail']):
            return '崩坏角色'
        elif any(keyword in tags_lower for keyword in ['genshin', '原神', 'impact']):
            return '原神角色'
        elif any(keyword in tags_lower for keyword in ['blue', 'archive', '碧蓝', '档案']):
            return '碧蓝档案角色'
        elif any(keyword in tags_lower for keyword in ['arknights', '明日', '方舟']):
            return '明日方舟角色'
        elif any(keyword in tags_lower for keyword in ['fate', 'fgo', 'grand', 'order']):
            return 'Fate角色'
        elif any(keyword in tags_lower for keyword in ['touhou', '东方']):
            return '东方角色'
        elif any(keyword in tags_lower for keyword in ['hololive', 'vtuber']):
            return '虚拟主播'
        elif 'anime' in tags_lower or '动漫' in tags_lower:
            return '动漫角色'
        else:
            return '未知角色'
    except Exception as e:
        logger.error(f"简单角色预测器失败: {e}")
        return '未知角色'
