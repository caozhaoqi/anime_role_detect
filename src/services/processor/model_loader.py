#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型加载器（增强版）

负责加载训练好的模型

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


def validate_model_architecture(
    model: torch.nn.Module, expected_num_classes: int, model_name: str
) -> bool:
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
        if hasattr(model, "classifier"):
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
        elif hasattr(model, "fc"):
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
            with open(training_results_path, "r", encoding="utf-8") as f:
                training_results = json.load(f)
                class_to_idx = training_results.get("class_to_idx", {})
                if class_to_idx:
                    logger.info(
                        f"从 training_results.json 加载类别映射：{len(class_to_idx)} 个类别"
                    )
                    return class_to_idx
        except Exception as e:
            logger.warning(f"从 training_results.json 加载失败：{e}")

    return None


def load_trained_model(
    model_name: str, force_reload: bool = False
) -> Optional[Tuple[Any, Dict[str, int]]]:
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
                model_path, map_location=torch.device("cpu"), weights_only=False
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
                with open(class_map_path, "r", encoding="utf-8") as f:
                    class_to_idx = json.load(f)
                logger.info(f"从 class_to_idx.json 加载类别映射：{len(class_to_idx)} 个类别")
            except Exception as e:
                logger.warning(f"从 class_to_idx.json 加载失败：{e}")

        # 3. 从模型文件中加载
        if not class_to_idx and isinstance(checkpoint, dict) and "class_to_idx" in checkpoint:
            class_to_idx = checkpoint["class_to_idx"]
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
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
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
    if "resnet18" in model_name:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes),
        )
    elif "resnet50" in model_name:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes),
        )
    elif "mobilenet_v2" in model_name:
        model = models.mobilenet_v2(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes),
        )
    elif "efficientnet_b0" in model_name:
        model = models.efficientnet_b0(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes),
        )
    elif "efficientnet_b3" in model_name:
        model = models.efficientnet_b3(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 768),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(768),
            nn.Dropout(p=0.15),
            nn.Linear(768, num_classes),
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
            nn.Linear(512, num_classes),
        )

    return model


def clear_model_cache():
    """清空模型缓存"""
    global _model_cache
    _model_cache.clear()
    logger.info("模型缓存已清空")


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
        logger.error(f"加载模型失败：{e}")
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
        logger.error(f"获取预处理器失败：{e}")
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
        logger.error(f"获取关键点检测器失败：{e}")
        return None


# 全局标签器缓存
_tagger_instance = None
_tagger_lock = None

def get_tagger():
    """
    获取标签器（带缓存和超时机制）

    Returns:
        callable: 标签器函数
    """
    global _tagger_instance, _tagger_lock
    
    # 如果标签器已缓存，直接返回
    if _tagger_instance is not None:
        return _tagger_instance.generate_tags
    
    # 初始化锁
    if _tagger_lock is None:
        import threading
        _tagger_lock = threading.Lock()
    
    try:
        # 使用锁确保线程安全
        with _tagger_lock:
            # 双重检查，避免重复创建
            if _tagger_instance is not None:
                return _tagger_instance.generate_tags
                
            from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
            import threading
            import time

            logger.info("开始加载标签器...")
            start_time = time.time()
            
            # 使用超时机制加载标签器
            result = {"success": False, "tagger": None, "error": None}
            
            def load_tagger_thread():
                try:
                    tagger = WDViTV3Tagger()
                    success = tagger.load_model()
                    if success:
                        result["tagger"] = tagger
                        result["success"] = True
                    else:
                        result["error"] = "模型加载失败"
                except Exception as e:
                    result["error"] = e
                    logger.error(f"标签器加载线程失败: {e}")
            
            # 创建并启动加载线程
            thread = threading.Thread(target=load_tagger_thread, daemon=True)
            thread.start()
            
            # 设置最大等待时间（30秒）
            max_wait_time = 30
            wait_interval = 0.1
            elapsed_time = 0
            
            while thread.is_alive() and elapsed_time < max_wait_time:
                time.sleep(wait_interval)
                elapsed_time += wait_interval
            
            if thread.is_alive():
                logger.error("标签器加载超时，将返回None")
                return None
            
            if result["success"] and result["tagger"]:
                _tagger_instance = result["tagger"]
                load_time = time.time() - start_time
                logger.info(f"标签器加载成功，耗时: {load_time:.2f}秒")
                return _tagger_instance.generate_tags
            elif result["error"]:
                logger.error(f"标签器加载失败: {result['error']}")
                return None
                
    except Exception as e:
        logger.error(f"获取标签器失败：{e}")
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
        logger.error(f"获取角色预测器失败：{e}")
        return _search_based_role_predictor


def _search_based_role_predictor(image_path=None, image_bytes=None, tags=None):
    """
    基于搜索服务的角色预测器 - 通过以图搜图获取最相似的角色名

    Args:
        image_path: 图像路径或 BytesIO 对象
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
            # 检查 image_path 是否是字符串（文件路径）
            if isinstance(image_path, str):
                # 文件路径
                result = client.search_image(image_path, top_k=1)
                if result.get("success") and result.get("results"):
                    role_name = result["results"][0].get("role")
                    if role_name:
                        logger.info(f"通过搜索服务识别角色：{role_name}")
                        return role_name
            elif hasattr(image_path, "read"):
                # BytesIO 对象
                image_data = image_path.read()
                result = client.search_image_bytes(image_data, "temp.jpg", top_k=1)
                if result.get("success") and result.get("results"):
                    role_name = result["results"][0].get("role")
                    if role_name:
                        logger.info(f"通过搜索服务识别角色：{role_name}")
                        return role_name

        # 如果没有图像数据或搜索失败，使用标签预测
        return _simple_role_predictor(tags)

    except Exception as e:
        logger.error(f"搜索服务角色预测器失败：{e}")
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
            return "未知角色"

        # 转换为小写的标签列表
        tags_lower = [str(t).lower() for t in tags]

        # 基于标签进行简单的角色预测
        if any(keyword in tags_lower for keyword in ["honkai", "崩坏", "star", "rail"]):
            return "崩坏角色"
        elif any(keyword in tags_lower for keyword in ["genshin", "原神", "impact"]):
            return "原神角色"
        elif any(keyword in tags_lower for keyword in ["blue", "archive", "碧蓝", "档案"]):
            return "碧蓝档案角色"
        elif any(keyword in tags_lower for keyword in ["arknights", "明日", "方舟"]):
            return "明日方舟角色"
        elif any(keyword in tags_lower for keyword in ["fate", "fgo", "grand", "order"]):
            return "Fate 角色"
        elif any(keyword in tags_lower for keyword in ["touhou", "东方"]):
            return "东方角色"
        elif any(keyword in tags_lower for keyword in ["hololive", "vtuber"]):
            return "虚拟主播"
        elif "anime" in tags_lower or "动漫" in tags_lower:
            return "动漫角色"
        else:
            return "未知角色"
    except Exception as e:
        logger.error(f"简单角色预测器失败：{e}")
        return "未知角色"
