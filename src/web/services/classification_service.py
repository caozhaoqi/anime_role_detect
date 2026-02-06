import numpy as np
from PIL import Image
from loguru import logger
from src.core.classification.general_classification import get_classifier
from src.web.config.config import DEFAULT_INDEX_PATH
from src.web.models.coreml_model import coreml_model, classify_with_coreml
from src.core.log_fusion.log_recorder import record_classification_log


def initialize_system():
    """初始化分类系统"""
    logger.debug("初始化分类系统...")
    # 这里只负责初始化，具体的索引加载由 GeneralClassification 内部处理
    # 默认加载 'role_index'
    classifier = get_classifier(index_path=DEFAULT_INDEX_PATH)
    classifier.initialize()


def classify_image(image_path, use_coreml=False, use_model=False):
    """分类图像
    
    Args:
        image_path: 图像路径
        use_coreml: 是否使用 Core ML 模型
        use_model: 是否使用专用模型
    
    Returns:
        (role, similarity, boxes): 角色名称、相似度、边界框
    """
    if use_coreml and coreml_model is not None:
        # 使用 Core ML 模型
        role, similarity, boxes = classify_with_coreml(image_path)
        mode = 'Core ML模型 (Apple设备)'
        # 记录分类日志
        record_classification_log(
            image_path=image_path,
            role=role,
            similarity=similarity,
            feature=[],  # Core ML 模型不提供特征向量
            boxes=boxes,
            metadata={'mode': mode, 'use_coreml': True}
        )
    else:
        # 使用默认模型
        classifier = get_classifier(index_path=DEFAULT_INDEX_PATH)
        role, similarity, boxes = classifier.classify_image(image_path, use_model=use_model)
        mode = '专用模型 (EfficientNet)' if use_model else '通用索引 (CLIP)'
        # 记录分类日志
        record_classification_log(
            image_path=image_path,
            role=role,
            similarity=similarity,
            feature=[],  # 简化处理，不记录特征向量
            boxes=boxes,
            metadata={'mode': mode, 'use_model': use_model}
        )

    # 安全检查：处理无穷大或无效值
    if similarity is None or not isinstance(similarity, (int, float)):
        similarity = 0.0
    elif np.isinf(similarity) or np.isnan(similarity):
        similarity = 0.0

    return role, similarity, boxes, mode


def get_image_info(image_path):
    """获取图像信息
    
    Args:
        image_path: 图像路径
    
    Returns:
        (img_width, img_height): 图像宽度和高度
    """
    img = Image.open(image_path)
    img_width, img_height = img.size
    return img_width, img_height
