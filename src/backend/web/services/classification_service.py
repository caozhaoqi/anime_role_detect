import numpy as np
from PIL import Image
from src.core.classification.general_classification import get_classifier
from src.web.config.config import DEFAULT_INDEX_PATH
from src.web.models.coreml_model import coreml_model, classify_with_coreml
from src.core.log_fusion.log_recorder import record_classification_log
import os

# 使用全局日志系统
from src.core.logging.global_logger import get_logger, log_system, log_inference, log_error
logger = get_logger("classification_service")


def initialize_system():
    """初始化分类系统"""
    logger.info("初始化分类系统...")
    # 这里只负责初始化，具体的索引加载由 GeneralClassification 内部处理
    # 默认加载 'role_index'
    classifier = get_classifier(index_path=DEFAULT_INDEX_PATH)
    classifier.initialize()
    logger.info("分类系统初始化完成")


def classify_image(image_path, use_coreml=False, use_model=False, use_deepdanbooru=False):
    """分类图像
    
    Args:
        image_path: 图像路径
        use_coreml: 是否使用 Core ML 模型
        use_model: 是否使用专用模型
        use_deepdanbooru: 是否使用集成DeepDanbooru的分类方法
    
    Returns:
        (role, similarity, boxes, mode): 角色名称、相似度、边界框、使用的模式
    """
    logger.info(f"开始分类图像: {image_path}, use_coreml={use_coreml}, use_model={use_model}, use_deepdanbooru={use_deepdanbooru}")
    
    if use_coreml and coreml_model is not None:
        # 使用 Core ML 模型
        logger.info("使用 Core ML 模型进行分类")
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
        # 使用全局日志系统记录推理结果
        log_inference(f"✅ 图像分类成功: {os.path.basename(image_path)}, 角色: {role}, 相似度: {similarity:.4f}, 模式: {mode}")
    elif use_deepdanbooru:
        # 使用集成DeepDanbooru的分类方法
        logger.info("使用集成DeepDanbooru的分类方法")
        classifier = get_classifier(index_path=DEFAULT_INDEX_PATH)
        role, similarity, boxes = classifier.classify_image_with_deepdanbooru(image_path)
        mode = '集成模型 (CLIP + 专用模型 + DeepDanbooru)'
        # 记录分类日志
        record_classification_log(
            image_path=image_path,
            role=role,
            similarity=similarity,
            feature=[],  # 简化处理，不记录特征向量
            boxes=boxes,
            metadata={'mode': mode, 'use_deepdanbooru': True}
        )
        # 使用全局日志系统记录推理结果
        log_inference(f"✅ 图像分类成功: {os.path.basename(image_path)}, 角色: {role}, 相似度: {similarity:.4f}, 模式: {mode}")
    else:
        # 使用默认模型
        logger.info(f"使用默认模型进行分类，use_model={use_model}")
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
        # 使用全局日志系统记录推理结果
        log_inference(f"✅ 图像分类成功: {os.path.basename(image_path)}, 角色: {role}, 相似度: {similarity:.4f}, 模式: {mode}")

    # 安全检查：处理无穷大或无效值
    if similarity is None or not isinstance(similarity, (int, float)):
        logger.warning(f"相似度值无效: {similarity}，设置为 0.0")
        similarity = 0.0
    elif np.isinf(similarity) or np.isnan(similarity):
        logger.warning(f"相似度值为无穷大或NaN: {similarity}，设置为 0.0")
        similarity = 0.0

    logger.info(f"分类完成，角色: {role}, 相似度: {similarity:.4f}, 模式: {mode}")
    return role, similarity, boxes, mode


def get_image_info(image_path):
    """获取图像信息
    
    Args:
        image_path: 图像路径
    
    Returns:
        (img_width, img_height): 图像宽度和高度
    """
    logger.debug(f"获取图像信息: {image_path}")
    img = Image.open(image_path)
    img_width, img_height = img.size
    logger.debug(f"图像信息: 宽度={img_width}, 高度={img_height}")
    return img_width, img_height
