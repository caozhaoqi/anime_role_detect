import numpy as np
from PIL import Image
import os
import sys


from src.core.classification.general_classification import get_classifier
from src.config import InferenceConfig
DEFAULT_INDEX_PATH = InferenceConfig.DEFAULT_INDEX_PATH
# CoreML 为可选依赖：模块缺失时置 None，use_coreml 分支自动跳过。
# 修复（#1 深层根因）：原为硬 import，src/services/coreml_model.py 不存在时
# classification_service 整个模块无法导入，视频识别 import 阶段即失败。
try:
    from src.services.coreml_model import coreml_model, classify_with_coreml
except Exception:
    coreml_model = None
    classify_with_coreml = None
from src.core.log_fusion.log_recorder import record_classification_log
from src.services.model.clip_faiss_adapter import get_clip_faiss_classifier
from src.services.model_service.classifiers import EfficientNetClassifier

# 使用全局日志系统
from src.core.logging import get_enhanced_logger as get_logger, log_system, log_inference, log_error

logger = get_logger("classification_service")


def classify_image_with_clip_faiss(image_path: str) -> tuple:
    """使用CLIP+Faiss进行角色识别

    Args:
        image_path: 图像路径

    Returns:
        (role, similarity, boxes, mode, attributes, text_detections)
    """
    logger.info(f"使用CLIP+Faiss进行角色识别: {image_path}")

    clip_classifier = get_clip_faiss_classifier()

    if not clip_classifier.is_available():
        # 尝试初始化
        success = clip_classifier.initialize()
        if not success:
            logger.warning("CLIP+Faiss分类器不可用，返回默认结果")
            return "未知角色", 0.0, [], "CLIP+Faiss (不可用)", [], []

    try:
        result = clip_classifier.classify(image_path, top_k=5)

        role = result.get("role", "未知角色")
        similarity = result.get("similarity", 0.0)
        mode = result.get("mode", "CLIP+Faiss")

        # 记录分类日志
        record_classification_log(
            image_path=image_path,
            role=role,
            similarity=similarity,
            feature=[],
            boxes=[],
            metadata={"mode": mode, "candidates": result.get("candidates", [])[:3]},
        )

        log_inference(
            f"✅ CLIP+Faiss识别成功: {os.path.basename(image_path)}, 角色: {role}, 相似度: {similarity:.4f}, 模式: {mode}"
        )

        return role, similarity, [], mode, [], []

    except Exception as e:
        logger.error(f"CLIP+Faiss识别失败: {e}")
        return "未知角色", 0.0, [], "CLIP+Faiss (错误)", [], []


def initialize_system():
    """初始化分类系统"""
    logger.info("初始化分类系统...")
    # 这里只负责初始化，具体的索引加载由 GeneralClassification 内部处理
    # 默认加载 'role_index'
    classifier = get_classifier(index_path=DEFAULT_INDEX_PATH)
    classifier.initialize()
    logger.info("分类系统初始化完成")


def classify_image(
    image_path,
    use_coreml=False,
    use_model=False,
    use_deepdanbooru=False,
    use_attributes=False,
    model_name=None,
    use_clip_faiss=False,
):
    """分类图像

    Args:
        image_path: 图像路径
        use_coreml: 是否使用 Core ML 模型
        use_model: 是否使用专用模型
        use_deepdanbooru: 是否使用集成DeepDanbooru的分类方法
        use_attributes: 是否使用属性预测
        model_name: 模型名称
        use_clip_faiss: 是否使用CLIP+Faiss检索

    Returns:
        (role, similarity, boxes, mode, attributes, text_detections): 角色名称、相似度、边界框、使用的模式、属性标签、文本检测结果
    """
    logger.info(
        f"开始分类图像: {image_path}, use_coreml={use_coreml}, use_model={use_model}, use_deepdanbooru={use_deepdanbooru}, use_attributes={use_attributes}, model_name={model_name}, use_clip_faiss={use_clip_faiss}"
    )

    # 初始化属性结果
    attributes = []
    text_detections = []

    if use_clip_faiss:
        # 使用CLIP+Faiss进行角色识别（优先）
        role, similarity, boxes, mode, attributes, text_detections = classify_image_with_clip_faiss(image_path)
        # 直接返回，不继续其他分支
        return role, similarity, boxes, mode, attributes, text_detections

    if use_coreml and coreml_model is not None:
        # 使用 Core ML 模型
        try:
            logger.info("使用 Core ML 模型进行分类")
            role, similarity, boxes = classify_with_coreml(image_path)
            mode = "Core ML模型 (Apple设备)"
            # 记录分类日志
            record_classification_log(
                image_path=image_path,
                role=role,
                similarity=similarity,
                feature=[],  # Core ML 模型不提供特征向量
                boxes=boxes,
                metadata={"mode": mode, "use_coreml": True},
            )
            # 使用全局日志系统记录推理结果
            log_inference(
                f"✅ 图像分类成功: {os.path.basename(image_path)}, 角色: {role}, 相似度: {similarity:.4f}, 模式: {mode}"
            )
        except Exception as e:
            logger.error(f"Core ML 分类失败: {e}")
            # 回退到默认模型
            logger.info("Core ML 分类失败，回退到默认模型")
            use_coreml = False
    elif use_deepdanbooru:
        # 使用集成DeepDanbooru的分类方法
        logger.info("使用集成DeepDanbooru的分类方法")
        classifier = get_classifier(index_path=DEFAULT_INDEX_PATH, model=model_name)
        role, similarity, boxes = classifier.classify_image_with_deepdanbooru(image_path)
        mode = "集成模型 (CLIP + 专用模型 + DeepDanbooru)"
        # 记录分类日志
        record_classification_log(
            image_path=image_path,
            role=role,
            similarity=similarity,
            feature=[],  # 简化处理，不记录特征向量
            boxes=boxes,
            metadata={"mode": mode, "use_deepdanbooru": True},
        )
        # 使用全局日志系统记录推理结果
        log_inference(
            f"✅ 图像分类成功: {os.path.basename(image_path)}, 角色: {role}, 相似度: {similarity:.4f}, 模式: {mode}"
        )
    else:
        # 使用 EfficientNet 专用模型直接分类（与 model_service 主路径一致）
        # 修复（#1 根因）：原实现调用 GeneralClassification.classify_image，
        # 但该类仅有 classify() 方法，导致 use_model=True 时抛 AttributeError，
        # 被上层吞掉返回 None —— 视频/批量识别全部“无法识别”。
        logger.info(
            f"使用 EfficientNet 专用模型分类: {image_path}, use_model={use_model}, use_attributes={use_attributes}, model_name={model_name}"
        )
        try:
            from PIL import Image

            classifier = EfficientNetClassifier.get_instance()
            pil_image = Image.open(image_path).convert("RGB")
            if classifier.model is None:
                role, similarity = "未知角色", 0.0
                mode = "EfficientNet (未加载)"
            else:
                role, similarity, _ = classifier.classify_with_features(pil_image)
                mode = "专用模型 (EfficientNet)"
            boxes = []
            if use_attributes:
                try:
                    from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
                    tagger = WDViTV3Tagger.get_instance()
                    tagger.load_model(force_reload=False)
                    attributes = tagger.generate_tags(pil_image)
                except Exception as te:
                    logger.warning(f"属性标签生成失败: {te}")
                    attributes = []
            # 记录分类日志
            record_classification_log(
                image_path=image_path,
                role=role,
                similarity=similarity,
                feature=[],  # 简化处理，不记录特征向量
                boxes=boxes,
                metadata={
                    "mode": mode,
                    "use_model": use_model,
                    "use_attributes": use_attributes,
                    "attributes": [attr["tag"] for attr in attributes[:5]] if attributes else {},
                },
            )
            # 使用全局日志系统记录推理结果
            log_inference(
                f"✅ 图像分类成功: {os.path.basename(image_path)}, 角色: {role}, 相似度: {similarity:.4f}, 模式: {mode}, 属性: {len(attributes)}个"
            )
        except Exception as e:
            logger.error(f"EfficientNet 分类失败: {e}")
            role, similarity, boxes, mode, attributes, text_detections = "未知角色", 0.0, [], "EfficientNet (错误)", [], []

    # 安全检查：处理无穷大或无效值
    if similarity is None or not isinstance(similarity, (int, float)):
        logger.warning(f"相似度值无效: {similarity}，设置为 0.0")
        similarity = 0.0
    elif np.isinf(similarity) or np.isnan(similarity):
        logger.warning(f"相似度值为无穷大或NaN: {similarity}，设置为 0.0")
        similarity = 0.0

    logger.info(
        f"分类完成，角色: {role}, 相似度: {similarity:.4f}, 模式: {mode}, 属性: {len(attributes)}个, 文本: {len(text_detections)}个"
    )
    return role, similarity, boxes, mode, attributes, text_detections


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
