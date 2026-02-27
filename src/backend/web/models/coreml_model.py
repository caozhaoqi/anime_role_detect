import json
import numpy as np
import os
import sys
from PIL import Image
from loguru import logger

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.web.config.config import COREML_MODEL_PATH, IS_DARWIN

# Core ML 模型实例
coreml_model = None


def load_coreml_model():
    """加载 Core ML 模型"""
    global coreml_model
    if IS_DARWIN:
        try:
            import coremltools
            if coreml_model is None and COREML_MODEL_PATH:
                if os.path.exists(COREML_MODEL_PATH):
                    coreml_model = coremltools.models.MLModel(COREML_MODEL_PATH)
                    logger.debug(f"Core ML模型已加载: {COREML_MODEL_PATH}")
                else:
                    logger.debug(f"Core ML模型文件不存在: {COREML_MODEL_PATH}")
        except ImportError:
            logger.debug("coremltools未安装，Core ML功能不可用")
        except Exception as e:
            logger.debug(f"Core ML模型加载失败: {e}")
    return coreml_model


def classify_with_coreml(image_path):
    """使用Core ML模型进行分类
    
    Args:
        image_path: 图像路径
    
    Returns:
        (role, similarity, boxes): 角色名称、相似度、边界框
    """
    import json
    import numpy as np
    import os

    # 加载类别映射
    mapping_path = os.path.join('models', 'character_classifier_best_improved_class_mapping.json')
    idx_to_class = None
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            idx_to_class = mapping['idx_to_class']

    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))

    # Core ML推理
    try:
        output = coreml_model.predict({'input': image})

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
                raise ValueError("无法找到Core ML模型输出")

        # 获取最高概率的类别
        if len(predictions.shape) == 2:
            predictions = predictions[0]

        # 应用softmax转换为概率
        exp_predictions = np.exp(predictions - np.max(predictions))  # 数值稳定
        probabilities = exp_predictions / np.sum(exp_predictions)

        predicted_idx = int(np.argmax(probabilities))
        similarity = float(probabilities[predicted_idx])

        # 转换为角色名称
        if idx_to_class:
            # 尝试将predicted_idx转换为字符串查找
            if str(predicted_idx) in idx_to_class:
                role = idx_to_class[str(predicted_idx)]
            else:
                role = f"类别_{predicted_idx}"
        else:
            role = f"类别_{predicted_idx}"

        # Core ML模型不提供边界框信息
        boxes = []

        return role, similarity, boxes

    except Exception as e:
        raise ValueError(f"Core ML推理失败: {e}")
