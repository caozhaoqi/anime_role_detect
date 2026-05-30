#!/usr/bin/env python3
"""
测试不同阈值设置对识别结果的影响
"""

import sys
import os
import numpy as np
from PIL import Image
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.feature_extraction.feature_extraction import FeatureExtraction
from core.classification.classification import Classification
from core.logging.global_logger import get_logger

logger = get_logger("test_threshold_effect")


def test_threshold_effect():
    """测试不同阈值设置对识别结果的影响"""
    logger.info("=" * 60)
    logger.info("测试不同阈值设置对识别结果的影响")
    logger.info("=" * 60)

    # 初始化特征提取器
    logger.info("\n1. 初始化特征提取器...")
    extractor = FeatureExtraction(quantize=False)

    # 测试图像
    test_image_path = project_root / "scripts" / "test_images" / "sample.jpg"
    if not test_image_path.exists():
        logger.error(f"测试图像不存在: {test_image_path}")
        return

    image = Image.open(test_image_path).convert("RGB")
    logger.info(f"测试图像: {test_image_path}, 大小: {image.size}")

    # 提取特征
    features = extractor.extract_features(image)
    logger.info(f"特征维度: {features.shape}")
    logger.info(f"特征范数: {np.linalg.norm(features):.4f}")

    # 测试不同阈值
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    index_path = project_root / "role_index_augmented.faiss"

    for threshold in thresholds:
        logger.info(f"\n2. 测试阈值: {threshold}")
        classifier = Classification(index_path=str(index_path), threshold=threshold)

        if classifier.index is None:
            logger.error("分类器索引加载失败！")
            continue

        # 分类
        role, similarity = classifier.classify(features, top_k=10)
        logger.info(f"分类结果: 角色={role}, 相似度={similarity:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_threshold_effect()
