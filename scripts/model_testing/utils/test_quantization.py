#!/usr/bin/env python3
"""
测试量化对特征提取的影响
"""

import sys
import numpy as np
from PIL import Image
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.feature_extraction.feature_extraction import FeatureExtraction
from core.classification.classification import Classification
from core.logging.global_logger import get_logger

logger = get_logger("test_quantization")


def test_quantization():
    """测试量化的影响"""
    logger.info("=" * 60)
    logger.info("测试量化对特征提取的影响")
    logger.info("=" * 60)

    # 测试图像
    test_image_path = project_root / "data" / "train" / "日奈" / "日奈_000.jpg"

    if not test_image_path.exists():
        logger.error(f"测试图像不存在: {test_image_path}")
        return

    image = Image.open(test_image_path).convert("RGB")
    logger.info(f"测试图像: {test_image_path}")
    logger.info(f"图像大小: {image.size}")

    # 加载分类器
    index_path = project_root / "role_index.faiss"
    classifier = Classification(index_path=str(index_path), threshold=0.5)

    if classifier.index is None:
        logger.error("分类器索引加载失败！")
        return

    # 测试1: 非量化特征提取
    logger.info("\n1. 测试非量化特征提取 (quantize=False)")
    extractor_non_quantized = FeatureExtraction(quantize=False)
    features_non_quantized = extractor_non_quantized.extract_features(image)

    logger.info(f"  特征范数: {np.linalg.norm(features_non_quantized):.4f}")
    logger.info(f"  特征前10个值: {features_non_quantized[:10]}")

    # 分类
    role_nq, similarity_nq = classifier.classify(features_non_quantized, top_k=10)
    logger.info(f"  分类结果: 角色={role_nq}, 相似度={similarity_nq:.4f}")

    # 测试2: 量化特征提取
    logger.info("\n2. 测试量化特征提取 (quantize=True)")
    extractor_quantized = FeatureExtraction(quantize=True)
    features_quantized = extractor_quantized.extract_features(image)

    logger.info(f"  特征范数: {np.linalg.norm(features_quantized):.4f}")
    logger.info(f"  特征前10个值: {features_quantized[:10]}")

    # 分类
    role_q, similarity_q = classifier.classify(features_quantized, top_k=10)
    logger.info(f"  分类结果: 角色={role_q}, 相似度={similarity_q:.4f}")

    # 测试3: 计算特征差异
    logger.info("\n3. 特征差异分析")
    feature_diff = features_non_quantized - features_quantized
    diff_norm = np.linalg.norm(feature_diff)
    cosine_sim = np.dot(features_non_quantized, features_quantized) / (
        np.linalg.norm(features_non_quantized) * np.linalg.norm(features_quantized)
    )

    logger.info(f"  特征向量差异范数: {diff_norm:.4f}")
    logger.info(f"  余弦相似度: {cosine_sim:.4f}")

    # 测试4: Core ML模式（非量化）
    logger.info("\n4. 测试Core ML模式")
    extractor_coreml = FeatureExtraction(quantize=False, coreml_mode=True)
    features_coreml = extractor_coreml.extract_features(image)

    logger.info(f"  特征范数: {np.linalg.norm(features_coreml):.4f}")
    logger.info(f"  特征前10个值: {features_coreml[:10]}")

    # 分类
    role_cml, similarity_cml = classifier.classify(features_coreml, top_k=10)
    logger.info(f"  分类结果: 角色={role_cml}, 相似度={similarity_cml:.4f}")

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    logger.info(f"非量化模型: 角色={role_nq}, 相似度={similarity_nq:.4f}")
    logger.info(f"量化模型: 角色={role_q}, 相似度={similarity_q:.4f}")
    logger.info(f"Core ML模型: 角色={role_cml}, 相似度={similarity_cml:.4f}")

    logger.info(
        f"\n相似度下降（量化 vs 非量化）: {(similarity_nq - similarity_q) / similarity_nq * 100:.1f}%"
    )
    logger.info(
        f"相似度下降（Core ML vs 量化）: {(similarity_cml - similarity_q) / similarity_cml * 100:.1f}%"
    )

    logger.info("\n" + "=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_quantization()
