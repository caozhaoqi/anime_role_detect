#!/usr/bin/env python3
"""
测试分类功能，调试为什么返回unknown
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

logger = get_logger("test_classification_debug")


def test_classification():
    """测试分类功能"""
    logger.info("=" * 60)
    logger.info("测试分类功能 - 调试模式")
    logger.info("=" * 60)
    
    # 初始化特征提取器
    logger.info("\n1. 初始化特征提取器...")
    extractor = FeatureExtraction(quantize=False)
    
    # 加载分类器
    logger.info("\n2. 加载分类器...")
    index_path = project_root / "role_index_augmented.faiss"
    logger.info(f"索引路径: {index_path}")
    logger.info(f"索引文件存在: {index_path.exists()}")
    
    mapping_path = project_root / "role_index_augmented_mapping.json"
    logger.info(f"映射文件存在: {mapping_path.exists()}")
    
    classifier = Classification(index_path=str(index_path), threshold=0.4)
    
    if classifier.index is None:
        logger.error("分类器索引加载失败！")
        return
    
    logger.info(f"索引加载成功，包含 {len(classifier.role_mapping)} 个角色映射")
    logger.info(f"唯一角色: {set(classifier.role_mapping)}")
    
    # 测试1: 使用新创建的测试图像
    logger.info("\n3. 测试1: 使用新创建的测试图像")
    test_image_path = project_root / "scripts" / "test_images" / "sample.jpg"
    if test_image_path.exists():
        logger.info(f"测试图像: {test_image_path}")
        image = Image.open(test_image_path).convert('RGB')
        logger.info(f"图像大小: {image.size}")
        
        # 提取特征
        features = extractor.extract_features(image)
        logger.info(f"特征维度: {features.shape}")
        logger.info(f"特征范数: {np.linalg.norm(features):.4f}")
        
        # 分类
        role, similarity = classifier.classify(features, top_k=10)
        logger.info(f"分类结果: 角色={role}, 相似度={similarity:.4f}")
    else:
        logger.warning(f"测试图像不存在: {test_image_path}")
    
    # 测试3: 使用纯色图像（模拟API测试中的情况）
    logger.info("\n5. 测试3: 使用纯色图像")
    test_image = Image.new('RGB', (224, 224), color='red')
    logger.info(f"测试图像: 224x224 红色纯色图像")
    
    # 提取特征
    features = extractor.extract_features(test_image)
    logger.info(f"特征维度: {features.shape}")
    logger.info(f"特征范数: {np.linalg.norm(features):.4f}")
    logger.info(f"特征前10个值: {features[:10]}")
    
    # 分类
    role, similarity = classifier.classify(features, top_k=10)
    logger.info(f"分类结果: 角色={role}, 相似度={similarity:.4f}")
    
    # 测试3: 检查索引中的特征分布
    logger.info("\n5. 检查索引中的特征分布")
    logger.info(f"索引中的向量数量: {classifier.index.ntotal}")
    logger.info(f"索引维度: {classifier.index.d}")
    
    # 获取索引中的一个向量样本
    if classifier.index.ntotal > 0:
        sample_vector = classifier.index.reconstruct(0)
        logger.info(f"样本向量维度: {sample_vector.shape}")
        logger.info(f"样本向量范数: {np.linalg.norm(sample_vector):.4f}")
        logger.info(f"样本向量前10个值: {sample_vector[:10]}")
    
    logger.info("\n" + "=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_classification()
