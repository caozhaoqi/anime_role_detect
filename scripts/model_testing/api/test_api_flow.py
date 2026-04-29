#!/usr/bin/env python3
"""
测试API调用流程，模拟完整的分类过程
"""

import sys
import os
import numpy as np
from PIL import Image
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from core.feature_extraction.feature_extraction import FeatureExtraction
from core.classification.classification import Classification
from config.config_manager import ConfigManager
from core.logging.global_logger import get_logger

logger = get_logger("test_api_flow")


def test_api_flow():
    """测试API调用流程"""
    logger.info("=" * 60)
    logger.info("测试API调用流程")
    logger.info("=" * 60)
    
    # 测试图像
    test_image_path = project_root / "data" / "train" / "日奈" / "日奈_000.jpg"
    
    if not test_image_path.exists():
        logger.error(f"测试图像不存在: {test_image_path}")
        return
    
    image = Image.open(test_image_path).convert('RGB')
    logger.info(f"测试图像: {test_image_path}")
    logger.info(f"图像大小: {image.size}")
    
    # 模拟API流程
    logger.info("\n1. 获取模型路径（模拟API）")
    config_manager = ConfigManager()
    model_name = "default"
    index_path = config_manager.get_model_path(model_name)
    logger.info(f"模型名称: {model_name}")
    logger.info(f"索引路径: {index_path}")
    logger.info(f"索引文件存在: {os.path.exists(index_path)}")
    
    # 检查映射文件
    mapping_path = index_path.replace(".faiss", "_mapping.json")
    logger.info(f"映射路径: {mapping_path}")
    logger.info(f"映射文件存在: {os.path.exists(mapping_path)}")
    
    # 初始化特征提取器（模拟API，使用非量化）
    logger.info("\n2. 初始化特征提取器（非量化）")
    extractor = FeatureExtraction(quantize=False)
    
    # 提取特征
    features = extractor.extract_features(image)
    logger.info(f"特征维度: {features.shape}")
    logger.info(f"特征范数: {np.linalg.norm(features):.4f}")
    logger.info(f"特征前10个值: {features[:10]}")
    
    # 初始化分类器（模拟API）
    logger.info("\n3. 初始化分类器")
    logger.info(f"使用索引路径: {index_path}")
    classifier = Classification(index_path=index_path, threshold=0.5)
    
    if classifier.index is None:
        logger.error("分类器索引加载失败！")
        return
    
    logger.info(f"分类器初始化成功，角色数量: {len(classifier.role_mapping)}")
    logger.info(f"唯一角色: {set(classifier.role_mapping)}")
    
    # 分类
    logger.info("\n4. 分类")
    role, similarity = classifier.classify(features, top_k=10)
    logger.info(f"分类结果: 角色={role}, 相似度={similarity:.4f}")
    
    # 检查索引中的向量
    logger.info("\n5. 检查索引中的向量")
    logger.info(f"索引中的向量数量: {classifier.index.ntotal}")
    logger.info(f"索引维度: {classifier.index.d}")
    
    # 获取索引中的一个向量样本
    if classifier.index.ntotal > 0:
        sample_vector = classifier.index.reconstruct(0)
        logger.info(f"样本向量维度: {sample_vector.shape}")
        logger.info(f"样本向量范数: {np.linalg.norm(sample_vector):.4f}")
        logger.info(f"样本向量前10个值: {sample_vector[:10]}")
        
        # 计算与样本向量的相似度
        cosine_sim = np.dot(features, sample_vector) / (np.linalg.norm(features) * np.linalg.norm(sample_vector))
        logger.info(f"与样本向量的余弦相似度: {cosine_sim:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_api_flow()
