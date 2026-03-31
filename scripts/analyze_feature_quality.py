#!/usr/bin/env python3
"""
分析特征向量质量和分布情况
"""

import sys
import os
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.feature_extraction.feature_extraction import FeatureExtraction
from core.classification.classification import Classification
from core.logging.global_logger import get_logger

logger = get_logger("analyze_feature_quality")

def analyze_feature_quality():
    """分析特征向量质量和分布情况"""
    logger.info("=" * 60)
    logger.info("分析特征向量质量和分布情况")
    logger.info("=" * 60)
    
    # 1. 分析索引中的特征向量
    logger.info("\n1. 分析索引中的特征向量")
    index_path = project_root / "role_index_augmented.faiss"
    classifier = Classification(index_path=str(index_path), threshold=0.4)
    
    if classifier.index is None:
        logger.error("分类器索引加载失败！")
        return
    
    logger.info(f"索引中的向量数量: {classifier.index.ntotal}")
    logger.info(f"索引维度: {classifier.index.d}")
    
    # 2. 分析特征向量的分布
    logger.info("\n2. 分析特征向量的分布")
    
    # 随机选择100个向量进行分析
    sample_size = min(100, classifier.index.ntotal)
    sample_indices = np.random.choice(classifier.index.ntotal, sample_size, replace=False)
    
    # 重建这些向量
    sample_vectors = []
    for idx in sample_indices:
        vector = classifier.index.reconstruct(int(idx))
        sample_vectors.append(vector)
    
    sample_vectors = np.array(sample_vectors)
    logger.info(f"样本向量形状: {sample_vectors.shape}")
    
    # 计算范数
    norms = np.linalg.norm(sample_vectors, axis=1)
    logger.info(f"特征向量范数统计:")
    logger.info(f"  平均值: {np.mean(norms):.4f}")
    logger.info(f"  最小值: {np.min(norms):.4f}")
    logger.info(f"  最大值: {np.max(norms):.4f}")
    logger.info(f"  标准差: {np.std(norms):.4f}")
    
    # 3. 分析特征向量之间的相似度
    logger.info("\n3. 分析特征向量之间的相似度")
    
    # 计算样本向量之间的余弦相似度
    if sample_size > 1:
        # 计算余弦相似度矩阵
        similarity_matrix = np.dot(sample_vectors, sample_vectors.T)
        # 排除对角线（自身相似度）
        np.fill_diagonal(similarity_matrix, 0)
        
        # 计算平均相似度
        avg_similarity = np.mean(similarity_matrix)
        logger.info(f"样本向量之间的平均相似度: {avg_similarity:.4f}")
        logger.info(f"相似度范围: {np.min(similarity_matrix):.4f} - {np.max(similarity_matrix):.4f}")
    
    # 4. 分析角色分布
    logger.info("\n4. 分析角色分布")
    role_counts = {}
    for role in classifier.role_mapping:
        if role not in role_counts:
            role_counts[role] = 0
        role_counts[role] += 1
    
    logger.info(f"唯一角色数量: {len(role_counts)}")
    logger.info("角色分布:")
    for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {role}: {count} 个向量")
    
    # 5. 测试特征提取质量
    logger.info("\n5. 测试特征提取质量")
    extractor = FeatureExtraction(quantize=False)
    
    # 测试图像
    test_image_path = project_root / "scripts" / "test_images" / "sample.jpg"
    if test_image_path.exists():
        from PIL import Image
        image = Image.open(test_image_path).convert('RGB')
        features = extractor.extract_features(image)
        
        logger.info(f"测试图像特征向量:")
        logger.info(f"  维度: {features.shape}")
        logger.info(f"  范数: {np.linalg.norm(features):.4f}")
        logger.info(f"  前10个值: {features[:10]}")
    
    logger.info("\n" + "=" * 60)
    logger.info("分析完成")
    logger.info("=" * 60)

if __name__ == "__main__":
    analyze_feature_quality()
