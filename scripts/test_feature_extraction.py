#!/usr/bin/env python3
"""
测试特征提取功能
"""

import sys
import os
import numpy as np
from PIL import Image
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.feature_extraction.feature_extraction import FeatureExtraction
from core.logging.global_logger import get_logger

logger = get_logger("test_feature_extraction")


def test_feature_extraction():
    """测试特征提取功能"""
    logger.info("=" * 60)
    logger.info("测试特征提取功能")
    logger.info("=" * 60)
    
    try:
        # 初始化特征提取器
        logger.info("初始化特征提取器...")
        extractor = FeatureExtraction(quantize=False)
        
        # 测试1: 测试空输入
        logger.info("\n测试1: 测试空输入")
        try:
            features = extractor.extract_features(None)
            logger.error("空输入应该抛出异常")
        except ValueError as e:
            logger.info(f"✓ 正确处理空输入: {e}")
        
        # 测试2: 测试正常图像
        logger.info("\n测试2: 测试正常图像")
        # 使用jpg文件而不是svg文件
        test_image_path = project_root / "data" / "train" / "日奈" / "日奈_000.jpg"
        if test_image_path.exists():
            logger.info(f"使用测试图像: {test_image_path}")
            image = Image.open(test_image_path).convert('RGB')
            logger.info(f"图像大小: {image.size}")
            
            features = extractor.extract_features(image)
            logger.info(f"✓ 特征提取成功")
            logger.info(f"  特征维度: {features.shape}")
            logger.info(f"  特征类型: {features.dtype}")
            logger.info(f"  特征范数: {np.linalg.norm(features):.4f}")
            logger.info(f"  特征前5个值: {features[:5]}")
            
            # 验证特征向量是否有效
            if features.shape == (512,):
                logger.info("✓ 特征维度正确")
            else:
                logger.error(f"✗ 特征维度错误: {features.shape}")
            
            if np.linalg.norm(features) > 0:
                logger.info("✓ 特征向量非零")
            else:
                logger.error("✗ 特征向量为零")
        else:
            logger.warning(f"测试图像不存在: {test_image_path}")
        
        # 测试3: 测试大图像
        logger.info("\n测试3: 测试大图像")
        large_image = Image.new('RGB', (3000, 3000), color='red')
        logger.info(f"大图像大小: {large_image.size} ({large_image.size[0]*large_image.size[1]}像素)")
        
        features = extractor.extract_features(large_image)
        logger.info(f"✓ 大图像处理成功")
        logger.info(f"  特征维度: {features.shape}")
        
        # 测试4: 测试不同颜色图像
        logger.info("\n测试4: 测试不同颜色图像")
        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        for color in colors:
            test_image = Image.new('RGB', (224, 224), color=color)
            features = extractor.extract_features(test_image)
            logger.info(f"  {color}: 特征范数 = {np.linalg.norm(features):.4f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("特征提取测试完成")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"特征提取测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_feature_extraction()
