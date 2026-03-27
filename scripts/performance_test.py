#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试脚本：测试不同加速方法的性能差异
"""

import os
import sys
import time
import numpy as np
from PIL import Image

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 使用全局日志系统
from core.logging.global_logger import get_logger
logger = get_logger("performance_test")


def test_feature_extraction():
    """测试特征提取性能"""
    from core.feature_extraction.feature_extraction import FeatureExtraction
    
    # 加载测试图像
    test_image_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train', '日奈', '日奈_49.jpg')
    if not os.path.exists(test_image_path):
        logger.error(f"测试图像不存在: {test_image_path}")
        return
    
    img = Image.open(test_image_path)
    logger.info(f"加载测试图像: {test_image_path}")
    
    # 测试特征提取
    logger.info("开始测试特征提取性能...")
    
    # 创建特征提取器
    extractor = FeatureExtraction()
    
    # 预热
    logger.info("预热模型...")
    for _ in range(3):
        extractor.extract_features(img)
    
    # 测试多次推理
    num_tests = 10
    times = []
    
    for i in range(num_tests):
        start_time = time.time()
        features = extractor.extract_features(img)
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        logger.info(f"测试 {i+1}/{num_tests}: {elapsed:.4f} 秒")
    
    # 计算统计信息
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    logger.info("特征提取性能测试结果:")
    logger.info(f"平均时间: {mean_time:.4f} 秒")
    logger.info(f"标准差: {std_time:.4f} 秒")
    logger.info(f"最小时间: {min_time:.4f} 秒")
    logger.info(f"最大时间: {max_time:.4f} 秒")
    
    return mean_time

def test_tag_generation():
    """测试标签生成性能"""
    from core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
    
    # 加载测试图像
    test_image_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train', '日奈', '日奈_49.jpg')
    if not os.path.exists(test_image_path):
        logger.error(f"测试图像不存在: {test_image_path}")
        return
    
    logger.info(f"加载测试图像: {test_image_path}")
    
    # 测试标签生成
    logger.info("开始测试标签生成性能...")
    
    # 创建标签生成器
    tagger = WDViTV3Tagger()
    # 非Core ML模式需要加载模型
    if not hasattr(tagger, 'coreml_mode') or not tagger.coreml_mode:
        tagger.load_model()
    
    # 预热
    logger.info("预热模型...")
    for _ in range(3):
        tagger.generate_tags(test_image_path)
    
    # 测试多次推理
    num_tests = 10
    times = []
    
    for i in range(num_tests):
        start_time = time.time()
        tags = tagger.generate_tags(test_image_path)
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        logger.info(f"测试 {i+1}/{num_tests}: {elapsed:.4f} 秒, 生成标签数: {len(tags)}")
    
    # 计算统计信息
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    logger.info("标签生成性能测试结果:")
    logger.info(f"平均时间: {mean_time:.4f} 秒")
    logger.info(f"标准差: {std_time:.4f} 秒")
    logger.info(f"最小时间: {min_time:.4f} 秒")
    logger.info(f"最大时间: {max_time:.4f} 秒")
    
    return mean_time

def main():
    """主函数"""
    try:
        logger.info("开始性能测试...")
        
        # 测试特征提取
        logger.info("=" * 60)
        logger.info("测试特征提取性能")
        logger.info("=" * 60)
        feature_extraction_time = test_feature_extraction()
        
        # 测试标签生成
        logger.info("=" * 60)
        logger.info("测试标签生成性能")
        logger.info("=" * 60)
        tag_generation_time = test_tag_generation()
        
        # 汇总结果
        logger.info("=" * 60)
        logger.info("性能测试汇总")
        logger.info("=" * 60)
        logger.info(f"特征提取平均时间: {feature_extraction_time:.4f} 秒")
        logger.info(f"标签生成平均时间: {tag_generation_time:.4f} 秒")
        logger.info(f"总时间: {feature_extraction_time + tag_generation_time:.4f} 秒")
        
        logger.info("性能测试完成!")
    except Exception as e:
        logger.error(f"性能测试失败: {e}")


if __name__ == "__main__":
    main()
