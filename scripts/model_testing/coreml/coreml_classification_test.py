#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Core ML分类器的性能
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
logger = get_logger("coreml_classification_test")


def test_coreml_classification():
    """测试Core ML分类器的性能"""
    try:
        # 导入分类模块
        from core.classification.classification import Classification
        
        # 创建Core ML分类器
        logger.info("创建Core ML分类器")
        classifier = Classification.use_coreml()
        
        # 加载测试图像
        test_images = [
            "日奈_1.svg",
            "日奈_2.svg", 
            "日奈_4.svg",
            "日奈_5.svg"
        ]
        
        # 测试性能
        total_time = 0
        results = []
        
        for i, img_name in enumerate(test_images):
            img_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train', '日奈', img_name)
            if not os.path.exists(img_path):
                logger.warning(f"测试图像不存在: {img_path}")
                continue
            
            logger.info(f"测试图像 {i+1}/{len(test_images)}: {img_path}")
            
            # 加载图像
            img = Image.open(img_path)
            
            # 测试分类性能
            start_time = time.time()
            role, similarity = classifier.classify_image(img)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            
            results.append({
                "image": img_name,
                "role": role,
                "similarity": similarity,
                "time": elapsed_time
            })
            
            logger.info(f"分类结果: {role}, 相似度: {similarity:.4f}, 耗时: {elapsed_time:.4f}秒")
        
        # 计算平均性能
        if results:
            avg_time = total_time / len(results)
            logger.info(f"平均分类时间: {avg_time:.4f}秒")
            
            # 打印所有结果
            logger.info("\n所有测试结果:")
            for result in results:
                logger.info(f"图像: {result['image']}, 角色: {result['role']}, 相似度: {result['similarity']:.4f}, 耗时: {result['time']:.4f}秒")
        else:
            logger.warning("没有测试图像")
        
        logger.info("Core ML分类器测试完成")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


def main():
    """主函数"""
    try:
        logger.info("开始测试Core ML分类器...")
        test_coreml_classification()
        logger.info("测试完成!")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")


if __name__ == "__main__":
    main()
