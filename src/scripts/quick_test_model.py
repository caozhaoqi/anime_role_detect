#!/usr/bin/env python3
"""
快速测试模型准确率
只测试一张图片，以快速获得结果
"""

import os
import sys

# 添加项目根目录和src目录到Python路径
src_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.dirname(src_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

# 导入必要的模块
from core.classification.classification import Classification
from core.preprocessing.preprocessing import Preprocessing
from core.feature_extraction.feature_extraction import FeatureExtraction

# 配置日志
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("quick_test_model")


def quick_test_model():
    """
    快速测试模型
    """
    # 初始化分类器
    classifier = Classification(index_path="role_index")

    # 初始化预处理和特征提取模块
    preprocessor = Preprocessing()
    extractor = FeatureExtraction()

    # 测试图片路径
    test_image = "data/train/日奈/日奈_49.jpg"
    true_role = "日奈"

    logger.info(f"测试图片: {test_image}")
    logger.info(f"真实角色: {true_role}")

    try:
        # 预处理图像
        normalized_img, _ = preprocessor.process(test_image)

        # 提取特征
        feature = extractor.extract_features(normalized_img)

        # 分类特征
        predicted_role, similarity = classifier.classify(feature)

        # 检查预测结果是否正确
        is_correct = true_role in predicted_role or predicted_role in true_role

        logger.info(f"预测角色: {predicted_role}")
        logger.info(f"相似度: {similarity:.4f}")
        logger.info(f"结果: {'正确' if is_correct else '错误'}")

        return is_correct, similarity
    except Exception as e:
        logger.error(f"处理图片时出错: {e}")
        return False, 0.0


if __name__ == "__main__":
    logger.info("开始快速测试模型...")
    is_correct, similarity = quick_test_model()
    logger.info("测试完成!")
