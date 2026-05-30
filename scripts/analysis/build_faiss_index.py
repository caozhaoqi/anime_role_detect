#!/usr/bin/env python3
"""
构建Faiss索引脚本
从训练数据中提取特征并构建向量索引
"""

import os
import sys
import numpy as np
import json

# 添加项目根目录和src目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification
from core.logging.global_logger import get_logger

logger = get_logger("build_index")


def get_project_root():
    """
    获取项目根目录
    """
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root


def load_training_data():
    """
    加载训练数据
    """
    project_root = get_project_root()
    train_dir = os.path.join(project_root, "data", "downloaded_images")

    if not os.path.exists(train_dir):
        logger.error(f"训练数据目录不存在: {train_dir}")
        return []

    training_data = []
    for role_name in os.listdir(train_dir):
        role_dir = os.path.join(train_dir, role_name)
        if os.path.isdir(role_dir):
            for image_file in os.listdir(role_dir):
                if image_file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".svg")):
                    image_path = os.path.join(role_dir, image_file)
                    training_data.append((image_path, role_name))

    logger.info(f"加载了 {len(training_data)} 个训练样本")
    return training_data


def build_index():
    """
    构建Faiss索引
    """
    project_root = get_project_root()
    index_path = os.path.join(project_root, "role_index")

    # 加载训练数据
    training_data = load_training_data()
    if not training_data:
        logger.error("没有找到训练数据，无法构建索引")
        return

    # 初始化预处理和特征提取模块
    preprocessor = Preprocessing()
    extractor = FeatureExtraction()

    # 提取特征
    features = []
    role_names = []

    logger.info("开始提取特征...")
    for i, (image_path, role_name) in enumerate(training_data):
        try:
            # 预处理图像
            normalized_img, _ = preprocessor.process(image_path)

            # 提取特征
            feature = extractor.extract_features(normalized_img)
            features.append(feature)
            role_names.append(role_name)

            if (i + 1) % 10 == 0:
                logger.info(f"已处理 {i + 1}/{len(training_data)} 个样本")
        except Exception as e:
            logger.error(f"处理图像 {image_path} 时出错: {e}")
            continue

    if not features:
        logger.error("没有成功提取任何特征，无法构建索引")
        return

    # 转换为numpy数组
    features = np.array(features, dtype=np.float32)
    logger.info(f"特征提取完成，特征形状: {features.shape}")

    # 构建索引
    classifier = Classification()
    classifier.build_index(features, role_names)

    # 保存索引
    classifier.save_index(index_path)
    logger.info(f"索引保存完成: {index_path}")


if __name__ == "__main__":
    logger.info("开始构建Faiss索引...")
    build_index()
    logger.info("索引构建完成！")
