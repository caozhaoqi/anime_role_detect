#!/usr/bin/env python3
"""
构建向量索引
使用sdv50_train数据集构建CLIP模型的向量索引
"""

import os
import sys
import numpy as np
from PIL import Image

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入核心模块
from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('build_index')


def build_index(data_dir, index_path):
    """
    构建向量索引
    
    Args:
        data_dir: 数据集目录，包含角色子目录
        index_path: 索引保存路径
    """
    logger.info(f"开始构建索引，数据集目录: {data_dir}, 索引路径: {index_path}")
    
    # 初始化模块
    preprocessor = Preprocessing()
    extractor = FeatureExtraction()
    classifier = Classification()
    
    # 收集数据
    features = []
    role_names = []
    
    # 获取角色目录
    role_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    logger.info(f"找到 {len(role_dirs)} 个角色目录")
    
    for role_dir in role_dirs:
        role_path = os.path.join(data_dir, role_dir)
        try:
            image_files = [f for f in os.listdir(role_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            
            logger.info(f"处理角色: {role_dir}, 共 {len(image_files)} 张图片")
            
            for img_file in image_files:
                img_path = os.path.join(role_path, img_file)
                try:
                    # 预处理
                    normalized_img, _ = preprocessor.process(img_path)
                    
                    # 提取特征
                    feature = extractor.extract_features(normalized_img)
                    features.append(feature)
                    role_names.append(role_dir)
                    
                except Exception as e:
                    logger.error(f"处理图片 {img_file} 时出错: {e}")
                    continue
        except Exception as e:
            logger.error(f"处理角色目录 {role_dir} 时出错: {e}")
            continue
    
    if not features:
        logger.error("无法提取任何特征")
        return False
    
    # 构建索引
    features_np = np.array(features).astype(np.float32)
    classifier.build_index(features_np, role_names)
    
    # 保存索引
    classifier.save_index(index_path)
    
    logger.info(f"索引构建成功，包含 {len(features)} 个向量，覆盖 {len(set(role_names))} 个角色")
    return True


if __name__ == '__main__':
    # 数据集目录
    data_dir = 'data/train'
    # 索引保存路径
    index_path = 'role_index'
    
    build_index(data_dir, index_path)
