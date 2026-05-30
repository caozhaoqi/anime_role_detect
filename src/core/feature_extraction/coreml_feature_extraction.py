#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core ML特征提取模块
"""

import os
import sys
import numpy as np
from PIL import Image
import coremltools as ct
from core.logging.global_logger import get_logger

logger = get_logger("coreml_feature_extraction")


class CoreMLFeatureExtraction:
    """Core ML特征提取模块"""

    # 全局模型实例缓存
    _model_instance = None
    _model_path = None

    def __init__(self, model_path="./coreml_models/clip_model.mlpackage"):
        """初始化Core ML特征提取模块

        Args:
            model_path: Core ML模型路径
        """
        # 检查是否需要重新加载模型
        if not self.__class__._model_instance or self.__class__._model_path != model_path:
            logger.info(f"加载Core ML特征提取模型: {model_path}")
            # 加载Core ML模型
            self.__class__._model_instance = ct.models.MLModel(model_path)
            self.__class__._model_path = model_path
            logger.info("Core ML模型加载成功")

        self.model = self.__class__._model_instance

    def extract_features(self, img):
        """提取图像特征

        Args:
            img: PIL图像对象

        Returns:
            特征向量
        """
        try:
            # 检查输入图像
            if img is None:
                raise ValueError("输入图像为None")

            # 预处理图像
            # 调整图像大小为224x224
            img = img.resize((224, 224))
            # 确保图像为RGB格式 (处理RGBA等其他格式)
            if img.mode != "RGB":
                img = img.convert("RGB")
            # 转换为numpy数组
            img_array = np.array(img).astype(np.float32)
            # 调整通道顺序 (H, W, C) -> (C, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))
            # 添加批次维度
            img_array = np.expand_dims(img_array, axis=0)
            # 归一化
            img_array = (img_array / 255.0 - 0.5) * 2.0

            # 构建输入
            input_data = {"image": img_array}

            # 推理
            logger.info("开始提取特征")
            result = self.model.predict(input_data)

            # 获取特征向量
            features = result["features"]
            # 归一化特征向量
            norm = np.linalg.norm(features)
            if norm > 1e-10:
                features = features / norm
            else:
                # 如果范数为零，使用随机向量
                logger.warning("特征向量范数为零，使用随机向量")
                features = np.random.randn(*features.shape)
                features = features / np.linalg.norm(features)

            # 移除批次维度
            features = features.squeeze()

            logger.info(f"特征提取完成，特征维度: {features.shape}")
            return features
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise

    def batch_extract_features(self, imgs, batch_size=8):
        """批量提取图像特征

        Args:
            imgs: 图像列表
            batch_size: 批量大小

        Returns:
            特征向量列表
        """
        try:
            # 检查输入图像列表
            if not imgs:
                return []

            # 分批处理
            all_features = []
            for i in range(0, len(imgs), batch_size):
                batch_imgs = imgs[i : i + batch_size]
                batch_features = []

                for img in batch_imgs:
                    feature = self.extract_features(img)
                    batch_features.append(feature)

                all_features.extend(batch_features)

            return np.array(all_features)
        except Exception as e:
            logger.error(f"批量特征提取失败: {e}")
            raise


if __name__ == "__main__":
    # 测试Core ML特征提取模块
    import argparse

    parser = argparse.ArgumentParser(description="测试Core ML特征提取模块")
    parser.add_argument("--image-path", type=str, default="test.jpg", help="测试图像路径")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./coreml_models/clip_model.mlpackage",
        help="Core ML模型路径",
    )

    args = parser.parse_args()

    try:
        # 加载图像
        img = Image.open(args.image_path)
        logger.info(f"加载图像: {args.image_path}")

        # 创建特征提取器
        extractor = CoreMLFeatureExtraction(args.model_path)

        # 提取特征
        features = extractor.extract_features(img)

        logger.info(f"特征向量维度: {features.shape}")
        logger.info(f"特征向量前10个元素: {features[:10]}")
        logger.info("特征提取成功!")
    except Exception as e:
        logger.error(f"测试失败: {e}")
