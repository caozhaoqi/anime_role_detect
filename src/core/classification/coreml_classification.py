#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core ML端到端分类模块
"""

import os
import sys
import json
import numpy as np
import faiss
import coremltools as ct
from PIL import Image

# 使用全局日志系统
from core.logging.global_logger import get_logger

logger = get_logger("coreml_classification")


class CoreMLClassification:
    """Core ML端到端分类模块"""

    # 全局模型实例缓存
    _clip_model = None
    _index = None
    _mapping = None
    _config = None

    def __init__(self, config_path="./coreml_models/end_to_end_config.json"):
        """初始化Core ML分类模块

        Args:
            config_path: 配置文件路径
        """
        # 检查是否需要重新加载配置
        if not self.__class__._config or self.__class__._config.get("config_path") != config_path:
            logger.info(f"加载配置文件: {config_path}")
            with open(config_path, "r") as f:
                self.__class__._config = json.load(f)
            self.__class__._config["config_path"] = config_path
            logger.info("配置文件加载成功")

        # 加载CLIP Core ML模型
        if not self.__class__._clip_model:
            clip_model_path = self.__class__._config.get("clip_model_path")
            if not clip_model_path or not os.path.exists(clip_model_path):
                raise FileNotFoundError(f"CLIP Core ML模型不存在: {clip_model_path}")

            logger.info(f"加载CLIP Core ML模型: {clip_model_path}")
            self.__class__._clip_model = ct.models.MLModel(clip_model_path)
            logger.info("CLIP Core ML模型加载成功")

        # 加载索引文件
        if not self.__class__._index:
            index_path = self.__class__._config.get("index_path")
            if not index_path or not os.path.exists(index_path):
                raise FileNotFoundError(f"索引文件不存在: {index_path}")

            logger.info(f"加载Faiss索引: {index_path}")
            self.__class__._index = faiss.read_index(index_path)
            logger.info(f"索引加载成功，包含 {self.__class__._index.ntotal} 个向量")

        # 加载映射文件
        if not self.__class__._mapping:
            mapping_path = self.__class__._config.get("mapping_path")
            if mapping_path and os.path.exists(mapping_path):
                logger.info(f"加载角色映射: {mapping_path}")
                with open(mapping_path, "r") as f:
                    self.__class__._mapping = json.load(f)
                logger.info(f"角色映射加载成功，包含 {len(self.__class__._mapping)} 个角色")
            else:
                logger.warning("角色映射文件不存在，将使用默认映射")
                self.__class__._mapping = {}

        self.config = self.__class__._config
        self.clip_model = self.__class__._clip_model
        self.index = self.__class__._index
        self.mapping = self.__class__._mapping
        self.role_mapping = self.__class__._mapping  # 添加role_mapping属性，与API代码兼容

    def classify_image(self, img, threshold=0.6):
        """分类图像

        Args:
            img: PIL图像对象
            threshold: 相似度阈值

        Returns:
            分类结果，包含角色名称和相似度
        """
        try:
            # 检查输入图像
            if img is None:
                raise ValueError("输入图像为None")

            # 预处理图像
            img = img.resize((224, 224))
            img_array = np.array(img).astype(np.float32)
            img_array = np.transpose(img_array, (2, 0, 1))
            img_array = np.expand_dims(img_array, axis=0)
            img_array = (img_array / 255.0 - 0.5) * 2.0

            # 使用Core ML提取特征
            logger.info("使用Core ML提取特征")
            input_data = {"image": img_array}
            result = self.clip_model.predict(input_data)
            features = result["features"]

            # 归一化特征向量
            norm = np.linalg.norm(features)
            if norm > 1e-10:
                features = features / norm
            else:
                logger.warning("特征向量范数为零，使用随机向量")
                features = np.random.randn(*features.shape)
                features = features / np.linalg.norm(features)

            # 使用Faiss搜索最相似的向量
            logger.info("使用Faiss搜索最相似的向量")
            distances, indices = self.index.search(features, 1)

            # 计算相似度
            similarity = 1.0 - distances[0][0]
            logger.info(f"搜索完成，相似度: {similarity:.4f}")

            # 获取角色名称
            role_id = indices[0][0]
            role = self.mapping.get(str(role_id), f"角色_{role_id}")

            # 检查相似度是否超过阈值
            if similarity < threshold:
                logger.info(f"相似度 {similarity:.4f} 低于阈值 {threshold}，返回未知角色")
                return "未知角色", similarity

            logger.info(f"分类结果: {role}，相似度: {similarity:.4f}")
            return role, similarity
        except Exception as e:
            logger.error(f"分类失败: {e}")
            raise

    def classify(self, feature, tags=None):
        """分类图像（兼容API代码）

        Args:
            feature: 特征向量
            tags: 图像标签（未使用）

        Returns:
            分类结果，包含角色名称和相似度
        """
        try:
            # 检查输入特征
            if feature is None:
                raise ValueError("输入特征为None")

            # 归一化特征向量
            norm = np.linalg.norm(feature)
            if norm > 1e-10:
                feature = feature / norm
            else:
                logger.warning("特征向量范数为零，使用随机向量")
                feature = np.random.randn(*feature.shape)
                feature = feature / np.linalg.norm(feature)

            # 使用Faiss搜索最相似的向量
            logger.info("使用Faiss搜索最相似的向量")
            logger.info(f"特征形状: {feature.shape}")
            logger.info(f"索引类型: {type(self.index)}")
            logger.info(f"索引维度: {self.index.d}")
            logger.info(f"索引向量数量: {self.index.ntotal}")

            # 确保特征是二维数组
            if len(feature.shape) == 1:
                feature = feature.reshape(1, -1)
                logger.info(f"特征形状已调整为: {feature.shape}")

            search_result = self.index.search(feature, 1)
            logger.info(f"搜索结果类型: {type(search_result)}")
            logger.info(f"搜索结果长度: {len(search_result)}")

            if len(search_result) == 2:
                distances, indices = search_result
                logger.info(f"距离形状: {distances.shape}")
                logger.info(f"索引形状: {indices.shape}")

                # 计算相似度
                similarity = 1.0 - distances[0][0]
                logger.info(f"搜索完成，相似度: {similarity:.4f}")

                # 获取角色名称
                role_id = indices[0][0]
                role = self.mapping.get(str(role_id), f"角色_{role_id}")

                # 检查相似度是否超过阈值
                if similarity < 0.6:
                    logger.info(f"相似度 {similarity:.4f} 低于阈值 0.6，返回未知角色")
                    return "unknown", similarity

                logger.info(f"分类结果: {role}，相似度: {similarity:.4f}")
                return role, similarity
            else:
                logger.error(f"搜索结果格式错误，期望2个值，实际得到{len(search_result)}个值")
                raise ValueError(f"搜索结果格式错误，期望2个值，实际得到{len(search_result)}个值")
        except Exception as e:
            logger.error(f"分类失败: {e}")
            raise

    def batch_classify_images(self, imgs, threshold=0.6):
        """批量分类图像

        Args:
            imgs: 图像列表
            threshold: 相似度阈值

        Returns:
            分类结果列表
        """
        try:
            # 检查输入图像列表
            if not imgs:
                return []

            # 批量处理
            results = []
            for img in imgs:
                role, similarity = self.classify_image(img, threshold)
                results.append((role, similarity))

            return results
        except Exception as e:
            logger.error(f"批量分类失败: {e}")
            raise


if __name__ == "__main__":
    # 测试Core ML分类模块
    import argparse

    parser = argparse.ArgumentParser(description="测试Core ML分类模块")
    parser.add_argument("--image-path", type=str, default="test.jpg", help="测试图像路径")
    parser.add_argument(
        "--config-path",
        type=str,
        default="./coreml_models/end_to_end_config.json",
        help="配置文件路径",
    )
    parser.add_argument("--threshold", type=float, default=0.6, help="相似度阈值")

    args = parser.parse_args()

    try:
        # 加载图像
        img = Image.open(args.image_path)
        logger.info(f"加载图像: {args.image_path}")

        # 创建分类器
        classifier = CoreMLClassification(args.config_path)

        # 分类图像
        role, similarity = classifier.classify_image(img, args.threshold)

        logger.info(f"分类结果: {role}")
        logger.info(f"相似度: {similarity:.4f}")
        logger.info("分类成功!")
    except Exception as e:
        logger.error(f"测试失败: {e}")
