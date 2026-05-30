#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用分类器

提供通用的分类功能
"""

import os
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from src.core.logging.global_logger import get_logger
from src.backend.services.cache_service import get_image_transform

logger = get_logger("general_classification")


class GeneralClassification:
    """
    通用分类器类
    """

    def __init__(self, index_path="role_index"):
        """
        初始化分类器

        Args:
            index_path: 索引路径
        """
        self.index_path = index_path
        self.model = None
        self.processor = None
        self.role_embeddings = {}
        self.role_names = []
        self._load_model()
        self._load_index()

    def _load_model(self):
        """
        加载CLIP模型
        """
        try:
            # 加载CLIP模型
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("成功加载CLIP模型")
        except Exception as e:
            logger.error(f"加载CLIP模型失败: {e}")
            self.model = None
            self.processor = None

    def _load_index(self):
        """
        加载角色索引
        """
        try:
            # 加载角色嵌入
            if os.path.exists(self.index_path):
                for filename in os.listdir(self.index_path):
                    if filename.endswith(".npy"):
                        role_name = os.path.splitext(filename)[0]
                        embedding_path = os.path.join(self.index_path, filename)
                        self.role_embeddings[role_name] = np.load(embedding_path)
                        self.role_names.append(role_name)
                logger.info(f"成功加载 {len(self.role_names)} 个角色的嵌入")
            else:
                logger.warning(f"索引路径不存在: {self.index_path}")
        except Exception as e:
            logger.error(f"加载索引失败: {e}")

    def _get_image_embedding(self, image):
        """
        获取图像嵌入

        Args:
            image: 图像路径或PIL图像

        Returns:
            np.array: 图像嵌入
        """
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")

            if self.model is None or self.processor is None:
                logger.error("模型未加载")
                return None

            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"获取图像嵌入失败: {e}")
            return None

    def _get_text_embedding(self, text):
        """
        获取文本嵌入

        Args:
            text: 文本

        Returns:
            np.array: 文本嵌入
        """
        try:
            if self.model is None or self.processor is None:
                logger.error("模型未加载")
                return None

            inputs = self.processor(text=[text], return_tensors="pt")
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"获取文本嵌入失败: {e}")
            return None

    def classify(self, image, top_k=5):
        """
        分类图像

        Args:
            image: 图像路径或PIL图像
            top_k: 返回前k个结果

        Returns:
            list: 分类结果列表
        """
        try:
            # 获取图像嵌入
            image_embedding = self._get_image_embedding(image)
            if image_embedding is None:
                return []

            # 计算相似度
            similarities = {}
            for role_name, role_embedding in self.role_embeddings.items():
                similarity = np.dot(image_embedding, role_embedding)
                similarities[role_name] = similarity

            # 排序
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

            # 返回前k个结果
            return sorted_similarities[:top_k]
        except Exception as e:
            logger.error(f"分类失败: {e}")
            return []

    def add_role(self, role_name, image_paths):
        """
        添加角色

        Args:
            role_name: 角色名称
            image_paths: 图像路径列表
        """
        try:
            # 计算角色嵌入
            embeddings = []
            for image_path in image_paths:
                embedding = self._get_image_embedding(image_path)
                if embedding is not None:
                    embeddings.append(embedding)

            if embeddings:
                # 计算平均嵌入
                average_embedding = np.mean(embeddings, axis=0)
                # 归一化
                average_embedding = average_embedding / np.linalg.norm(average_embedding)

                # 保存嵌入
                os.makedirs(self.index_path, exist_ok=True)
                embedding_path = os.path.join(self.index_path, f"{role_name}.npy")
                np.save(embedding_path, average_embedding)

                # 更新内存中的嵌入
                self.role_embeddings[role_name] = average_embedding
                if role_name not in self.role_names:
                    self.role_names.append(role_name)

                logger.info(f"成功添加角色: {role_name}")
            else:
                logger.warning(f"无法为角色 {role_name} 生成嵌入")
        except Exception as e:
            logger.error(f"添加角色失败: {e}")

    def remove_role(self, role_name):
        """
        移除角色

        Args:
            role_name: 角色名称
        """
        try:
            # 从内存中移除
            if role_name in self.role_embeddings:
                del self.role_embeddings[role_name]
            if role_name in self.role_names:
                self.role_names.remove(role_name)

            # 从磁盘中移除
            embedding_path = os.path.join(self.index_path, f"{role_name}.npy")
            if os.path.exists(embedding_path):
                os.remove(embedding_path)
                logger.info(f"成功移除角色: {role_name}")
            else:
                logger.warning(f"角色 {role_name} 不存在")
        except Exception as e:
            logger.error(f"移除角色失败: {e}")

    def get_roles(self):
        """
        获取所有角色

        Returns:
            list: 角色名称列表
        """
        return self.role_names

    def update_role(self, role_name, image_paths):
        """
        更新角色

        Args:
            role_name: 角色名称
            image_paths: 图像路径列表
        """
        # 先移除旧角色
        self.remove_role(role_name)
        # 再添加新角色
        self.add_role(role_name, image_paths)
