#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像预处理器

负责图像的预处理操作
"""

import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from src.core.logging import get_enhanced_logger as get_logger
from src.utils.image_utils import ImageUtils

logger = get_logger("preprocessing")


class ImagePreprocessor:
    """
    图像预处理器类
    """

    def __init__(self, size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        初始化图像预处理器

        Args:
            size: 图像大小
            mean: 均值
            std: 标准差
        """
        self.size = size
        self.mean = mean
        self.std = std
        self.transform = self._get_transform()

    def _get_transform(self):
        """
        获取图像变换

        Returns:
            transforms.Compose: 图像变换
        """
        return transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def preprocess(self, image):
        """
        预处理图像

        Args:
            image: 图像路径或PIL图像

        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        try:
            if isinstance(image, str):
                # 加载图像
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                logger.error("输入不是图像路径或PIL图像")
                return None

            # 应用变换
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            logger.error(f"预处理图像失败: {e}")
            return None

    def preprocess_batch(self, images):
        """
        批量预处理图像

        Args:
            images: 图像列表

        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        try:
            image_tensors = []
            for image in images:
                image_tensor = self.preprocess(image)
                if image_tensor is not None:
                    image_tensors.append(image_tensor)

            if image_tensors:
                return torch.stack(image_tensors)
            else:
                return None
        except Exception as e:
            logger.error(f"批量预处理图像失败: {e}")
            return None

    def resize_image(self, image, size=None):
        """
        调整图像大小

        Args:
            image: 图像路径或PIL图像
            size: 目标大小

        Returns:
            PIL.Image: 调整大小后的图像
        """
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")

            target_size = size or self.size
            resized_image = image.resize(target_size, Image.LANCZOS)
            return resized_image
        except Exception as e:
            logger.error(f"调整图像大小失败: {e}")
            return None

    def normalize_image(self, image):
        """
        归一化图像

        Args:
            image: 图像路径或PIL图像

        Returns:
            torch.Tensor: 归一化后的图像张量
        """
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")

            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]
            )

            return transform(image)
        except Exception as e:
            logger.error(f"归一化图像失败: {e}")
            return None

    def augment_image(self, image):
        """
        增强图像

        Args:
            image: PIL图像

        Returns:
            PIL.Image: 增强后的图像
        """
        try:
            augment_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ]
            )

            return augment_transform(image)
        except Exception as e:
            logger.error(f"增强图像失败: {e}")
            return image

    def save_preprocessed(self, image_tensor, save_path):
        """
        保存预处理后的图像

        Args:
            image_tensor: 图像张量
            save_path: 保存路径
        """
        try:
            # 反归一化
            inv_transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[-m / s for m, s in zip(self.mean, self.std)],
                        std=[1 / s for s in self.std],
                    ),
                    transforms.ToPILImage(),
                ]
            )

            image = inv_transform(image_tensor)

            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 保存图像
            image.save(save_path)
        except Exception as e:
            logger.error(f"保存预处理后的图像失败: {e}")
