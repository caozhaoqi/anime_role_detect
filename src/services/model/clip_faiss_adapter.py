#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP+Faiss 适配器 - 提供与传统分类器兼容的接口
CLIP+Faiss Adapter - Provides compatible interface with traditional classifier

用于将 CharacterRetriever 适配到 recognition_service 中
"""

import os
import sys
import platform
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

if platform.system() == "Darwin":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.core.recognition import CharacterRetriever
from src.core.logging.global_logger import get_logger

logger = get_logger("clip_faiss_adapter")


class CLIPFaissAdapter:
    """
    CLIP+Faiss 适配器

    提供与传统GeneralClassification相似的接口：
    - classify(image_path) -> (role, similarity, mode, ...)
    - initialize() -> 加载模型

    优势：
    - 懒加载，避免启动开销
    - 自动设备选择
    - 兼容现有调用方式
    """

    _instance: Optional['CLIPFaissAdapter'] = None

    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized_once = False
        return cls._instance

    def __init__(
        self,
        index_path: str = "data/feature_store/character_index.faiss",
        metadata_path: str = "data/feature_store/character_metadata.json",
        clip_model_name: str = "ViT-B/32",
        use_huggingface: bool = True,
        similarity_threshold: float = 0.4,
    ):
        if self._initialized_once:
            return

        self.index_path = index_path
        self.metadata_path = metadata_path
        self.similarity_threshold = similarity_threshold
        self._retriever: Optional[CharacterRetriever] = None
        self._initialized = False
        self._initialized_once = True

    def initialize(self) -> bool:
        """懒加载"""
        if self._initialized:
            return True

        try:
            if not Path(self.index_path).exists():
                logger.warning(f"特征库不存在: {self.index_path}")
                logger.warning("请先运行 build_feature_store.py 构建特征库")
                return False

            logger.info("🚀 初始化CLIP+Faiss识别器...")
            self._retriever = CharacterRetriever(
                clip_model_name=clip_model_name,
                feature_store_path=self.index_path,
                metadata_path=self.metadata_path,
                use_huggingface=use_huggingface,
                similarity_threshold=self.similarity_threshold,
            )
            self._retriever.initialize()
            self._initialized = True
            logger.info("✅ CLIP+Faiss识别器初始化完成")
            return True
        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            return False

    def classify(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        top_k: int = 5,
    ) -> Dict:
        """
        分类图片 - 兼容传统接口

        Args:
            image_input: 图片路径/PIL Image/numpy数组
            top_k: 返回前k个候选

        Returns:
            {
                "role": str,  # 最佳匹配角色
                "similarity": float,  # 相似度
                "mode": str,  # 识别模式 (clip_faiss)
                "candidates": List[Dict],  # 所有候选
            }
        """
        if not self._initialized and not self.initialize():
            return {
                "role": "unknown",
                "similarity": 0.0,
                "mode": "uninitialized",
                "candidates": [],
            }

        try:
            candidates = self._retriever.identify(image_input, top_k=top_k)

            if not candidates:
                return {
                    "role": "unknown",
                    "similarity": 0.0,
                    "mode": "clip_faiss",
                    "candidates": [],
                }

            best = candidates[0]
            return {
                "role": best["character"],
                "similarity": best["similarity"],
                "confidence": best.get("confidence", 0.0),
                "mode": "clip_faiss",
                "candidates": candidates,
            }
        except Exception as e:
            logger.error(f"识别失败: {e}")
            return {
                "role": "unknown",
                "similarity": 0.0,
                "mode": "error",
                "candidates": [],
                "error": str(e),
            }

    def classify_batch(
        self,
        image_inputs: List[Union[str, Image.Image, np.ndarray]],
        top_k: int = 5,
    ) -> List[Dict]:
        """批量分类"""
        if not self._initialized and not self.initialize():
            return [{
                "role": "unknown",
                "similarity": 0.0,
                "mode": "uninitialized",
                "candidates": [],
            } for _ in image_inputs]

        try:
            batch_results = self._retriever.identify_batch(image_inputs, top_k=top_k)

            results = []
            for candidates in batch_results:
                if not candidates:
                    results.append({
                        "role": "unknown",
                        "similarity": 0.0,
                        "mode": "clip_faiss",
                        "candidates": [],
                    })
                else:
                    best = candidates[0]
                    results.append({
                        "role": best["character"],
                        "similarity": best["similarity"],
                        "confidence": best.get("confidence", 0.0),
                        "mode": "clip_faiss",
                        "candidates": candidates,
                    })
            return results
        except Exception as e:
            logger.error(f"批量识别失败: {e}")
            return [{
                "role": "unknown",
                "similarity": 0.0,
                "mode": "error",
                "candidates": [],
                "error": str(e),
            } for _ in image_inputs]

    def add_character(
        self,
        character_name: str,
        image_paths: List[str],
        max_samples: int = 30,
    ) -> bool:
        """增量添加角色"""
        if not self._initialized and not self.initialize():
            return False

        result = self._retriever.register_character(
            character_name=character_name,
            image_paths=image_paths,
            max_samples=max_samples,
        )
        return result.get("success", False)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self._initialized:
            return {"initialized": False}
        return self._retriever.get_stats()

    def is_available(self) -> bool:
        """检查是否可用"""
        return self._initialized and self._retriever is not None


# 提供工厂函数
def get_clip_faiss_classifier() -> CLIPFaissAdapter:
    """获取单例分类器"""
    return CLIPFaissAdapter()