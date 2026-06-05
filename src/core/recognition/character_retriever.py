#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色检索器 - 整合 CLIP Embedder 和 Feature Store
Character Retriever - Combine CLIP Embedder and Feature Store

提供：
- 角色识别（图片 -> 角色名）
- 角色注册（添加新角色）
- 角色库管理
- 多模态融合（可选）
"""

import os
import platform
from pathlib import Path
from typing import List, Dict, Union
import numpy as np
from PIL import Image

if platform.system() == "Darwin":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from .clip_embedder import CLIPEmbedder
from .feature_store import FeatureStore
import logging

logger = logging.getLogger(__name__)


class CharacterRetriever:
    """
    角色检索器

    整合 CLIP 特征提取和 Faiss 特征检索
    替代传统分类模型，支持：
    - 增量学习（添加新角色无需重训）
    - 多特征融合
    - 相似度阈值过滤
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        feature_store_path: str = "data/feature_store/character_index.faiss",
        metadata_path: str = "data/feature_store/character_metadata.json",
        use_huggingface: bool = True,
        similarity_threshold: float = 0.5,
    ):
        """
        初始化角色检索器

        Args:
            clip_model_name: CLIP模型名称
            feature_store_path: 特征库路径
            metadata_path: 元数据路径
            use_huggingface: 是否使用HF CLIP
            similarity_threshold: 相似度阈值
        """
        self.embedder = CLIPEmbedder(
            model_name=clip_model_name,
            use_huggingface=use_huggingface,
        )
        # 根据模型名推断特征维度（不触发模型加载）
        if "ViT-H" in clip_model_name or "huge" in clip_model_name.lower():
            embedding_dim = 1024
        elif "ViT-L" in clip_model_name or "large" in clip_model_name.lower():
            embedding_dim = 768
        else:
            # ViT-B/32, ViT-B/16 等默认512维
            embedding_dim = 512
        self.feature_store = FeatureStore(
            dimension=embedding_dim,
            index_path=feature_store_path,
            metadata_path=metadata_path,
        )
        self.similarity_threshold = similarity_threshold
        self._initialized = False

        logger.info(
            f"CharacterRetriever创建: 模型={clip_model_name}, "
            f"特征库={feature_store_path}"
        )

    def initialize(self):
        """懒加载所有组件"""
        if self._initialized:
            return
        self.embedder.initialize()
        self.feature_store.initialize()
        self._initialized = True
        logger.info("✅ CharacterRetriever初始化完成")

    def register_character(
        self,
        character_name: str,
        image_paths: List[str],
        max_samples: int = 50,
        replace: bool = False,
        use_prototype: Union[bool, int] = False,
    ) -> Dict:
        """
        注册新角色到特征库

        Args:
            character_name: 角色名
            image_paths: 训练图片路径列表
            max_samples: 最多使用多少张图片
            replace: 是否替换已存在角色
            use_prototype: False=每图入库, True/1=单原型, >1=多原型(KMeans聚类数)

        Returns:
            注册结果统计
        """
        self.initialize()

        if not image_paths:
            return {"success": False, "error": "图片列表为空"}

        # 限制样本数
        image_paths = image_paths[:max_samples]

        # 提取特征
        logger.info(f"正在为角色 {character_name} 提取特征 ({len(image_paths)} 张)...")
        features = self.embedder.embed_images(image_paths, batch_size=8)

        # 过滤失败的
        valid_features = []
        valid_paths = []
        for feat, path in zip(features, image_paths):
            if feat is not None:
                valid_features.append(feat)
                valid_paths.append(path)

        if not valid_features:
            return {"success": False, "error": "没有有效特征被提取"}

        features_array = np.stack(valid_features)

        # 添加到特征库
        success = self.feature_store.add_character(
            character_name=character_name,
            features=features_array,
            image_paths=valid_paths,
            replace=replace,
            use_prototype=use_prototype,
        )

        # 不自动保存，让调用者控制保存时机

        return {
            "success": success,
            "character_name": character_name,
            "total_samples": len(image_paths),
            "valid_samples": len(valid_features),
            "feature_count": 1 if use_prototype else len(valid_features),
            "use_prototype": use_prototype,
        }

    def register_character_from_dir(
        self,
        character_name: str,
        directory: str,
        max_samples: int = 50,
        replace: bool = False,
        use_prototype: Union[bool, int] = False,
    ) -> Dict:
        """从目录注册角色"""
        dir_path = Path(directory)
        if not dir_path.exists():
            return {"success": False, "error": f"目录不存在: {directory}"}

        # 收集图片
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.gif')
        image_paths = []
        for ext in image_extensions:
            image_paths.extend([str(p) for p in dir_path.rglob(f"*{ext}")])
            image_paths.extend([str(p) for p in dir_path.rglob(f"*{ext.upper()}")])

        image_paths = sorted(set(image_paths))

        if not image_paths:
            return {"success": False, "error": f"目录中没有图片: {directory}"}

        return self.register_character(
            character_name=character_name,
            image_paths=image_paths,
            max_samples=max_samples,
            replace=replace,
            use_prototype=use_prototype,
        )

    def register_characters_from_dataset(
        self,
        dataset_dir: str = "data/final_dataset",
        max_samples_per_character: int = 30,
        skip_existing: bool = True,
        use_prototype: bool = False,
    ) -> List[Dict]:
        """
        从数据集目录批量注册所有角色

        Args:
            dataset_dir: 数据集根目录（每个子目录是一个角色）
            max_samples_per_character: 每个角色最多使用样本数
            skip_existing: 是否跳过已注册角色
            use_prototype: 是否使用原型向量

        Returns:
            注册结果列表
        """
        self.initialize()

        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            logger.error(f"数据集目录不存在: {dataset_dir}")
            return []

        results = []
        for character_dir in dataset_path.iterdir():
            if not character_dir.is_dir():
                continue

            character_name = character_dir.name

            if skip_existing and self.feature_store.has_character(character_name):
                logger.info(f"跳过已存在角色: {character_name}")
                results.append({
                    "success": True,
                    "character_name": character_name,
                    "skipped": True,
                })
                continue

            result = self.register_character_from_dir(
                character_name=character_name,
                directory=str(character_dir),
                max_samples=max_samples_per_character,
                use_prototype=use_prototype,
            )
            results.append(result)
            logger.info(f"注册角色: {character_name} - {result}")

        return results

    def identify(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        top_k: int = 5,
        return_threshold: bool = False,
        use_voting: bool = False,
        voting_top_n: int = 10,
    ) -> List[Dict]:
        """
        识别单张图片中的角色

        Args:
            image_input: 图片路径/PIL Image/numpy数组
            top_k: 返回前k个候选
            return_threshold: 是否返回相似度阈值
            use_voting: 是否使用Top-K投票（当每个角色有多个特征时有效）
            voting_top_n: 投票时使用的近邻数量

        Returns:
            识别结果列表，每个元素包含:
            - character: 角色名
            - similarity: 相似度
            - confidence: 置信度（归一化）
        """
        self.initialize()

        # 提取特征
        feature = self.embedder.embed_image(image_input)
        if feature is None:
            return []

        # 检查是否使用原型模式（每个角色只有一个特征）
        stats = self.feature_store.get_stats()
        total_chars = stats["total_characters"]
        total_features = stats["total_features"]
        using_prototype = (total_chars > 0) and (total_features == total_chars)
        
        # 如果使用原型模式，投票无效（每个角色只有一个向量）
        effective_voting = use_voting and not using_prototype

        # 检索
        results = self.feature_store.search(
            feature, 
            top_k=top_k,
            use_voting=effective_voting,
            voting_top_n=voting_top_n,
        )

        if not results:
            return []

        # 整理结果
        candidates = []
        for character_name, similarity in results[0]:
            candidates.append({
                "character": character_name,
                "similarity": similarity,
                "above_threshold": similarity >= self.similarity_threshold,
            })

        # 归一化置信度（使用softmax）
        if candidates:
            sims = np.array([c["similarity"] for c in candidates])
            exp_sims = np.exp((sims - sims.max()) * 5)  # 温度系数5
            confidences = exp_sims / exp_sims.sum()
            for i, c in enumerate(candidates):
                c["confidence"] = float(confidences[i])

        if return_threshold:
            for c in candidates:
                c["threshold"] = self.similarity_threshold

        return candidates

    def identify_batch(
        self,
        image_inputs: List[Union[str, Image.Image, np.ndarray]],
        top_k: int = 5,
        use_voting: bool = False,
        voting_top_n: int = 10,
    ) -> List[List[Dict]]:
        """批量识别"""
        self.initialize()

        # 批量提取特征
        features = self.embedder.embed_images(image_inputs, batch_size=8)

        # 批量检索
        valid_features = [f for f in features if f is not None]
        valid_indices = [i for i, f in enumerate(features) if f is not None]

        if not valid_features:
            return [[] for _ in image_inputs]

        features_array = np.stack(valid_features)
        batch_results = self.feature_store.search(
            features_array, 
            top_k=top_k,
            use_voting=use_voting,
            voting_top_n=voting_top_n,
        )

        # 整理结果
        all_results = [[] for _ in image_inputs]
        for batch_idx, results in enumerate(batch_results):
            orig_idx = valid_indices[batch_idx]
            candidates = []
            for character_name, similarity in results:
                candidates.append({
                    "character": character_name,
                    "similarity": similarity,
                    "above_threshold": similarity >= self.similarity_threshold,
                })
            # 归一化
            if candidates:
                sims = np.array([c["similarity"] for c in candidates])
                exp_sims = np.exp((sims - sims.max()) * 5)
                confidences = exp_sims / exp_sims.sum()
                for i, c in enumerate(candidates):
                    c["confidence"] = float(confidences[i])
            all_results[orig_idx] = candidates

        return all_results

    def get_character_count(self) -> int:
        """获取角色数量"""
        self.initialize()
        return self.feature_store.get_stats()["total_characters"]

    def get_stats(self) -> Dict:
        """获取统计信息"""
        self.initialize()
        return {
            "embedder": {
                "model": self.embedder.model_name,
                "dimension": self.embedder.embedding_dim,
                "device": self.embedder.device,
                "initialized": self.embedder.is_initialized(),
            },
            "feature_store": self.feature_store.get_stats(),
            "similarity_threshold": self.similarity_threshold,
        }

    def save(self) -> bool:
        """保存特征库"""
        return self.feature_store.save()

    def reload(self) -> bool:
        """重新加载特征库"""
        return self.feature_store.load()
