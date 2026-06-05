#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征库管理 - 基于 Faiss
Feature Store - Faiss-based

提供：
- 角色特征库构建
- 特征库增量更新
- 特征库持久化
- 特征库加载
"""

import os
import json
import platform
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import numpy as np

if platform.system() == "Darwin":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import faiss
import logging

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    特征库管理

    使用 Faiss IndexFlatIP (内积) 存储归一化特征向量
    支持：
    - 构建索引
    - 添加特征
    - 删除特征
    - 持久化存储
    - 元数据管理
    """

    def __init__(
        self,
        dimension: int = 512,
        index_path: str = "data/feature_store/character_index.faiss",
        metadata_path: str = "data/feature_store/character_metadata.json",
        index_type: str = "FlatIP",
    ):
        """
        初始化特征库

        Args:
            dimension: 特征维度
            index_path: Faiss索引保存路径
            metadata_path: 元数据保存路径
            index_type: 索引类型 (FlatIP, FlatL2, IVFFlat)
        """
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index_type = index_type

        # 索引和元数据
        self._index: Optional[faiss.Index] = None
        self._metadata: Dict[str, dict] = {}  # character_name -> info
        self._id_to_idx: Dict[int, str] = {}  # faiss_id -> character_name
        self._idx_to_id: Dict[str, int] = {}  # character_name -> faiss_id

        # 统计
        self._stats = {
            "total_features": 0,
            "total_characters": 0,
            "created_at": None,
            "updated_at": None,
        }

        # 创建目录
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_index(self) -> faiss.Index:
        """创建新的Faiss索引"""
        if self.index_type == "FlatIP":
            # 内积相似度（适合归一化向量，等价于余弦相似度）
            index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "FlatL2":
            # L2距离
            index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVFFlat":
            # IVF索引（适合大规模数据）
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = min(100, max(1, self._stats["total_characters"] // 10))
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            index.nprobe = min(10, nlist)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

        return index

    def initialize(self):
        """初始化（懒加载）"""
        if self._index is not None:
            return

        # 尝试从磁盘加载
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.load()
                logger.info(f"特征库已加载: {len(self._metadata)} 个角色, {self._stats['total_features']} 个特征")
                return
            except Exception as e:
                logger.warning(f"加载特征库失败，将创建新的: {e}")

        # 创建新索引
        self._index = self._create_index()
        self._metadata = {}
        self._id_to_idx = {}
        self._idx_to_id = {}
        self._stats["created_at"] = datetime.now().isoformat()
        logger.info(f"创建新特征库 (维度={self.dimension}, 类型={self.index_type})")

    def add_character(
        self,
        character_name: str,
        features: np.ndarray,
        image_paths: Optional[List[str]] = None,
        replace: bool = False,
        use_prototype: Union[bool, int] = False,
    ) -> bool:
        """
        添加角色特征到特征库

        Args:
            character_name: 角色名
            features: 特征向量 (N, D) 或 (D,)
            image_paths: 对应的图片路径
            replace: 是否替换已存在的角色
            use_prototype: False=每图入库, True/1=单原型, >1=多原型(KMeans聚类数)
            use_prototype: 是否使用原型向量（将所有特征平均为一个中心向量）

        Returns:
            是否成功
        """
        self.initialize()

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != self.dimension:
            raise ValueError(
                f"特征维度不匹配: 期望 {self.dimension}, 实际 {features.shape[1]}"
            )

        # 检查是否已存在
        if character_name in self._metadata and not replace:
            return self.add_features_to_character(character_name, features, image_paths, use_prototype=use_prototype)

        # 归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features_normalized = features / norms
        features_normalized = features_normalized.astype(np.float32)

        # 多原型模式：使用KMeans聚类
        if use_prototype and use_prototype > 1:
            k = min(use_prototype, len(features_normalized))
            from sklearn.cluster import KMeans
            
            print(f"DEBUG: Multi-prototype mode, k={k}, features={len(features_normalized)}")
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features_normalized)
            
            # 获取聚类中心并归一化
            prototypes = kmeans.cluster_centers_
            norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            prototypes = prototypes / norms
            
            features_to_add = prototypes.astype(np.float32)
            original_feature_count = len(features)
            use_prototype = k  # 记录实际使用的聚类数
            print(f"DEBUG: Generated {len(features_to_add)} prototypes")
        # 单原型模式：计算平均向量
        elif use_prototype == 1:
            prototype = np.mean(features_normalized, axis=0, keepdims=True)
            # 再次归一化
            norm = np.linalg.norm(prototype)
            if norm > 0:
                prototype = prototype / norm
            features_to_add = prototype
            original_feature_count = len(features)
        else:
            features_to_add = features_normalized
            original_feature_count = len(features)

        # 添加到索引
        start_id = self._index.ntotal
        self._index.add(features_to_add)

        # 更新元数据
        character_info = {
            "name": character_name,
            "feature_count": len(features_to_add),
            "original_feature_count": original_feature_count,
            "start_idx": start_id,
            "end_idx": start_id + len(features_to_add) - 1,
            "image_paths": image_paths or [],
            "use_prototype": use_prototype,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._metadata[character_name] = character_info

        # 更新映射
        for i in range(len(features_to_add)):
            idx = start_id + i
            self._id_to_idx[idx] = character_name
            self._idx_to_id[character_name] = idx  # 最后的位置

        # 更新统计
        self._stats["total_features"] = self._index.ntotal
        self._stats["total_characters"] = len(self._metadata)
        self._stats["updated_at"] = datetime.now().isoformat()

        logger.info(
            f"添加角色: {character_name}, 特征数: {len(features_to_add)}, "
            f"原始特征数: {original_feature_count}, 原型模式: {use_prototype}, "
            f"总特征数: {self._stats['total_features']}"
        )
        return True

    def add_features_to_character(
        self,
        character_name: str,
        features: np.ndarray,
        image_paths: Optional[List[str]] = None,
        use_prototype: Union[bool, int] = False,
    ) -> bool:
        """向已存在角色添加新特征"""
        if character_name not in self._metadata:
            return self.add_character(character_name, features, image_paths, replace=False, use_prototype=use_prototype)

        # 对于原型模式，需要重建角色
        info = self._metadata[character_name]
        if info.get("use_prototype") or use_prototype:
            logger.info(f"角色 {character_name} 使用原型模式，将重建索引")
            # 获取已有特征（需要重新计算原型）
            old_features = self._get_character_features(character_name)
            if old_features is not None:
                features = np.concatenate([old_features, features])
            
            # 删除旧的并重新添加
            self._remove_character_internal(character_name)
            return self.add_character(character_name, features, image_paths, replace=True, use_prototype=use_prototype)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features = features / norms
        features = features.astype(np.float32)

        # 添加到索引
        start_id = self._index.ntotal
        self._index.add(features)

        # 更新角色信息
        info["feature_count"] += len(features)
        info["end_idx"] = start_id + len(features) - 1
        info["updated_at"] = datetime.now().isoformat()
        if image_paths:
            info["image_paths"].extend(image_paths)

        # 更新映射
        for i in range(len(features)):
            self._id_to_idx[start_id + i] = character_name

        self._stats["total_features"] = self._index.ntotal
        self._stats["updated_at"] = datetime.now().isoformat()

        logger.info(
            f"为角色 {character_name} 添加 {len(features)} 个特征, "
            f"总数: {info['feature_count']}"
        )
        return True

    def _get_character_features(self, character_name: str) -> Optional[np.ndarray]:
        """获取角色的所有特征向量（内部方法）"""
        if character_name not in self._metadata:
            return None
        
        info = self._metadata[character_name]
        start_idx = info["start_idx"]
        end_idx = info["end_idx"]
        
        # 直接从索引获取特征
        if hasattr(self._index, 'reconstruct_n'):
            return self._index.reconstruct_n(start_idx, end_idx - start_idx + 1)
        elif hasattr(self._index, 'reconstruct'):
            features = []
            for i in range(start_idx, end_idx + 1):
                features.append(self._index.reconstruct(i))
            return np.array(features) if features else None
        return None

    def _remove_character_internal(self, character_name: str):
        """内部方法：移除角色（需要重建索引）"""
        if character_name not in self._metadata:
            return
        
        # 保存其他角色的数据（包括 use_prototype 设置）
        other_chars = []
        for char_name in self._metadata:
            if char_name != character_name:
                features = self._get_character_features(char_name)
                if features is not None:
                    use_prototype = self._metadata[char_name].get("use_prototype", False)
                    other_chars.append((char_name, features, self._metadata[char_name]["image_paths"], use_prototype))
        
        # 重建索引
        self._index = self._create_index()
        self._metadata = {}
        self._id_to_idx = {}
        self._idx_to_id = {}
        self._stats["total_features"] = 0
        self._stats["total_characters"] = 0
        
        # 重新添加其他角色
        for char_name, features, paths, use_prototype in other_chars:
            self.add_character(char_name, features, paths, use_prototype=use_prototype)

    def remove_character(self, character_name: str) -> bool:
        """
        移除角色

        注意: Faiss的IndexFlat不支持直接删除单个向量
        需要重建索引以实现删除
        """
        if character_name not in self._metadata:
            return False

        logger.warning(
            f"移除角色 {character_name} 将触发索引重建 (Faiss FlatIP 不支持直接删除)"
        )

        # 由于 Faiss FlatIP 不支持按角色查询特征
        # 简化方案: 暂不实现删除（业务中很少删除）
        logger.warning("特征库暂不支持删除角色，请使用重建整个特征库")
        return False

    def search(
        self,
        query_features: np.ndarray,
        top_k: int = 5,
        use_voting: bool = False,
        voting_top_n: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        搜索最相似的角色

        Args:
            query_features: 查询特征 (N, D) 或 (D,)
            top_k: 返回前k个结果
            use_voting: 是否使用Top-K投票（统计多个近邻中角色出现次数）
            voting_top_n: 投票时使用的近邻数量

        Returns:
            每个查询的结果列表: [(character_name, similarity), ...]
        """
        self.initialize()

        if self._index.ntotal == 0:
            n_queries = 1 if query_features.ndim == 1 else len(query_features)
            return [[] for _ in range(n_queries)]

        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)

        # 归一化
        norms = np.linalg.norm(query_features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        query_features = query_features / norms
        query_features = query_features.astype(np.float32)

        # 搜索
        if use_voting:
            # 使用更大的top_n进行投票
            search_top_k = min(voting_top_n, self._index.ntotal)
        else:
            search_top_k = min(top_k, self._index.ntotal)
        
        similarities, indices = self._index.search(query_features, search_top_k)

        # 整理结果
        results = []
        for sim_row, idx_row in zip(similarities, indices):
            if use_voting:
                # Top-K投票：统计角色出现次数
                votes = {}
                weighted_votes = {}
                for i, (sim, idx) in enumerate(zip(sim_row, idx_row)):
                    if idx < 0:
                        continue
                    character_name = self._id_to_idx.get(int(idx), "unknown")
                    if character_name not in votes:
                        votes[character_name] = 0
                        weighted_votes[character_name] = 0.0
                    votes[character_name] += 1
                    # 使用更平滑的指数衰减权重（距离越近权重越高）
                    # 指数衰减: weight = exp(-i/3)，前几个结果权重显著更高
                    weight = np.exp(-i / 3)
                    weighted_votes[character_name] += sim * weight
                
                # 综合排序：票数占60%权重，加权相似度占40%权重
                combined_scores = {}
                for char in votes:
                    # 票数归一化（除以总票数）
                    vote_score = votes[char] / len(idx_row)
                    # 相似度归一化（除以最大相似度）
                    max_sim = max(weighted_votes.values()) if weighted_votes else 1.0
                    sim_score = weighted_votes[char] / max_sim if max_sim > 0 else 0
                    # 综合得分
                    combined_scores[char] = 0.6 * vote_score + 0.4 * sim_score
                
                # 按综合得分排序
                sorted_chars = sorted(
                    combined_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k]
                result = [(char, score) for char, score in sorted_chars]
            else:
                result = []
                for sim, idx in zip(sim_row, idx_row):
                    if idx < 0:
                        continue
                    character_name = self._id_to_idx.get(int(idx), "unknown")
                    result.append((character_name, float(sim)))
            
            results.append(result)

        return results

    def has_character(self, character_name: str) -> bool:
        """检查角色是否存在"""
        return character_name in self._metadata

    def get_character_info(self, character_name: str) -> Optional[Dict]:
        """获取角色信息"""
        return self._metadata.get(character_name)

    def list_characters(self) -> List[str]:
        """列出所有角色"""
        return list(self._metadata.keys())

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self._stats,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "characters": self.list_characters(),
        }

    def save(self) -> bool:
        """保存特征库到磁盘"""
        try:
            self.initialize()

            # 保存Faiss索引
            faiss.write_index(self._index, str(self.index_path))

            # 保存元数据
            save_data = {
                "metadata": self._metadata,
                "id_to_idx": {str(k): v for k, v in self._id_to_idx.items()},
                "stats": self._stats,
                "dimension": self.dimension,
                "index_type": self.index_type,
            }
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            logger.info(
                f"特征库已保存: {self._stats['total_characters']} 个角色, "
                f"{self._stats['total_features']} 个特征"
            )
            return True

        except Exception as e:
            logger.error(f"保存特征库失败: {e}")
            return False

    def load(self) -> bool:
        """从磁盘加载特征库"""
        try:
            # 加载Faiss索引
            self._index = faiss.read_index(str(self.index_path))

            # 加载元数据
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                save_data = json.load(f)

            self._metadata = save_data["metadata"]
            self._id_to_idx = {int(k): v for k, v in save_data["id_to_idx"].items()}
            self._stats = save_data["stats"]
            self.dimension = save_data.get("dimension", self.dimension)
            self.index_type = save_data.get("index_type", self.index_type)

            logger.info(
                f"特征库加载完成: {len(self._metadata)} 个角色, "
                f"{self._stats['total_features']} 个特征"
            )
            return True

        except Exception as e:
            logger.error(f"加载特征库失败: {e}")
            return False

    def clear(self):
        """清空特征库"""
        self._index = self._create_index()
        self._metadata = {}
        self._id_to_idx = {}
        self._idx_to_id = {}
        self._stats = {
            "total_features": 0,
            "total_characters": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        logger.info("特征库已清空")
