#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-set 识别模块 - 未知角色阈值检测
当模型提取的特征与库中所有角色的相似度都低于某个值时，判定为"未知角色"
"""

import os
import sys
import json
import numpy as np
import faiss
from typing import List, Tuple, Optional

# 添加项目根目录
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

class OpenSetRecognizer:
    """Open-set 角色识别器"""

    def __init__(
        self,
        index_path: str,
        mapping_path: str,
        role_info_path: str,
        unknown_threshold: float = 0.3,
        fuzzy_threshold: float = 0.5,
    ):
        """
        初始化 Open-set 识别器

        Args:
            index_path: FAISS 索引文件路径
            mapping_path: 角色映射文件路径
            role_info_path: 角色信息文件路径
            unknown_threshold: 未知角色阈值（低于此值判定为未知）
            fuzzy_threshold: 模糊样本阈值（用于记录待标注样本）
        """
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.role_info_path = role_info_path
        self.unknown_threshold = unknown_threshold
        self.fuzzy_threshold = fuzzy_threshold

        self.index = None
        self.role_mapping = []
        self.role_info = {}

        self._load_index()
        self._load_mapping()
        self._load_role_info()

    def _load_index(self):
        """加载 FAISS 索引"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"✅ FAISS 索引加载成功: {self.index.ntotal} 个向量")
        else:
            raise FileNotFoundError(f"索引文件不存在: {self.index_path}")

    def _load_mapping(self):
        """加载角色映射"""
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.role_mapping = json.load(f)
            print(f"✅ 角色映射加载成功: {len(self.role_mapping)} 个角色")
        else:
            print(f"⚠️ 映射文件不存在，使用空映射")
            self.role_mapping = []

    def _load_role_info(self):
        """加载角色信息"""
        if os.path.exists(self.role_info_path):
            with open(self.role_info_path, "r", encoding="utf-8") as f:
                self.role_info = json.load(f)
            print(f"✅ 角色信息加载成功: {len(self.role_info)} 个角色")
        else:
            print(f"⚠️ 角色信息文件不存在")

    def compute_similarity(self, distance: float) -> float:
        """将 FAISS L2 距离转换为相似度 (0-1)"""
        return 1.0 / (1.0 + distance)

    def recognize(self, feature_vector: np.ndarray, top_k: int = 5) -> dict:
        """
        识别角色

        Args:
            feature_vector: 特征向量 (1D 或 2D array)
            top_k: 返回前 k 个候选

        Returns:
            识别结果字典
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # 搜索最近邻
        distances, indices = self.index.search(feature_vector.astype("float32"), top_k)

        results = {
            "is_known": True,
            "is_fuzzy": False,
            "is_unknown": False,
            "predictions": [],
            "max_similarity": 0.0,
            "decision": "unknown",
        }

        if len(distances[0]) == 0:
            results["is_unknown"] = True
            results["decision"] = "unknown"
            return results

        # 处理每个预测结果
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.role_mapping):
                role_name = self.role_mapping[idx]
                similarity = self.compute_similarity(dist)

                role_detail = {
                    "rank": i + 1,
                    "role": role_name,
                    "distance": float(dist),
                    "similarity": float(similarity),
                    "en": role_name,
                    "cn": role_name,
                    "jp": "",
                    "anime": "",
                }

                # 补充角色信息
                if role_name in self.role_info:
                    info = self.role_info[role_name]
                    role_detail.update(
                        {
                            "en": info.get("en", role_name),
                            "cn": info.get("cn", role_name),
                            "jp": info.get("jp", ""),
                            "anime": info.get("anime", ""),
                        }
                    )

                results["predictions"].append(role_detail)

                # 更新最大相似度
                if i == 0:
                    results["max_similarity"] = similarity

        # 决策逻辑
        top_similarity = results["max_similarity"]

        if top_similarity < self.unknown_threshold:
            results["is_unknown"] = True
            results["is_known"] = False
            results["decision"] = "unknown"
        elif top_similarity < self.fuzzy_threshold:
            results["is_fuzzy"] = True
            results["decision"] = "fuzzy"
        else:
            results["decision"] = "known"

        return results

    def batch_recognize(self, feature_vectors: np.ndarray, top_k: int = 5) -> List[dict]:
        """批量识别"""
        if feature_vectors.ndim == 1:
            feature_vectors = feature_vectors.reshape(1, -1)

        distances, indices = self.index.search(feature_vectors.astype("float32"), top_k)

        results = []
        for i in range(len(feature_vectors)):
            single_result = {
                "is_known": True,
                "is_fuzzy": False,
                "is_unknown": False,
                "predictions": [],
                "max_similarity": 0.0,
                "decision": "unknown",
            }

            for j, (idx, dist) in enumerate(zip(indices[i], distances[i])):
                if idx < len(self.role_mapping):
                    role_name = self.role_mapping[idx]
                    similarity = self.compute_similarity(dist)

                    role_detail = {
                        "rank": j + 1,
                        "role": role_name,
                        "distance": float(dist),
                        "similarity": float(similarity),
                    }

                    if role_name in self.role_info:
                        info = self.role_info[role_name]
                        role_detail.update(
                            {
                                "en": info.get("en", role_name),
                                "cn": info.get("cn", role_name),
                                "jp": info.get("jp", ""),
                                "anime": info.get("anime", ""),
                            }
                        )

                    single_result["predictions"].append(role_detail)

                    if j == 0:
                        single_result["max_similarity"] = similarity

            top_similarity = single_result["max_similarity"]

            if top_similarity < self.unknown_threshold:
                single_result["is_unknown"] = True
                single_result["is_known"] = False
                single_result["decision"] = "unknown"
            elif top_similarity < self.fuzzy_threshold:
                single_result["is_fuzzy"] = True
                single_result["decision"] = "fuzzy"

            results.append(single_result)

        return results

    def update_thresholds(self, unknown_threshold: float, fuzzy_threshold: float):
        """更新阈值"""
        self.unknown_threshold = unknown_threshold
        self.fuzzy_threshold = fuzzy_threshold
        print(f"✅ 阈值已更新: unknown={unknown_threshold}, fuzzy={fuzzy_threshold}")

    def get_statistics(self, feature_vectors: np.ndarray, top_k: int = 1) -> dict:
        """获取识别统计信息"""
        if feature_vectors.ndim == 1:
            feature_vectors = feature_vectors.reshape(1, -1)

        distances, _ = self.index.search(feature_vectors.astype("float32"), top_k)
        similarities = [self.compute_similarity(d) for d in distances[0]]

        stats = {
            "total_samples": len(similarities),
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "unknown_count": sum(1 for s in similarities if s < self.unknown_threshold),
            "fuzzy_count": sum(
                1 for s in similarities if self.unknown_threshold <= s < self.fuzzy_threshold
            ),
            "known_count": sum(1 for s in similarities if s >= self.fuzzy_threshold),
        }

        stats["unknown_ratio"] = (
            stats["unknown_count"] / stats["total_samples"] if stats["total_samples"] > 0 else 0
        )
        stats["fuzzy_ratio"] = (
            stats["fuzzy_count"] / stats["total_samples"] if stats["total_samples"] > 0 else 0
        )
        stats["known_ratio"] = (
            stats["known_count"] / stats["total_samples"] if stats["total_samples"] > 0 else 0
        )

        return stats


def main():
    """测试 Open-set 识别器"""
    # 路径配置
    MODEL_NAME = "efficientnet_b3_loli_optimized_v2_20260529_133654"
    MODEL_DIR = os.path.join(project_root, "models", MODEL_NAME)

    index_path = os.path.join(MODEL_DIR, "role_index_final.faiss")
    mapping_path = os.path.join(MODEL_DIR, "role_index_final_mapping.json")
    role_info_path = os.path.join(project_root, "src", "core", "data", "role_info.json")

    print("=" * 60)
    print("🔍 Open-set 角色识别测试")
    print("=" * 60)

    # 创建识别器
    recognizer = OpenSetRecognizer(
        index_path=index_path,
        mapping_path=mapping_path,
        role_info_path=role_info_path,
        unknown_threshold=0.3,
        fuzzy_threshold=0.5,
    )

    # 生成一些随机测试向量
    print("\n📊 测试识别功能...")
    test_vectors = np.random.randn(10, 512).astype("float32")

    results = recognizer.batch_recognize(test_vectors, top_k=3)

    # 统计
    unknown_count = sum(1 for r in results if r["is_unknown"])
    fuzzy_count = sum(1 for r in results if r["is_fuzzy"])
    known_count = sum(1 for r in results if r["decision"] == "known")

    print(f"\n📋 测试结果统计:")
    print(f"   总样本数: {len(results)}")
    print(f"   未知角色: {unknown_count} ({unknown_count/len(results)*100:.1f}%)")
    print(f"   模糊样本: {fuzzy_count} ({fuzzy_count/len(results)*100:.1f}%)")
    print(f"   已知角色: {known_count} ({known_count/len(results)*100:.1f}%)")

    print("\n💡 阈值说明:")
    print(f"   unknown_threshold=0.3: 相似度 < 0.3 判定为未知角色")
    print(f"   fuzzy_threshold=0.5: 0.3 <= 相似度 < 0.5 记录为模糊样本")
    print(f"   已知角色: 相似度 >= 0.5")


if __name__ == "__main__":
    main()
