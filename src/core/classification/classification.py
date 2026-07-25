import faiss
import numpy as np
import json
import os
import hashlib
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass

from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("classification")


@dataclass
class ClassificationResult:
    """分类结果"""
    role: str
    similarity: float


class Classification:
    """角色分类模块 - 基于Faiss向量检索"""

    _index_cache: Dict[str, Tuple[faiss.Index, List[str]]] = {}
    _result_cache: Dict[str, ClassificationResult] = {}
    _cache_size = 100
    _index_cache_size = 2

    def __init__(self, index_path: Optional[str] = None, threshold: float = 0.4):
        self.threshold = threshold
        self.index: Optional[faiss.Index] = None
        self.role_mapping: List[str] = []

        logger.info(f"初始化分类模块，阈值: {threshold}")

        if index_path:
            self._try_load_index(index_path)
        else:
            self._try_load_default_index()

    def _try_load_index(self, index_path: str) -> None:
        """尝试加载索引文件"""
        try:
            faiss_path = self._resolve_path(index_path)
            mapping_path = self._get_mapping_path(faiss_path)

            if os.path.exists(faiss_path) and os.path.exists(mapping_path):
                logger.info(f"加载索引: {faiss_path}")
                self.load_index(faiss_path)
            else:
                logger.warning(f"索引文件不存在，创建空索引")
                self._create_empty_index()
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            self._create_empty_index()

    def _resolve_path(self, path: str) -> str:
        """解析路径为绝对路径"""
        if os.path.isabs(path):
            return path if path.endswith(".faiss") else f"{path}.faiss"

        project_root = self._get_project_root()
        faiss_path = os.path.join(project_root, path)
        return faiss_path if faiss_path.endswith(".faiss") else f"{faiss_path}.faiss"

    def _get_mapping_path(self, faiss_path: str) -> str:
        """获取映射文件路径"""
        return faiss_path.replace(".faiss", "_mapping.json")

    def _get_project_root(self) -> str:
        """获取项目根目录"""
        current_file = os.path.abspath(__file__)
        return os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

    def _try_load_default_index(self) -> None:
        """尝试从默认路径加载索引"""
        project_root = self._get_project_root()
        default_paths = [
            os.path.join(project_root, "data", "index", "faiss_index.faiss"),
            os.path.join(project_root, "models", "faiss_index.faiss"),
        ]

        for faiss_path in default_paths:
            if os.path.exists(faiss_path):
                logger.info(f"找到默认索引: {faiss_path}")
                self.load_index(faiss_path)
                return

        logger.info("未找到默认索引，创建空索引")
        self._create_empty_index()

    def _create_empty_index(self) -> None:
        """创建空索引"""
        dim = 512
        self.index = faiss.IndexFlatIP(dim)
        self.role_mapping = []
        logger.info(f"创建空索引完成，维度: {dim}")

    def build_index(self, features: np.ndarray, role_names: List[str]) -> None:
        """构建向量索引"""
        dim = features.shape[1]
        logger.info(f"构建索引，维度: {dim}, 特征数: {features.shape[0]}, 角色数: {len(role_names)}")

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(features)
        self.role_mapping = role_names

        logger.info(f"索引构建完成")

    def save_index(self, index_path: str) -> None:
        """保存索引到文件"""
        if self.index is None:
            raise ValueError("索引尚未构建")

        logger.info(f"保存索引到: {index_path}")

        faiss.write_index(self.index, f"{index_path}.faiss")

        with open(f"{index_path}_mapping.json", "w", encoding="utf-8") as f:
            json.dump(self.role_mapping, f, ensure_ascii=False, indent=2)

        logger.info(f"索引保存完成")

    def load_index(self, index_path: str) -> None:
        """从文件加载索引"""
        if index_path in self._index_cache:
            logger.info(f"从缓存加载索引: {index_path}")
            self.index, self.role_mapping = self._index_cache[index_path]
            return

        if len(self._index_cache) >= self._index_cache_size:
            oldest_key = next(iter(self._index_cache))
            del self._index_cache[oldest_key]
            logger.info(f"移除最早的索引缓存: {oldest_key}")

        logger.info(f"加载索引: {index_path}")

        self.index = faiss.read_index(index_path)
        mapping_path = self._get_mapping_path(index_path)

        with open(mapping_path, "r", encoding="utf-8") as f:
            self.role_mapping = json.load(f)

        self._index_cache[index_path] = (self.index, self.role_mapping)

        logger.info(f"索引加载完成，角色数: {len(self.role_mapping)}")

    def _get_feature_hash(self, feature: np.ndarray) -> str:
        """计算特征哈希"""
        if len(feature.shape) > 1:
            feature = feature.flatten()
        return hashlib.md5(feature.tobytes()).hexdigest()

    def _cache_result(self, feature: np.ndarray, result: ClassificationResult) -> None:
        """缓存结果"""
        cache_key = self._get_feature_hash(feature)

        if len(self._result_cache) >= self._cache_size:
            oldest_key = next(iter(self._result_cache))
            del self._result_cache[oldest_key]

        self._result_cache[cache_key] = result

    def _get_cached_result(self, feature: np.ndarray) -> Optional[ClassificationResult]:
        """获取缓存结果"""
        return self._result_cache.get(self._get_feature_hash(feature))

    def classify(
        self,
        feature: np.ndarray,
        top_k: int = 5,
        tags: Optional[List[Any]] = None,
        multilabel: bool = False,
        threshold: Optional[float] = None,
    ) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
        """分类单个特征向量"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("索引为空")
            return [] if multilabel else ("unknown", 0.0)

        current_threshold = threshold if threshold is not None else self.threshold

        if len(feature.shape) == 1:
            feature = feature.reshape(1, -1)

        distances, indices = self.index.search(feature, top_k)

        results = []
        for i in range(top_k):
            idx = indices[0][i]
            inner_product = distances[0][i]

            if idx < len(self.role_mapping):
                similarity = (inner_product + 1.0) / 2.0
                results.append({"role": self.role_mapping[idx], "similarity": float(similarity)})

        if not results:
            return [] if multilabel else ("unknown", 0.0)

        role_similarities = self._aggregate_similarities(results)
        sorted_roles = sorted(role_similarities.items(), key=lambda x: x[1], reverse=True)

        if tags:
            sorted_roles = self._apply_tag_filter(sorted_roles, tags)

        if multilabel:
            return [
                (role, sim)
                for role, sim in sorted_roles
                if sim >= current_threshold
            ]
        else:
            return sorted_roles[0] if sorted_roles else ("unknown", 0.0)

    def _aggregate_similarities(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """聚合相似度"""
        role_similarities = {}
        role_counts = {}

        for result in results:
            role = result["role"]
            similarity = result["similarity"]

            role_similarities[role] = role_similarities.get(role, 0) + similarity
            role_counts[role] = role_counts.get(role, 0) + 1

        return {
            role: total / role_counts[role]
            for role, total in role_similarities.items()
        }

    def _apply_tag_filter(
        self,
        sorted_roles: List[Tuple[str, float]],
        tags: List[Any],
    ) -> List[Tuple[str, float]]:
        """应用标签过滤"""
        tags_lower = self._normalize_tags(tags)

        for role, similarity in sorted_roles:
            if self._role_matches_tags(role, tags_lower):
                return [(role, similarity)]

        return sorted_roles

    def _normalize_tags(self, tags: List[Any]) -> List[str]:
        """标准化标签"""
        tags_lower = []
        for tag in tags:
            if isinstance(tag, dict) and "tag" in tag:
                tags_lower.append(tag["tag"].lower())
            elif isinstance(tag, str):
                tags_lower.append(tag.lower())
        return tags_lower

    def _role_matches_tags(self, role: str, tags_lower: List[str]) -> bool:
        """检查角色是否匹配标签"""
        tag_keywords = {
            "日奈": ["hina", "日奈"],
            "伊织": ["izumi", "伊织"],
            "阿罗娜": ["arona", "阿罗娜", "a1luo2na4"],
            "普拉娜": ["prana", "普拉娜"],
        }

        for tag in tags_lower:
            for role_name, keywords in tag_keywords.items():
                if role == role_name and any(kw in tag for kw in keywords):
                    return True
        return False

    def batch_classify(
        self, features: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """批量分类"""
        if self.index is None:
            raise ValueError("索引尚未构建")

        logger.info(f"批量分类，特征数: {features.shape[0]}, top_k: {top_k}")

        distances, indices = self.index.search(features, top_k)

        batch_results = []
        for i in range(features.shape[0]):
            results = []
            for j in range(top_k):
                idx = indices[i][j]
                distance = distances[i][j]

                if idx < len(self.role_mapping):
                    similarity = 1.0 / (1.0 + distance)
                    results.append({
                        "role": self.role_mapping[idx],
                        "similarity": float(similarity)
                    })

            if not results:
                batch_results.append(("unknown", 0.0))
                continue

            role_similarities = self._aggregate_similarities(results)
            sorted_roles = sorted(role_similarities.items(), key=lambda x: x[1], reverse=True)

            if sorted_roles and sorted_roles[0][1] >= self.threshold:
                batch_results.append(sorted_roles[0])
            else:
                batch_results.append(("unknown", results[0]["similarity"] if results else 0.0))

        logger.info(f"批量分类完成")
        return batch_results

    def classify_multiple_characters(
        self, characters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """分类多个角色"""
        if self.index is None:
            raise ValueError("索引尚未构建")

        logger.info(f"多角色分类，角色数: {len(characters)}")

        try:
            features = np.array([char["feature"] for char in characters], dtype=np.float32)
            results = self.batch_classify(features)

            for i, char in enumerate(characters):
                char["role"], char["similarity"] = results[i]

            logger.info(f"多角色分类完成")
            return characters
        except Exception as e:
            logger.error(f"多角色分类失败: {e}")
            return []

    def update_index(self, new_features: np.ndarray, new_role_names: List[str]) -> None:
        """更新索引"""
        if self.index is None:
            logger.info(f"索引不存在，构建新索引")
            self.build_index(new_features, new_role_names)
        else:
            logger.info(f"更新索引，添加特征数: {new_features.shape[0]}")
            self.index.add(new_features)
            self.role_mapping.extend(new_role_names)
            logger.info(f"索引更新完成，角色数: {len(self.role_mapping)}")

    def incremental_learning(self, image_path: str, correct_role: str) -> bool:
        """增量学习"""
        from src.core.preprocessing.preprocessing import Preprocessing
        from src.core.feature_extraction.feature_extraction import FeatureExtraction

        logger.info(f"增量学习，图像: {image_path}, 角色: {correct_role}")

        try:
            preprocessor = Preprocessing()
            extractor = FeatureExtraction()

            normalized_img, _ = preprocessor.process(image_path)
            feature = extractor.extract_features(normalized_img)
            feature = feature.reshape(1, -1)

            self.update_index(feature, [correct_role])

            logger.info(f"增量学习成功")
            return True
        except Exception as e:
            logger.error(f"增量学习失败: {e}")
            return False


if __name__ == "__main__":
    import sys

    classifier = Classification(threshold=0.7)

    dim = 512
    num_roles = 2
    num_samples_per_role = 5

    features = np.random.randn(num_roles * num_samples_per_role, dim).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    role_names = []
    for i in range(num_roles):
        role_name = f"角色{i+1}"
        role_names.extend([role_name] * num_samples_per_role)

    classifier.build_index(features, role_names)

    index_path = "test_index"
    classifier.save_index(index_path)
    print("索引已保存")

    new_classifier = Classification()
    new_classifier.load_index(index_path)
    print("索引已加载")

    test_feature = np.random.randn(dim).astype(np.float32)
    test_feature = test_feature / np.linalg.norm(test_feature)

    role, similarity = new_classifier.classify(test_feature)
    print(f"分类结果: 角色={role}, 相似度={similarity:.4f}")
