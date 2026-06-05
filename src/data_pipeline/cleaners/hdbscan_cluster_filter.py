#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDBSCAN聚类过滤器
使用HDBSCAN聚类算法检测异常图片
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("hdbscan_cluster_filter")


class HDBSCANClusterFilter:
    """
    HDBSCAN聚类过滤器
    
    使用HDBSCAN（层次密度聚类）识别图片簇，
    将不属于任何簇的异常点（边缘图片、错误标注等）检测出来
    """
    
    def __init__(
        self,
        embedder=None,
        min_cluster_size: int = 5,
        min_samples: int = 5,
        cluster_selection_epsilon: float = 0.0,
        device: Optional[str] = None,
    ):
        """
        初始化过滤器
        
        Args:
            embedder: CLIP特征提取器
            min_cluster_size: 最小簇大小
            min_samples: 最小采样数
            cluster_selection_epsilon: 簇选择epsilon
            device: 运行设备
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        
        if embedder is None:
            from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached
            self.embedder = CLIPEmbedderCached(
                model_name="ViT-B/32",
                device=device,
            )
        else:
            self.embedder = embedder
        
        logger.info(f"HDBSCAN过滤器初始化: min_cluster={min_cluster_size}")
    
    def extract_features(self, image_paths: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        批量提取特征
        
        Returns:
            (有效路径列表, 特征矩阵)
        """
        features = []
        valid_paths = []
        
        for path in tqdm(image_paths, desc="提取特征"):
            feat = self.embedder.embed_image(path)
            if feat is not None:
                features.append(feat)
                valid_paths.append(path)
        
        if features:
            features = np.array(features)
            # 特征已经是归一化的
            
        return valid_paths, features
    
    def cluster(
        self,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行HDBSCAN聚类
        
        Returns:
            (聚类标签, 异常分数)
        """
        try:
            import hdbscan
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric='euclidean',
                prediction_data=True,
            )
            
            labels = clusterer.fit_predict(features)
            probabilities = clusterer.probabilities_
            outlier_scores = 1 - probabilities  # 异常分数
            
            return labels, outlier_scores
            
        except ImportError:
            logger.warning("HDBSCAN未安装，使用sklearn替代")
            return self._cluster_sklearn(features)
    
    def _cluster_sklearn(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用sklearn的DBSCAN作为替代"""
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        
        # 使用DBSCAN
        clustering = DBSCAN(
            eps=0.5,
            min_samples=self.min_samples,
            metric='euclidean',
        )
        labels = clustering.fit_predict(features)
        
        # 计算异常分数（到最近簇的距离）
        outlier_scores = np.zeros(len(features))
        
        unique_labels = set(labels)
        unique_labels.discard(-1)  # 移除噪声标签
        
        if unique_labels:
            for i, label in enumerate(labels):
                if label == -1:
                    # 噪声点，计算到最近簇的距离
                    outlier_scores[i] = 1.0
        
        return labels, outlier_scores
    
    def filter_outliers(
        self,
        image_paths: List[str],
        outlier_threshold: float = 0.7,
        return_scores: bool = False,
    ) -> List[str]:
        """
        过滤异常图片
        
        Args:
            image_paths: 图片路径列表
            outlier_threshold: 异常阈值，超过此值认为是异常
            return_scores: 是否返回分数
            
        Returns:
            过滤后的图片列表，或 [(路径, 异常分数)] 列表
        """
        if len(image_paths) < self.min_cluster_size:
            logger.warning(f"图片数量不足 ({len(image_paths)} < {self.min_cluster_size})")
            return image_paths if not return_scores else [(p, 0.0) for p in image_paths]
        
        # 提取特征
        valid_paths, features = self.extract_features(image_paths)
        
        if len(valid_paths) < self.min_cluster_size:
            return valid_paths if not return_scores else [(p, 0.0) for p in valid_paths]
        
        # 聚类
        labels, outlier_scores = self.cluster(features)
        
        # 分类
        results = list(zip(valid_paths, labels, outlier_scores))
        kept = [(p, s) for p, l, s in results if l != -1 and s < outlier_threshold]
        outliers = [(p, l, s) for p, l, s in results if l == -1 or s >= outlier_threshold]
        
        # 统计
        n_clusters = len(set(labels) - {-1})
        n_outliers = len(outliers)
        
        logger.info(f"聚类结果: {n_clusters} 个簇, {n_outliers} 个异常点")
        
        if return_scores:
            return [(p, s) for p, _, s in results]
        
        return [p for p, _ in kept]
    
    def analyze_clusters(
        self,
        image_paths: List[str],
    ) -> Dict:
        """
        分析聚类结果
        
        Returns:
            聚类分析结果
        """
        if len(image_paths) < self.min_cluster_size:
            return {"message": "图片数量不足"}
        
        valid_paths, features = self.extract_features(image_paths)
        
        if len(valid_paths) < self.min_cluster_size:
            return {"message": "有效图片数量不足"}
        
        labels, outlier_scores = self.cluster(features)
        
        # 统计每个簇
        cluster_stats = {}
        
        for i, (path, label, score) in enumerate(zip(valid_paths, labels, outlier_scores)):
            if label not in cluster_stats:
                cluster_stats[label] = {
                    "count": 0,
                    "images": [],
                    "avg_outlier_score": 0,
                    "max_outlier_score": 0,
                }
            
            cluster_stats[label]["count"] += 1
            cluster_stats[label]["images"].append({
                "path": path,
                "outlier_score": float(score),
            })
        
        # 计算簇统计
        for label, stats in cluster_stats.items():
            scores = [img["outlier_score"] for img in stats["images"]]
            stats["avg_outlier_score"] = float(np.mean(scores))
            stats["max_outlier_score"] = float(np.max(scores))
            
            if label == -1:
                stats["type"] = "noise"
            else:
                stats["type"] = "cluster"
        
        return {
            "total_images": len(valid_paths),
            "num_clusters": len(set(labels) - {-1}),
            "num_noise": list(labels).count(-1),
            "cluster_stats": cluster_stats,
        }
    
    def filter_directory(
        self,
        directory: str,
        outlier_threshold: float = 0.7,
        move_to_subdir: str = "_outliers",
        dry_run: bool = False,
    ) -> Dict:
        """
        过滤目录中的异常图片
        """
        dir_path = Path(directory)
        
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            image_paths.extend([str(p) for p in dir_path.glob(ext)])
        
        if not image_paths:
            return {"message": "目录中没有图片"}
        
        # 分析聚类
        analysis = self.analyze_clusters(image_paths)
        
        if "message" in analysis:
            return analysis
        
        # 找出异常点
        outliers = []
        for label, stats in analysis["cluster_stats"].items():
            if label == -1:  # 噪声点
                outliers.extend(stats["images"])
            else:
                # 检查簇内异常分数高的
                for img in stats["images"]:
                    if img["outlier_score"] >= outlier_threshold:
                        outliers.append(img)
        
        outlier_paths = [o["path"] for o in outliers]
        
        result = {
            "total_images": len(image_paths),
            "kept_images": len(image_paths) - len(outlier_paths),
            "removed_images": len(outlier_paths),
            "outliers": outliers,
        }
        
        if not dry_run and outlier_paths:
            outlier_dir = dir_path / move_to_subdir
            outlier_dir.mkdir(exist_ok=True)
            
            for path in outlier_paths:
                try:
                    filename = Path(path).name
                    new_path = outlier_dir / filename
                    
                    # 处理重名
                    counter = 1
                    while new_path.exists():
                        stem = Path(path).stem
                        suffix = Path(path).suffix
                        new_path = outlier_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    os.rename(path, new_path)
                    logger.info(f"移动异常图片: {filename}")
                except Exception as e:
                    logger.error(f"移动失败 {path}: {e}")
        
        return result


class PerCharacterClusterFilter:
    """
    角色内聚类过滤器
    
    对每个角色单独进行聚类，检测该角色内的异常图片
    """
    
    def __init__(self, cluster_filter: HDBSCANClusterFilter = None):
        self.filter = cluster_filter or HDBSCANClusterFilter()
    
    def batch_filter(
        self,
        root_dir: str,
        outlier_threshold: float = 0.7,
        min_images: int = 10,
        dry_run: bool = False,
    ) -> Dict:
        """
        批量过滤角色目录
        """
        results = {}
        root_path = Path(root_dir)
        
        for char_dir in tqdm(list(root_path.iterdir()), desc="聚类过滤"):
            if not char_dir.is_dir():
                continue
            
            char_name = char_dir.name
            
            # 获取图片数
            image_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                image_paths.extend([str(p) for p in char_dir.glob(ext)])
            
            if len(image_paths) < min_images:
                results[char_name] = {
                    "message": f"图片数不足 ({len(image_paths)} < {min_images})",
                    "total_images": len(image_paths),
                }
                continue
            
            result = self.filter.filter_directory(
                str(char_dir),
                outlier_threshold=outlier_threshold,
                dry_run=dry_run,
            )
            results[char_name] = result
        
        # 汇总
        total_kept = sum(
            r.get("kept_images", 0) for r in results.values()
        )
        total_removed = sum(
            r.get("removed_images", 0) for r in results.values()
        )
        total = total_kept + total_removed
        
        return {
            "total_characters": len(results),
            "total_images_before": total,
            "kept_images": total_kept,
            "removed_images": total_removed,
            "keep_rate": total_kept / total if total > 0 else 0,
            "character_results": results,
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HDBSCAN聚类过滤")
    parser.add_argument("--dir", "-d", required=True, help="图片目录")
    parser.add_argument("--threshold", "-t", type=float, default=0.7, help="异常阈值")
    parser.add_argument("--min-cluster", "-m", type=int, default=5, help="最小簇大小")
    parser.add_argument("--dry-run", action="store_true", help="只报告不移动")
    
    args = parser.parse_args()
    
    cluster_filter = HDBSCANClusterFilter(min_cluster_size=args.min_cluster)
    result = cluster_filter.filter_directory(args.dir, outlier_threshold=args.threshold, dry_run=args.dry_run)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
