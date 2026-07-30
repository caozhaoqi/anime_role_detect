#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP特征去重器
使用CLIP特征计算图片相似度，去除重复或高度相似的图片
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("clip_deduplicator")


class CLIPDeduplicator:
    """
    基于CLIP特征的图片去重
    
    功能：
    - 提取图片的CLIP特征
    - 计算特征相似度
    - 识别并去除重复/高度相似图片
    - 支持多角色批量去重
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        embedder=None,
        device: Optional[str] = None,
    ):
        """
        初始化去重器
        
        Args:
            similarity_threshold: 相似度阈值，超过此值认为重复（0-1）
            embedder: CLIP特征提取器，None则自动创建
            device: 运行设备
        """
        self.similarity_threshold = similarity_threshold
        
        if embedder is None:
            from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached
            self.embedder = CLIPEmbedderCached(
                model_name="ViT-B/32",
                device=device,
            )
        else:
            self.embedder = embedder
        
        logger.info(f"CLIP去重器初始化完成，阈值: {similarity_threshold}")
    
    def extract_features(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        批量提取特征
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            {路径: 特征向量} 字典
        """
        features = {}
        
        for path in tqdm(image_paths, desc="提取特征"):
            try:
                feature = self.embedder.embed_image(path)
                if feature is not None:
                    features[path] = feature
            except Exception as e:
                logger.warning(f"特征提取失败 {path}: {e}")
        
        return features
    
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        计算两个特征的余弦相似度
        
        Args:
            feat1: 特征向量1
            feat2: 特征向量2
            
        Returns:
            相似度分数 (0-1)
        """
        return float(np.dot(feat1, feat2))
    
    def find_duplicates(
        self,
        features: Dict[str, np.ndarray],
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        找出重复图片对
        
        Args:
            features: 特征字典
            threshold: 相似度阈值，None则使用默认值
            
        Returns:
            [(图片1路径, 图片2路径, 相似度)] 列表
        """
        threshold = threshold or self.similarity_threshold
        paths = list(features.keys())
        n = len(paths)
        
        duplicates = []
        
        logger.info(f"开始检测重复图片，图片数: {n}")
        
        for i in tqdm(range(n), desc="计算相似度"):
            for j in range(i + 1, n):
                sim = self.compute_similarity(features[paths[i]], features[paths[j]])
                
                if sim >= threshold:
                    duplicates.append((paths[i], paths[j], sim))
        
        duplicates.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"发现 {len(duplicates)} 对重复图片")
        return duplicates
    
    def select_unique_images(
        self,
        features: Dict[str, np.ndarray],
        duplicates: List[Tuple[str, str, float]],
    ) -> List[str]:
        """
        从重复图片中选择保留的图片
        
        Args:
            features: 特征字典
            duplicates: 重复图片对列表
            
        Returns:
            保留的图片路径列表
        """
        duplicate_count = {path: 0 for path in features.keys()}
        
        for path1, path2, _ in duplicates:
            duplicate_count[path1] += 1
            duplicate_count[path2] += 1
        
        parent = {path: path for path in features.keys()}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for path1, path2, _ in duplicates:
            union(path1, path2)
        
        components = {}
        for path in features.keys():
            root = find(path)
            if root not in components:
                components[root] = []
            components[root].append(path)
        
        selected = []
        
        for paths in components.values():
            if len(paths) == 1:
                selected.append(paths[0])
            else:
                best = max(paths, key=lambda p: self._get_image_quality(p))
                selected.append(best)
                logger.info(f"去重组: 保留 {Path(best).name}, 删除 {len(paths)-1} 张")
        
        return selected
    
    def _get_image_quality(self, path: str) -> Tuple[int, int]:
        """获取图片质量分数"""
        try:
            with Image.open(path) as img:
                w, h = img.size
                return w * h
        except:
            return 0, 0
    
    def deduplicate_directory(
        self,
        directory: str,
        recursive: bool = True,
        dry_run: bool = False,
    ) -> Dict:
        """
        对目录下的图片去重
        """
        image_paths = []
        dir_path = Path(directory)
        
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        for pattern in patterns:
            if recursive:
                image_paths.extend([str(p) for p in dir_path.rglob(pattern)])
            else:
                image_paths.extend([str(p) for p in dir_path.glob(pattern)])
        
        logger.info(f"找到 {len(image_paths)} 张图片")
        
        if len(image_paths) < 2:
            return {"message": "图片数量不足2张，无需去重"}
        
        features = self.extract_features(image_paths)
        
        if len(features) < 2:
            return {"message": "有效图片数量不足2张"}
        
        duplicates = self.find_duplicates(features)
        selected = self.select_unique_images(features, duplicates)
        removed = set(image_paths) - set(selected)
        
        result = {
            "total_images": len(image_paths),
            "valid_images": len(features),
            "duplicate_pairs": len(duplicates),
            "removed_images": len(removed),
            "kept_images": len(selected),
            "removed_paths": list(removed) if not dry_run else [],
        }
        
        if not dry_run and removed:
            for path in removed:
                try:
                    os.remove(path)
                    logger.info(f"删除重复图片: {path}")
                except Exception as e:
                    logger.error(f"删除失败 {path}: {e}")
        
        return result
    
    def deduplicate_character_dirs(
        self,
        root_dir: str,
        dry_run: bool = False,
    ) -> Dict[str, Dict]:
        """
        对角色目录批量去重
        """
        results = {}
        root_path = Path(root_dir)
        
        for char_dir in tqdm(list(root_path.iterdir()), desc="处理角色"):
            if not char_dir.is_dir():
                continue
            
            char_name = char_dir.name
            result = self.deduplicate_directory(str(char_dir), recursive=False, dry_run=dry_run)
            results[char_name] = result
        
        total_removed = sum(r.get("removed_images", 0) for r in results.values())
        total_images = sum(r.get("total_images", 0) for r in results.values())
        
        return {
            "total_characters": len(results),
            "total_images_before": total_images,
            "total_images_removed": total_removed,
            "total_images_after": total_images - total_removed,
            "character_results": results,
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP特征图片去重")
    parser.add_argument("--dir", "-d", required=True, help="图片目录")
    parser.add_argument("--threshold", "-t", type=float, default=0.95, help="相似度阈值")
    parser.add_argument("--dry-run", action="store_true", help="只报告不删除")
    parser.add_argument("--batch", action="store_true", help="批量处理角色目录")
    
    args = parser.parse_args()
    
    deduplicator = CLIPDeduplicator(similarity_threshold=args.threshold)
    
    if args.batch:
        result = deduplicator.deduplicate_character_dirs(args.dir, dry_run=args.dry_run)
    else:
        result = deduplicator.deduplicate_directory(args.dir, dry_run=args.dry_run)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
