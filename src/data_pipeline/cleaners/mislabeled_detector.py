#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误标签检测器
使用多种策略检测可能被错误标注的图片
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger("mislabeled_detector")


class MislabeledDetector:
    """
    错误标签检测器
    
    检测策略：
    1. CLIP文本-图片匹配度
    2. 角色间特征混淆度
    3. 角色内异常检测
    4. 视觉质量异常
    """
    
    def __init__(
        self,
        embedder=None,
        device: Optional[str] = None,
    ):
        self.embedder = embedder or self._create_embedder(device)
        self.character_features: Dict[str, List[np.ndarray]] = {}
        self.character_names: Dict[str, str] = {}  # 目录名 -> 标准名
    
    def _create_embedder(self, device):
        from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached
        return CLIPEmbedderCached(model_name="ViT-B/32", device=device)
    
    def build_feature_library(
        self,
        root_dir: str,
        name_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        构建角色特征库
        
        Args:
            root_dir: 包含角色子目录的根目录
            name_mapping: 目录名到标准角色名的映射
        """
        logger.info("构建角色特征库...")
        
        root_path = Path(root_dir)
        self.character_features = {}
        self.character_names = {}
        
        for char_dir in tqdm(list(root_path.iterdir()), desc="提取特征"):
            if not char_dir.is_dir():
                continue
            
            char_key = char_dir.name
            char_name = (name_mapping or {}).get(char_key, char_key)
            self.character_names[char_key] = char_name
            
            features = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                for img_path in char_dir.glob(ext):
                    feat = self.embedder.embed_image(str(img_path))
                    if feat is not None:
                        features.append(feat)
            
            if features:
                self.character_features[char_key] = features
                logger.info(f"  {char_name}: {len(features)} 个特征")
    
    def compute_text_similarity(
        self,
        image_path: str,
        character_name: str,
    ) -> float:
        """计算图片与角色文本的相似度"""
        prompts = [
            f"anime character {character_name}",
            f"{character_name}",
            f"anime girl named {character_name}",
        ]
        
        try:
            image_feat = self.embedder.embed_image(image_path)
            if image_feat is None:
                return 0.0
            
            text_feats = self.embedder.embed_texts(prompts)
            if text_feats is None:
                return 0.0
            
            text_feat = np.mean(text_feats, axis=0)
            text_feat = text_feat / (np.linalg.norm(text_feat) + 1e-8)
            
            return float(np.dot(image_feat, text_feat))
        except:
            return 0.0
    
    def compute_cross_character_similarity(
        self,
        image_path: str,
    ) -> Dict[str, float]:
        """计算图片与所有角色的相似度"""
        image_feat = self.embedder.embed_image(image_path)
        if image_feat is None:
            return {}
        
        similarities = {}
        for char_key, features in self.character_features.items():
            if not features:
                continue
            
            # 计算与该角色所有样本的最大相似度
            max_sim = 0.0
            for feat in features:
                sim = float(np.dot(image_feat, feat))
                if sim > max_sim:
                    max_sim = sim
            similarities[self.character_names.get(char_key, char_key)] = max_sim
        
        return similarities
    
    def detect_text_mismatch(
        self,
        image_path: str,
        character_name: str,
        threshold: float = 0.2,
    ) -> Tuple[bool, float]:
        """
        检测文本-图片不匹配
        
        Returns:
            (是否可疑, 相似度分数)
        """
        score = self.compute_text_similarity(image_path, character_name)
        is_suspicious = score < threshold
        return is_suspicious, score
    
    def detect_cross_character_confusion(
        self,
        image_path: str,
        target_character: str,
        confusion_gap: float = 0.08,
    ) -> Tuple[bool, str]:
        """
        检测跨角色混淆
        
        如果图片与其他角色的相似度高于目标角色，则可疑
        
        Returns:
            (是否可疑, 原因描述)
        """
        similarities = self.compute_cross_character_similarity(image_path)
        
        if not similarities:
            return True, "无法计算相似度"
        
        target_sim = similarities.get(target_character, 0.0)
        
        # 找出与目标角色最相似的其他角色
        other_sims = {k: v for k, v in similarities.items() if k != target_character}
        
        if not other_sims:
            return False, ""
        
        best_other = max(other_sims.items(), key=lambda x: x[1])
        
        # 如果另一个角色更匹配
        if best_other[1] > target_sim + confusion_gap:
            return True, f"更匹配 {best_other[0]} ({best_other[1]:.3f} > {target_sim:.3f})"
        
        return False, ""
    
    def detect_intra_character_outlier(
        self,
        image_path: str,
        character_key: str,
        outlier_threshold: float = 0.7,
    ) -> Tuple[bool, float]:
        """
        检测角色内异常点
        
        计算图片与角色特征中心的距离
        
        Returns:
            (是否异常, 异常分数)
        """
        features = self.character_features.get(character_key, [])
        if not features:
            return True, 1.0
        
        image_feat = self.embedder.embed_image(image_path)
        if image_feat is None:
            return True, 1.0
        
        # 计算到所有样本的平均距离
        distances = []
        for feat in features:
            dist = np.linalg.norm(image_feat - feat)
            distances.append(dist)
        
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # 计算异常分数（距离超过均值+2倍标准差为异常）
        if std_dist > 0:
            outlier_score = min(1.0, (avg_dist - np.mean(distances)) / (2 * std_dist) + 0.5)
        else:
            outlier_score = 0.5
        
        return outlier_score > outlier_threshold, float(outlier_score)
    
    def detect_image_quality_issues(
        self,
        image_path: str,
        min_resolution: Tuple[int, int] = (128, 128),
    ) -> Tuple[bool, str]:
        """
        检测图片质量问题
        
        Returns:
            (是否有问题, 问题描述)
        """
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                
                # 检查分辨率
                if w < min_resolution[0] or h < min_resolution[1]:
                    return True, f"分辨率过低 ({w}x{h})"
                
                # 检查是否为纯色/低信息量
                if img.mode == 'RGB':
                    arr = np.array(img)
                    std_r = np.std(arr[:, :, 0])
                    std_g = np.std(arr[:, :, 1])
                    std_b = np.std(arr[:, :, 2])
                    
                    if std_r < 5 and std_g < 5 and std_b < 5:
                        return True, "图片信息量过低（可能是纯色图）"
                
                return False, ""
                
        except Exception as e:
            return True, f"无法读取图片: {e}"
    
    def detect_all(
        self,
        image_path: str,
        character_key: str,
        thresholds: Optional[Dict] = None,
    ) -> Dict:
        """
        综合检测所有问题
        
        Args:
            image_path: 图片路径
            character_key: 角色目录名
            thresholds: 各检测项阈值
            
        Returns:
            检测结果字典
        """
        thresholds = thresholds or {
            "text_similarity": 0.2,
            "confusion_gap": 0.08,
            "outlier_score": 0.7,
        }
        
        character_name = self.character_names.get(character_key, character_key)
        
        result = {
            "path": image_path,
            "character": character_name,
            "suspicious": False,
            "issues": [],
            "scores": {},
        }
        
        # 1. 文本匹配检测
        is_text_suspicious, text_score = self.detect_text_mismatch(
            image_path, character_name, thresholds["text_similarity"]
        )
        result["scores"]["text_similarity"] = text_score
        
        if is_text_suspicious:
            result["suspicious"] = True
            result["issues"].append(f"文本匹配度低 ({text_score:.3f})")
        
        # 2. 跨角色混淆检测
        if self.character_features:
            is_confused, confusion_reason = self.detect_cross_character_confusion(
                image_path, character_name, thresholds["confusion_gap"]
            )
            
            if is_confused:
                result["suspicious"] = True
                result["issues"].append(f"角色混淆: {confusion_reason}")
        
        # 3. 角色内异常检测
        if character_key in self.character_features:
            is_outlier, outlier_score = self.detect_intra_character_outlier(
                image_path, character_key, thresholds["outlier_score"]
            )
            result["scores"]["outlier_score"] = outlier_score
            
            if is_outlier:
                result["suspicious"] = True
                result["issues"].append(f"角色内异常 ({outlier_score:.3f})")
        
        # 4. 图片质量问题
        has_quality_issue, quality_reason = self.detect_image_quality_issues(image_path)
        if has_quality_issue:
            result["suspicious"] = True
            result["issues"].append(f"质量问题: {quality_reason}")
        
        return result
    
    def scan_directory(
        self,
        directory: str,
        character_key: Optional[str] = None,
        thresholds: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        扫描目录中的可疑图片
        
        Args:
            directory: 目录路径
            character_key: 角色目录名，None则使用目录名
            thresholds: 检测阈值
            
        Returns:
            可疑图片列表
        """
        char_key = character_key or Path(directory).name
        suspicious_images = []
        
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            image_paths.extend([str(p) for p in Path(directory).glob(ext)])
        
        for path in tqdm(image_paths, desc=f"检测 {char_key}"):
            result = self.detect_all(path, char_key, thresholds)
            if result["suspicious"]:
                suspicious_images.append(result)
        
        # 按可疑度排序
        suspicious_images.sort(
            key=lambda x: (
                -len(x["issues"]),
                -sum(1 for i in x["issues"] if "混淆" in i or "文本" in i)
            )
        )
        
        return suspicious_images
    
    def batch_scan(
        self,
        root_dir: str,
        name_mapping: Optional[Dict[str, str]] = None,
        thresholds: Optional[Dict] = None,
    ) -> Dict:
        """
        批量扫描角色目录
        """
        # 先构建特征库
        if not self.character_features:
            self.build_feature_library(root_dir, name_mapping)
        
        results = {}
        total_suspicious = 0
        
        root_path = Path(root_dir)
        for char_dir in tqdm(list(root_path.iterdir()), desc="批量检测"):
            if not char_dir.is_dir():
                continue
            
            char_key = char_dir.name
            suspicious = self.scan_directory(str(char_dir), char_key, thresholds)
            
            results[char_key] = {
                "total_images": len(list(char_dir.glob("*.jpg"))) + 
                               len(list(char_dir.glob("*.jpeg"))) +
                               len(list(char_dir.glob("*.png"))) +
                               len(list(char_dir.glob("*.webp"))),
                "suspicious_count": len(suspicious),
                "suspicious_images": suspicious,
            }
            
            total_suspicious += len(suspicious)
        
        return {
            "total_suspicious": total_suspicious,
            "character_results": results,
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="错误标签检测")
    parser.add_argument("--dir", "-d", required=True, help="角色目录")
    parser.add_argument("--text-threshold", type=float, default=0.2, help="文本匹配阈值")
    parser.add_argument("--output", "-o", help="输出JSON文件")
    
    args = parser.parse_args()
    
    detector = MislabeledDetector()
    detector.build_feature_library(args.dir)
    results = detector.batch_scan(args.dir, thresholds={"text_similarity": args.text_threshold})
    
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
