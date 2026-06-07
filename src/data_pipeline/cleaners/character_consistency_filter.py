#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色一致性过滤器
检测图片是否真正属于标签标注的角色
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("character_consistency")


class CharacterConsistencyFilter:
    """
    角色一致性过滤器
    
    使用CLIP计算图片与角色名称文本的匹配度，
    过滤掉标注错误或不包含目标角色的图片
    """
    
    def __init__(
        self,
        embedder=None,
        consistency_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        """
        初始化过滤器
        
        Args:
            embedder: CLIP特征提取器
            consistency_threshold: 一致性阈值，低于此值认为不属于该角色
            device: 运行设备
        """
        self.consistency_threshold = consistency_threshold
        
        if embedder is None:
            from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached
            self.embedder = CLIPEmbedderCached(
                model_name="ViT-B/32",
                device=device,
            )
        else:
            self.embedder = embedder
        
        logger.info(f"角色一致性过滤器初始化，阈值: {consistency_threshold}")
    
    def compute_text_similarity(
        self,
        image_path: str,
        character_name: str,
        work_title: Optional[str] = None,
    ) -> float:
        """
        计算图片与角色文本的相似度
        
        Args:
            image_path: 图片路径
            character_name: 角色名称
            work_title: 作品名称（可选）
            
        Returns:
            相似度分数 (0-1)
        """
        # 构建文本提示
        if work_title:
            prompts = [
                f"anime character {character_name}",
                f"{character_name} from {work_title}",
                f"anime girl named {character_name}",
                f"character portrait of {character_name}",
            ]
        else:
            prompts = [
                f"anime character {character_name}",
                f"{character_name}",
                f"anime girl named {character_name}",
            ]
        
        try:
            # 提取图片特征
            image_feat = self.embedder.embed_image(image_path)
            if image_feat is None:
                return 0.0
            
            # 提取文本特征
            text_feats = self.embedder.embed_texts(prompts)
            if text_feats is None:
                return 0.0
            
            # 平均文本特征
            text_feat = np.mean(text_feats, axis=0)
            text_feat = text_feat / (np.linalg.norm(text_feat) + 1e-8)
            
            # 计算相似度
            similarity = float(np.dot(image_feat, text_feat))
            return max(0.0, similarity)
            
        except Exception as e:
            logger.warning(f"相似度计算失败 {image_path}: {e}")
            return 0.0
    
    def filter_character_images(
        self,
        image_paths: List[str],
        character_name: str,
        work_title: Optional[str] = None,
        return_scores: bool = False,
    ) -> List[str]:
        """
        过滤角色图片
        
        Args:
            image_paths: 图片路径列表
            character_name: 角色名称
            work_title: 作品名称
            return_scores: 是否返回分数
            
        Returns:
            过滤后的图片列表，或 [(路径, 分数)] 列表
        """
        results = []
        kept = []
        
        for path in tqdm(image_paths, desc=f"过滤 {character_name}"):
            score = self.compute_text_similarity(path, character_name, work_title)
            results.append((path, score))
            
            if score >= self.consistency_threshold:
                kept.append(path)
            else:
                logger.debug(f"过滤低一致性图片: {Path(path).name} ({score:.3f})")
        
        removed = len(image_paths) - len(kept)
        logger.info(f"角色 '{character_name}': 保留 {len(kept)}/{len(image_paths)}, 过滤 {removed}")
        
        if return_scores:
            return results
        return kept
    
    def filter_character_directory(
        self,
        directory: str,
        character_name: str,
        work_title: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict:
        """
        过滤角色目录
        
        Args:
            directory: 目录路径
            character_name: 角色名称
            work_title: 作品名称
            dry_run: 是否只报告不删除
            
        Returns:
            过滤统计结果
        """
        dir_path = Path(directory)
        
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            image_paths.extend([str(p) for p in dir_path.glob(ext)])
        
        if not image_paths:
            return {"message": "目录中没有图片"}
        
        # 计算所有图片的一致性分数
        results = self.filter_character_images(
            image_paths, character_name, work_title, return_scores=True
        )
        
        # 分类
        kept = [(p, s) for p, s in results if s >= self.consistency_threshold]
        removed = [(p, s) for p, s in results if s < self.consistency_threshold]
        
        # 按分数排序（可疑的排在前面）
        removed.sort(key=lambda x: x[1])
        
        result = {
            "character": character_name,
            "total_images": len(image_paths),
            "kept_images": len(kept),
            "removed_images": len(removed),
            "consistency_scores": [
                {"path": p, "score": s} for p, s in removed
            ] if return_scores else [],
        }
        
        if not dry_run and removed:
            removed_paths = [p for p, _ in removed]
            for path in removed_paths:
                try:
                    # 移动到 _filtered 目录
                    filtered_dir = dir_path / "_filtered"
                    filtered_dir.mkdir(exist_ok=True)
                    
                    filename = Path(path).name
                    new_path = filtered_dir / filename
                    
                    # 处理重名
                    counter = 1
                    while new_path.exists():
                        stem = Path(path).stem
                        suffix = Path(path).suffix
                        new_path = filtered_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    os.rename(path, new_path)
                    logger.info(f"移至过滤目录: {filename}")
                except Exception as e:
                    logger.error(f"移动文件失败 {path}: {e}")
        
        return result
    
    def batch_filter(
        self,
        root_dir: str,
        character_mapping: Dict[str, Tuple[str, Optional[str]]] = None,
        dry_run: bool = False,
    ) -> Dict:
        """
        批量过滤角色目录
        
        Args:
            root_dir: 根目录（包含各角色子目录）
            character_mapping: 角色名到(角色名称, 作品名)的映射
            dry_run: 是否只报告不删除
            
        Returns:
            各角色的过滤统计
        """
        results = {}
        root_path = Path(root_dir)
        
        for char_dir in tqdm(list(root_path.iterdir()), desc="批量过滤"):
            if not char_dir.is_dir():
                continue
            
            char_name = char_dir.name
            
            # 获取角色信息
            if character_mapping and char_name in character_mapping:
                eng_name, work_title = character_mapping[char_name]
            else:
                eng_name = char_name
                work_title = None
            
            result = self.filter_character_directory(
                str(char_dir), eng_name, work_title, dry_run=dry_run
            )
            results[char_name] = result
        
        # 汇总
        total_kept = sum(r.get("kept_images", 0) for r in results.values())
        total_removed = sum(r.get("removed_images", 0) for r in results.values())
        total = total_kept + total_removed
        
        summary = {
            "total_characters": len(results),
            "total_images_before": total,
            "kept_images": total_kept,
            "removed_images": total_removed,
            "keep_rate": total_kept / total if total > 0 else 0,
            "character_results": results,
        }
        
        logger.info(f"批量过滤完成: 保留 {total_kept}/{total} ({summary['keep_rate']:.1%})")
        return summary


class CharacterContrastiveFilter:
    """
    角色对比过滤器
    
    使用角色间的特征差异来判断图片是否属于目标角色
    原理：如果一张图片与其他角色的相似度高于目标角色，则可能是标注错误
    """
    
    def __init__(
        self,
        embedder=None,
        device: Optional[str] = None,
    ):
        self.embedder = embedder or self._create_embedder(device)
        self.target_features: Dict[str, List[np.ndarray]] = {}
        
    def _create_embedder(self, device):
        from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached
        return CLIPEmbedderCached(model_name="ViT-B/32", device=device)
    
    def build_character_features(
        self,
        character_dirs: Dict[str, str],
    ):
        """
        构建各角色的特征库
        
        Args:
            character_dirs: {角色名: 目录路径}
        """
        logger.info("构建角色特征库...")
        
        for char_name, dir_path in tqdm(character_dirs.items(), desc="提取角色特征"):
            features = []
            
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                for img_path in Path(dir_path).glob(ext):
                    feat = self.embedder.embed_image(str(img_path))
                    if feat is not None:
                        features.append(feat)
            
            if features:
                self.target_features[char_name] = features
                logger.info(f"  {char_name}: {len(features)} 个特征")
    
    def check_consistency(
        self,
        image_path: str,
        target_character: str,
    ) -> Dict[str, float]:
        """
        检查图片与各角色的相似度
        
        Returns:
            {角色名: 相似度}
        """
        image_feat = self.embedder.embed_image(image_path)
        if image_feat is None:
            return {}
        
        similarities = {}
        for char_name, features in self.target_features.items():
            if not features:
                continue
            
            # 计算与该角色所有样本的平均相似度
            sims = [float(np.dot(image_feat, f)) for f in features]
            similarities[char_name] = max(sims) if sims else 0.0
        
        return similarities
    
    def detect_mislabeled(
        self,
        image_path: str,
        target_character: str,
        confusion_threshold: float = 0.1,
    ) -> Tuple[bool, str]:
        """
        检测图片是否可能标注错误
        
        Args:
            image_path: 图片路径
            target_character: 目标角色名
            confusion_threshold: 混淆阈值
            
        Returns:
            (是否可疑, 原因)
        """
        similarities = self.check_consistency(image_path, target_character)
        
        if not similarities:
            return True, "无法计算相似度"
        
        target_sim = similarities.get(target_character, 0.0)
        
        # 找出与目标角色最相似的其他角色
        other_chars = {k: v for k, v in similarities.items() if k != target_character}
        
        if not other_chars:
            return False, ""
        
        best_other = max(other_chars.items(), key=lambda x: x[1])
        
        # 如果另一个角色的相似度更高，可能是错误标注
        if best_other[1] > target_sim + confusion_threshold:
            return True, f"更接近 {best_other[0]} ({best_other[1]:.3f} vs {target_sim:.3f})"
        
        # 如果与其他所有角色都很相似，可能是噪声/非角色图
        if all(sim > 0.7 for sim in other_chars.values()):
            return True, f"与多个角色都相似（可能非角色图）"
        
        return False, ""


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="角色一致性过滤")
    parser.add_argument("--dir", "-d", required=True, help="图片目录")
    parser.add_argument("--character", "-c", required=True, help="角色名称")
    parser.add_argument("--work", "-w", help="作品名称")
    parser.add_argument("--threshold", "-t", type=float, default=0.25, help="一致性阈值")
    parser.add_argument("--dry-run", action="store_true", help="只报告不移动")
    
    args = parser.parse_args()
    
    filter = CharacterConsistencyFilter(consistency_threshold=args.threshold)
    result = filter.filter_character_directory(
        args.dir, args.character, args.work, dry_run=args.dry_run
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
