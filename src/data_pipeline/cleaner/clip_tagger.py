#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP标签生成器 - 使用CLIP模型生成图片标签
"""

import os
import sys
import platform
from pathlib import Path
from typing import List, Dict, Tuple

project_root = Path(__file__).parent.parent.parent.parent

# Mac平台允许MPS加速

from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CLIPTagger:
    """
    基于CLIP的标签生成器
    
    使用CLIP模型为图片生成相关标签
    """
    
    def __init__(self, device: str = None):
        """
        初始化CLIP标签生成器
        
        Args:
            device: 运行设备，可选 'cuda', 'mps', 'cpu'
        """
        self.device = self._select_device(device)
        self.model = None
        self.processor = None
        self._initialized = False
        
        # 预定义的标签类别
        self.tag_categories = {
            "style": [
                "anime", "manga", "cartoon", "chibi", "kawaii",
                "realistic", "3D", "pixel art", "watercolor", "oil painting"
            ],
            "genre": [
                "fantasy", "scifi", "romance", "action", "comedy",
                "horror", "mecha", "school life", "isekai", "slice of life"
            ],
            "character": [
                "girl", "boy", "woman", "man", "child",
                "loli", "shota", "neko", "robot", "monster"
            ],
            "hair": [
                "blonde", "black hair", "brown hair", "red hair", "blue hair",
                "pink hair", "purple hair", "green hair", "white hair", "silver hair"
            ],
            "eyes": [
                "blue eyes", "brown eyes", "green eyes", "red eyes", "purple eyes",
                "yellow eyes", "pink eyes", "golden eyes", "heterochromia", "closed eyes"
            ],
            "expression": [
                "smiling", "happy", "sad", "angry", "surprised",
                "blushing", "sleepy", "serious", "cute", "cool"
            ],
            "pose": [
                "standing", "sitting", "lying", "running", "jumping",
                "dancing", "waving", "pointing", "hugging", "fighting"
            ],
            "clothing": [
                "school uniform", "maid outfit", "swimsuit", "kimono", "suit",
                "casual", "sports wear", "armor", "cosplay", "nurse outfit"
            ],
            "background": [
                "school", "city", "nature", "beach", "forest",
                "night", "sunset", "snow", "space", "cyberpunk"
            ],
            "accessories": [
                "glasses", "hat", "ribbon", "bow", "earrings",
                "necklace", "bracelet", "headphones", "wings", "tail"
            ]
        }
    
    def _select_device(self, device: str = None) -> str:
        """选择计算设备"""
        if device is not None:
            return device
        
        # 检查平台
        system = platform.system()
        
        # Mac平台尝试使用MPS加速
        if system == "Darwin":
            try:
                import torch
                if torch.backends.mps.is_available():
                    return "mps"
            except:
                pass
            return "cpu"
        
        # 检查是否已禁用CUDA
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
            return "cpu"
        
        # 其他平台尝试使用CUDA或MPS
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except:
            return "cpu"
    
    def initialize(self):
        """初始化CLIP模型"""
        if self._initialized:
            return
        
        logger.info(f"📥 正在加载CLIP标签模型...")
        
        try:
            import torch
            import torch.nn.functional as F
            from transformers import CLIPProcessor, CLIPModel
            
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # 尝试移动到MPS设备，如果失败则使用CPU
            if self.device == "mps":
                try:
                    self.model = self.model.to(self.device)
                except Exception as e:
                    logger.warning(f"⚠️ MPS设备不支持，自动切换到CPU: {e}")
                    self.device = "cpu"
                    self.model = self.model.to("cpu")
            else:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._initialized = True
            logger.info(f"✅ CLIP标签模型加载完成，运行设备: {self.device}")
        except Exception as e:
            logger.error(f"❌ CLIP模型加载失败: {e}")
            raise
    
    def generate_tags(self, image_path: str, top_k: int = 5, 
                     threshold: float = 0.2) -> List[Dict]:
        """
        为图片生成标签
        
        Args:
            image_path: 图片路径
            top_k: 每个类别返回的标签数量
            threshold: 置信度阈值
        
        Returns:
            标签列表，每个标签包含 category, tag, confidence
        """
        if not self._initialized:
            self.initialize()
        
        import torch
        import torch.nn.functional as F
        
        tags = []
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            for category, tag_list in self.tag_categories.items():
                # 处理当前类别的标签
                inputs = self.processor(
                    text=tag_list,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                logits_per_image = outputs.logits_per_image
                probs = F.softmax(logits_per_image, dim=1)
                
                # 获取排序后的标签和置信度
                sorted_indices = torch.argsort(probs, descending=True)
                sorted_probs = probs[0, sorted_indices[0]]
                
                for i in range(min(top_k, len(tag_list))):
                    idx = sorted_indices[0][i].item()
                    confidence = sorted_probs[i].item()
                    
                    if confidence >= threshold:
                        tags.append({
                            "category": category,
                            "tag": tag_list[idx],
                            "confidence": confidence
                        })
            
            # 按置信度排序
            tags.sort(key=lambda x: x["confidence"], reverse=True)
            return tags
            
        except Exception as e:
            logger.error(f"❌ 生成标签失败 {image_path}: {e}")
            return []
    
    def batch_generate_tags(self, image_paths: list, top_k: int = 5, 
                           threshold: float = 0.2) -> Dict[str, List[Dict]]:
        """
        批量为图片生成标签
        
        Args:
            image_paths: 图片路径列表
            top_k: 每个类别返回的标签数量
            threshold: 置信度阈值
        
        Returns:
            字典，key为图片路径，value为标签列表
        """
        results = {}
        for path in image_paths:
            tags = self.generate_tags(path, top_k, threshold)
            results[path] = tags
        return results


class MultiTagger:
    """
    多维度标签生成器
    
    整合多种标签生成方法
    """
    
    def __init__(self):
        self.clip_tagger = CLIPTagger()
        self._initialized = False
    
    def initialize(self):
        """初始化所有标签器"""
        if self._initialized:
            return
        
        self.clip_tagger.initialize()
        self._initialized = True
    
    def generate_comprehensive_tags(self, image_path: str) -> Dict:
        """
        生成综合标签
        
        Returns:
            包含各类标签的字典
        """
        if not self._initialized:
            self.initialize()
        
        clip_tags = self.clip_tagger.generate_tags(image_path)
        
        # 按类别分组
        tags_by_category = {}
        for tag in clip_tags:
            category = tag["category"]
            if category not in tags_by_category:
                tags_by_category[category] = []
            tags_by_category[category].append({
                "tag": tag["tag"],
                "confidence": tag["confidence"]
            })
        
        return {
            "all_tags": clip_tags,
            "by_category": tags_by_category,
            "top_tags": [t["tag"] for t in clip_tags[:10]]
        }


if __name__ == "__main__":
    # 测试标签生成器
    tagger = MultiTagger()
    tagger.initialize()
    
    print("✅ 多维度标签生成器初始化完成")
