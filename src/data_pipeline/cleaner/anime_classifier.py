#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动漫分类器 - 区分动漫图片和非动漫图片
"""

import os
import sys
import platform
from pathlib import Path
from typing import Tuple, Dict

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Mac平台允许MPS加速

from PIL import Image
import logging

logger = logging.getLogger(__name__)


class AnimeClassifier:
    """
    基于CLIP的动漫分类器
    
    使用CLIP模型来判断图片是否为动漫风格
    通过对比动漫相关文本和非动漫文本的相似度来分类
    """
    
    def __init__(self, device: str = None):
        """
        初始化动漫分类器
        
        Args:
            device: 运行设备，可选 'cuda', 'mps', 'cpu'
        """
        self.device = self._select_device(device)
        self.model = None
        self.processor = None
        self._initialized = False
        
        # 动漫和非动漫的提示词
        self.anime_prompts = [
            "anime style",
            "anime character",
            "manga style",
            "Japanese animation",
            "anime art",
            "cartoon anime"
        ]
        
        self.non_anime_prompts = [
            "realistic photo",
            "real person",
            "photography",
            "real life",
            "live action",
            "3D rendering"
        ]
    
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
        
        logger.info(f"📥 正在加载CLIP模型用于动漫分类...")
        
        try:
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
            logger.info(f"✅ CLIP模型加载完成，运行设备: {self.device}")
        except Exception as e:
            logger.error(f"❌ CLIP模型加载失败: {e}")
            raise
    
    def classify(self, image_path: str) -> Tuple[float, str]:
        """
        分类单张图片
        
        Args:
            image_path: 图片路径
        
        Returns:
            (置信度, 分类结果)
            置信度: 0-1，表示为动漫的概率
            分类结果: 'anime' 或 'non-anime'
        """
        if not self._initialized:
            self.initialize()
        
        try:
            import torch
            import torch.nn.functional as F
            
            image = Image.open(image_path).convert("RGB")
            
            # 处理图片和文本
            inputs = self.processor(
                text=self.anime_prompts + self.non_anime_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 计算相似度
            logits_per_image = outputs.logits_per_image  # shape: (1, num_texts)
            probs = F.softmax(logits_per_image, dim=1)
            
            # 动漫提示词的平均概率
            anime_prob = probs[0, :len(self.anime_prompts)].mean().item()
            non_anime_prob = probs[0, len(self.anime_prompts):].mean().item()
            
            # 归一化
            total = anime_prob + non_anime_prob
            if total > 0:
                anime_prob = anime_prob / total
            
            result = "anime" if anime_prob >= 0.5 else "non-anime"
            return anime_prob, result
            
        except Exception as e:
            logger.error(f"❌ 分类图片失败 {image_path}: {e}")
            return 0.0, "non-anime"
    
    def batch_classify(self, image_paths: list) -> list:
        """
        批量分类图片
        
        Args:
            image_paths: 图片路径列表
        
        Returns:
            分类结果列表，每个元素为 (path, probability, result)
        """
        results = []
        for path in image_paths:
            prob, result = self.classify(path)
            results.append((path, prob, result))
        return results


class QualityFilter:
    """
    图片质量过滤器
    
    根据分辨率、模糊度等指标过滤低质量图片
    """
    
    def __init__(self):
        self.min_width = 256
        self.min_height = 256
        self.min_area = 256 * 256  # 最小像素面积
        self.min_ratio = 0.1  # 最小宽高比
        self.max_ratio = 10.0  # 最大宽高比
    
    def check_resolution(self, image_path: str) -> Tuple[bool, Dict]:
        try:
            with Image.open(image_path) as img:
                size = img.size
                # 调试：打印 size 的类型和值
                logger.debug(f"size = {size}, type = {type(size)}")
                
                # 防御：确保 size 是包含两个整数的元组
                if not isinstance(size, (tuple, list)) or len(size) != 2:
                    return False, {"reason": f"无效的图片尺寸结构: {size}"}
                
                width, height = size
                
                # 确保 width, height 是数值类型
                if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
                    return False, {"reason": f"尺寸类型错误: {type(width)}, {type(height)}"}
                
                if width <= 0 or height <= 0:
                    return False, {"reason": "图片尺寸无效"}
                
                # 后续检查...
        except Exception as e:
            logger.error(f"❌ 检查分辨率失败 {image_path}: {type(e).__name__} - {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, {"reason": str(e)}
            
    def check_format(self, image_path: str) -> Tuple[bool, str]:
        """
        检查图片格式
        
        Returns:
            (是否通过, 原因)
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.gif')
        ext = os.path.splitext(image_path)[1].lower()
        
        if ext not in valid_extensions:
            return False, f"不支持的格式: {ext}"
        
        return True, None
    
    def filter(self, image_path: str) -> Tuple[bool, Dict]:
        """
        综合过滤图片
        
        Returns:
            (是否通过, 检查结果)
        """
        # 检查格式
        format_ok, format_reason = self.check_format(image_path)
        if not format_ok:
            return False, {"check": "format", "reason": format_reason}
        
        # 检查分辨率
        res_ok, res_info = self.check_resolution(image_path)
        if not res_ok:
            return False, {"check": "resolution", **res_info}
        
        return True, {"check": "passed"}


if __name__ == "__main__":
    # 测试动漫分类器
    classifier = AnimeClassifier()
    classifier.initialize()
    
    # 测试质量过滤器
    quality_filter = QualityFilter()
    
    print("✅ 动漫分类器和质量过滤器初始化完成")
