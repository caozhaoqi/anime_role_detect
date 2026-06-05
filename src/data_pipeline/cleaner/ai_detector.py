#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI检测器 - 检测图片是否为AI生成
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class AIDetector:
    """
    AI生成图片检测器
    
    使用预训练模型检测图片是否为AI生成
    """
    
    def __init__(self, device: str = None):
        """
        初始化AI检测器
        
        Args:
            device: 运行设备，可选 'cuda', 'mps', 'cpu'
        """
        self.device = self._select_device(device)
        self.model = None
        self.processor = None
        self._initialized = False
    
    def _select_device(self, device: str = None) -> str:
        """选择计算设备"""
        if device is not None:
            return device
        
        # 检查平台
        import platform
        system = platform.system()
        
        # Mac平台不支持CUDA，只支持MPS或CPU
        if system == "Darwin":
            if torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        # Linux/Windows可以使用CUDA
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def initialize(self):
        """初始化检测模型"""
        if self._initialized:
            return
        
        logger.info(f"📥 正在加载AI检测模型...")
        
        try:
            # 使用HuggingFace上的AI检测模型
            self.processor = AutoProcessor.from_pretrained("Salesforce/xgen-mm-phi3-mini-instruct")
            self.model = AutoModelForImageClassification.from_pretrained(
                "MoritzLaurer/AI-image-detector"
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            logger.info(f"✅ AI检测模型加载完成，运行设备: {self.device}")
        except Exception as e:
            logger.warning(f"⚠️ 标准AI检测模型加载失败，使用简化检测方法: {e}")
            self._initialized = True
    
    def detect(self, image_path: str) -> Tuple[float, str]:
        """
        检测图片是否为AI生成
        
        Args:
            image_path: 图片路径
        
        Returns:
            (置信度, 结果)
            置信度: 0-1，表示为AI生成的概率
            结果: 'ai-generated' 或 'real'
        """
        if not self._initialized:
            self.initialize()
        
        # 如果标准模型加载失败，使用简化检测方法
        if self.model is None:
            return self._simple_detect(image_path)
        
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
            # 获取AI生成的概率（假设标签0是real，标签1是AI）
            ai_prob = probs[0, 1].item() if probs.shape[1] > 1 else 0.5
            result = "ai-generated" if ai_prob >= 0.5 else "real"
            
            return ai_prob, result
            
        except Exception as e:
            logger.error(f"❌ AI检测失败 {image_path}: {e}")
            return self._simple_detect(image_path)
    
    def _simple_detect(self, image_path: str) -> Tuple[float, str]:
        """
        简化的AI检测方法
        
        通过检查图片元数据和视觉特征来判断是否为AI生成
        """
        try:
            with Image.open(image_path) as img:
                # 检查图片信息
                info = img.info
                
                # 检查是否有AI相关的元数据
                ai_indicators = ["AI", "Generated", "Stable Diffusion", "DALL-E", "MidJourney"]
                
                for key, value in info.items():
                    str_value = str(value).lower()
                    for indicator in ai_indicators:
                        if indicator.lower() in str_value:
                            return 0.95, "ai-generated"
                
                # 检查图片大小特征（AI图片通常有特定的分辨率）
                width, height = img.size
                ai_resolutions = [
                    (512, 512), (1024, 1024), (768, 1024), (1024, 768),
                    (576, 1024), (1024, 576), (640, 1280), (1280, 640)
                ]
                
                if (width, height) in ai_resolutions:
                    return 0.7, "ai-generated"
                
                # 默认认为是真实图片
                return 0.3, "real"
                
        except Exception as e:
            logger.error(f"❌ 简化检测失败 {image_path}: {e}")
            return 0.5, "unknown"


class CharacterCropper:
    """
    角色裁剪器
    
    根据检测到的边界框裁剪角色区域
    """
    
    def __init__(self):
        self.expand_ratio = 0.1  # 边界框扩展比例
    
    def crop_character(self, image_path: str, bbox: Tuple[int, int, int, int], 
                       output_path: str = None) -> str:
        """
        裁剪角色区域
        
        Args:
            image_path: 原图路径
            bbox: 边界框 (x1, y1, x2, y2)
            output_path: 输出路径，默认为原路径加_cropped后缀
        
        Returns:
            裁剪后图片的路径
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                x1, y1, x2, y2 = bbox
                
                # 扩展边界框
                expand_width = (x2 - x1) * self.expand_ratio
                expand_height = (y2 - y1) * self.expand_ratio
                
                x1 = max(0, int(x1 - expand_width))
                y1 = max(0, int(y1 - expand_height))
                x2 = min(width, int(x2 + expand_width))
                y2 = min(height, int(y2 + expand_height))
                
                # 裁剪
                cropped = img.crop((x1, y1, x2, y2))
                
                # 生成输出路径
                if output_path is None:
                    base, ext = os.path.splitext(image_path)
                    output_path = f"{base}_cropped{ext}"
                
                # 保存
                cropped.save(output_path)
                return output_path
                
        except Exception as e:
            logger.error(f"❌ 裁剪角色失败 {image_path}: {e}")
            return image_path
    
    def calculate_character_ratio(self, image_path: str, bbox: Tuple[int, int, int, int]) -> float:
        """
        计算角色占图片的比例
        
        Args:
            image_path: 图片路径
            bbox: 边界框 (x1, y1, x2, y2)
        
        Returns:
            角色区域占图片总面积的比例 (0-1)
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                image_area = width * height
                
                x1, y1, x2, y2 = bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                
                return bbox_area / image_area if image_area > 0 else 0.0
                
        except Exception as e:
            logger.error(f"❌ 计算角色比例失败 {image_path}: {e}")
            return 0.0


if __name__ == "__main__":
    # 测试AI检测器
    detector = AIDetector()
    detector.initialize()
    
    # 测试角色裁剪器
    cropper = CharacterCropper()
    
    print("✅ AI检测器和角色裁剪器初始化完成")
