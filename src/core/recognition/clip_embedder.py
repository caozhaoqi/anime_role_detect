#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP 特征提取器
CLIP Embedder

用于将图片转换为归一化的特征向量
"""

# 必须在导入PyTorch前设置环境变量
import os
import platform
from typing import List, Optional, Union
import numpy as np

if platform.system() == "Darwin":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["PYTORCH_MPS_ENABLED"] = "0"
    os.environ["FORCE_CPU"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """
    CLIP 特征提取器

    特点：
    - 支持懒加载
    - 支持批处理
    - 支持CPU/MPS/CUDA自动选择
    - 兼容OpenAI CLIP和HuggingFace CLIP
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        use_huggingface: bool = True,
    ):
        """
        初始化CLIP Embedder

        Args:
            model_name: 模型名称 (OpenAI: "ViT-B/32", "ViT-L/14"; HF: "openai/clip-vit-base-patch32")
            device: 运行设备, None为自动选择
            use_huggingface: 是否使用HuggingFace CLIP
        """
        self.model_name = model_name
        self.use_huggingface = use_huggingface
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._initialized = False
        self._model_type = None

        # 设备选择（不触发PyTorch加载）
        self.device = self._select_device(device)

        # 维度推断（基于模型名）
        if "ViT-L" in model_name or "large" in model_name.lower():
            self._dim = 768
        else:
            self._dim = 512

        logger.info(f"CLIPEmbedder 创建完成，模型: {model_name}, 设备: {self.device}")

    @property
    def embedding_dim(self) -> int:
        """获取特征维度（基于模型名推断，无需加载）"""
        if "ViT-L" in self.model_name or "large" in self.model_name.lower():
            return 768
        return 512

    def _select_device(self, device: Optional[str]) -> str:
        """选择运行设备"""
        if device is not None:
            return device

        # Mac平台直接使用CPU，避免mutex错误
        if platform.system() == "Darwin":
            return "cpu"

        if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
            return "cpu"

        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def initialize(self):
        """懒加载模型"""
        if self._initialized:
            return

        logger.info(f"正在加载CLIP模型: {self.model_name}")

        try:
            if self.use_huggingface:
                self._init_huggingface()
            else:
                self._init_openai_clip()

            self._initialized = True
            logger.info(f"✅ CLIP模型加载完成")

        except Exception as e:
            logger.error(f"❌ CLIP模型加载失败: {e}")
            raise

    def _init_huggingface(self):
        """初始化HuggingFace CLIP"""
        from transformers import CLIPModel, CLIPProcessor

        # 根据模型名构建HuggingFace模型名
        hf_model_name = self.model_name
        if not hf_model_name.startswith("openai/"):
            # 转换标准CLIP模型名为HF格式
            if self.model_name == "ViT-B/32":
                hf_model_name = "openai/clip-vit-base-patch32"
            elif self.model_name == "ViT-B/16":
                hf_model_name = "openai/clip-vit-base-patch16"
            elif self.model_name == "ViT-L/14":
                hf_model_name = "openai/clip-vit-large-patch14"
            elif self.model_name == "ViT-L/14@336px":
                hf_model_name = "openai/clip-vit-large-patch14-336"
            else:
                hf_model_name = "openai/clip-vit-base-patch32"

        self._processor = CLIPProcessor.from_pretrained(hf_model_name)
        self._model = CLIPModel.from_pretrained(hf_model_name)
        self._model = self._model.to(self.device)
        self._model.eval()
        self._model_type = "hf"

    def _init_openai_clip(self):
        """初始化OpenAI CLIP"""
        import clip

        model, preprocess = clip.load(self.model_name, device=self.device)
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = clip.tokenize
        self._model_type = "openai"

    def _load_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """加载并转换图片为RGB PIL Image"""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"不支持的图片输入类型: {type(image_input)}")
        return image

    def embed_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Optional[np.ndarray]:
        """
        提取单张图片的特征向量

        Args:
            image_input: 图片路径/PIL Image/numpy数组

        Returns:
            归一化后的特征向量 (D,) 或 None (失败时)
        """
        if not self._initialized:
            self.initialize()

        try:
            image = self._load_image(image_input)

            if self._model_type == "hf":
                import torch
                inputs = self._processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    features = self._model.get_image_features(**inputs)
            else:
                import torch
                image_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self._model.encode_image(image_tensor)

            # 归一化
            embedding = features.cpu().numpy().flatten()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"提取图片特征失败: {e}")
            return None

    def embed_images(
        self,
        image_inputs: List[Union[str, Image.Image, np.ndarray]],
        batch_size: int = 16,
    ) -> List[Optional[np.ndarray]]:
        """
        批量提取图片特征

        Args:
            image_inputs: 图片列表
            batch_size: 批处理大小

        Returns:
            特征向量列表
        """
        if not self._initialized:
            self.initialize()

        results: List[Optional[np.ndarray]] = []

        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i:i + batch_size]
            batch_results = self._embed_batch(batch)
            results.extend(batch_results)

        return results

    def _embed_batch(
        self,
        batch: List[Union[str, Image.Image, np.ndarray]],
    ) -> List[Optional[np.ndarray]]:
        """批量处理一个批次"""
        try:
            images = [self._load_image(img) for img in batch]

            if self._model_type == "hf":
                import torch
                inputs = self._processor(images=images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    features = self._model.get_image_features(**inputs)
            else:
                import torch
                image_tensors = torch.stack(
                    [self._preprocess(img) for img in images]
                ).to(self.device)
                with torch.no_grad():
                    features = self._model.encode_image(image_tensors)

            features_np = features.cpu().numpy()

            # 归一化每个向量
            results = []
            for vec in features_np:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                results.append(vec.astype(np.float32))

            return results

        except Exception as e:
            logger.error(f"批量特征提取失败: {e}")
            return [None] * len(batch)

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        提取文本特征（用于多模态融合）

        Args:
            text: 输入文本

        Returns:
            归一化的文本特征向量
        """
        if not self._initialized:
            self.initialize()

        try:
            if self._model_type == "hf":
                import torch
                inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    features = self._model.get_text_features(**inputs)
            else:
                import torch
                tokens = self._tokenize([text]).to(self.device)
                with torch.no_grad():
                    features = self._model.encode_text(tokens)

            embedding = features.cpu().numpy().flatten()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"提取文本特征失败: {e}")
            return None
    
    def embed_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        批量提取文本特征

        Args:
            texts: 文本列表

        Returns:
            文本特征矩阵 (len(texts), embedding_dim)
        """
        if not self._initialized:
            self.initialize()
        
        if not texts:
            return None
        
        try:
            if self._model_type == "hf":
                import torch
                inputs = self._processor(text=texts, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    features = self._model.get_text_features(**inputs)
            else:
                import torch
                tokens = self._tokenize(texts).to(self.device)
                with torch.no_grad():
                    features = self._model.encode_text(tokens)
            
            embeddings = features.cpu().numpy()
            # 归一化
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            return embeddings.astype(np.float32)
        
        except Exception as e:
            logger.error(f"批量提取文本特征失败: {e}")
            return None

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
