#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带缓存的CLIP特征提取器
在CLIPEmbedder基础上增加特征缓存功能，避免重复计算
"""

import os
import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CLIPFeatureCache:
    """
    CLIP特征缓存管理器
    支持内存缓存和磁盘缓存
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_memory_cache: int = 1000):
        """
        初始化缓存
        
        Args:
            cache_dir: 磁盘缓存目录，None则只使用内存缓存
            max_memory_cache: 内存缓存最大条目数
        """
        self.memory_cache = {}
        self.max_memory_cache = max_memory_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"CLIP特征缓存目录: {self.cache_dir}")
        
        self._access_count = 0
        self._hit_count = 0
    
    def _compute_key(self, image_input: Union[str, Image.Image, np.ndarray]) -> str:
        """计算图片的缓存键"""
        if isinstance(image_input, str):
            # 文件路径 - 使用文件修改时间和路径哈希
            file_path = Path(image_input)
            if file_path.exists():
                stat = file_path.stat()
                key_data = f"{image_input}:{stat.st_mtime}:{stat.st_size}"
                return hashlib.md5(key_data.encode()).hexdigest()
            else:
                return hashlib.md5(image_input.encode()).hexdigest()
        elif isinstance(image_input, Image.Image):
            # PIL Image - 使用图片内容哈希
            img_bytes = pickle.dumps(image_input.tobytes())
            return hashlib.md5(img_bytes).hexdigest()
        elif isinstance(image_input, np.ndarray):
            # Numpy数组 - 使用数组内容哈希
            return hashlib.md5(image_input.tobytes()).hexdigest()
        else:
            raise ValueError(f"不支持的输入类型: {type(image_input)}")
    
    def get(self, image_input: Union[str, Image.Image, np.ndarray]) -> Optional[np.ndarray]:
        """
        获取缓存的特征
        
        Args:
            image_input: 图片输入
            
        Returns:
            缓存的特征向量或None
        """
        key = self._compute_key(image_input)
        self._access_count += 1
        
        # 先检查内存缓存
        if key in self.memory_cache:
            self._hit_count += 1
            logger.debug(f"内存缓存命中: {key[:8]}...")
            return self.memory_cache[key]
        
        # 再检查磁盘缓存
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.npy"
            if cache_file.exists():
                try:
                    feature = np.load(cache_file)
                    # 放入内存缓存
                    self._add_to_memory(key, feature)
                    self._hit_count += 1
                    logger.debug(f"磁盘缓存命中: {key[:8]}...")
                    return feature
                except Exception as e:
                    logger.warning(f"读取缓存文件失败: {e}")
        
        return None
    
    def set(self, image_input: Union[str, Image.Image, np.ndarray], feature: np.ndarray):
        """
        设置缓存
        
        Args:
            image_input: 图片输入
            feature: 特征向量
        """
        key = self._compute_key(image_input)
        
        # 放入内存缓存
        self._add_to_memory(key, feature)
        
        # 写入磁盘缓存
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.npy"
            try:
                np.save(cache_file, feature)
            except Exception as e:
                logger.warning(f"写入缓存文件失败: {e}")
    
    def _add_to_memory(self, key: str, feature: np.ndarray):
        """添加到内存缓存（LRU策略）"""
        if len(self.memory_cache) >= self.max_memory_cache:
            # 移除最旧的条目（简单LRU）
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = feature
    
    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        hit_rate = self._hit_count / max(self._access_count, 1) * 100
        return {
            "memory_cache_size": len(self.memory_cache),
            "max_memory_cache": self.max_memory_cache,
            "disk_cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "total_access": self._access_count,
            "total_hits": self._hit_count,
            "hit_rate": f"{hit_rate:.2f}%",
        }
    
    def clear(self):
        """清空缓存"""
        self.memory_cache.clear()
        if self.cache_dir:
            for f in self.cache_dir.glob("*.npy"):
                f.unlink()
        logger.info("缓存已清空")


class CLIPEmbedderCached:
    """
    带缓存的CLIP特征提取器
    包装CLIPEmbedder，增加特征缓存功能
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        use_huggingface: bool = True,
        cache_dir: Optional[str] = None,
        max_memory_cache: int = 1000,
    ):
        """
        初始化
        
        Args:
            model_name: CLIP模型名称
            device: 运行设备
            use_huggingface: 是否使用HuggingFace CLIP
            cache_dir: 磁盘缓存目录
            max_memory_cache: 内存缓存最大条目数
        """
        from .clip_embedder import CLIPEmbedder
        
        self.embedder = CLIPEmbedder(model_name, device, use_huggingface)
        self.cache = CLIPFeatureCache(cache_dir, max_memory_cache)
        
        logger.info(f"CLIPEmbedderCached 初始化完成，缓存目录: {cache_dir}")
    
    @property
    def embedding_dim(self) -> int:
        return self.embedder.embedding_dim
    
    def embed_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Optional[np.ndarray]:
        """
        提取单张图片特征（带缓存）
        
        Args:
            image_input: 图片输入
            
        Returns:
            特征向量
        """
        # 先检查缓存
        cached = self.cache.get(image_input)
        if cached is not None:
            return cached
        
        # 计算特征
        feature = self.embedder.embed_image(image_input)
        
        # 存入缓存
        if feature is not None:
            self.cache.set(image_input, feature)
        
        return feature
    
    def embed_images(
        self,
        image_inputs: List[Union[str, Image.Image, np.ndarray]],
        batch_size: int = 16,
    ) -> List[Optional[np.ndarray]]:
        """
        批量提取图片特征（带缓存）
        
        Args:
            image_inputs: 图片列表
            batch_size: 批处理大小
            
        Returns:
            特征向量列表
        """
        results = []
        uncached_inputs = []
        uncached_indices = []
        
        # 先检查缓存
        for i, img_input in enumerate(image_inputs):
            cached = self.cache.get(img_input)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_inputs.append(img_input)
                uncached_indices.append(i)
        
        # 批量计算未缓存的特征
        if uncached_inputs:
            features = self.embedder.embed_images(uncached_inputs, batch_size)
            
            # 存入缓存并填充结果
            for idx, feature in zip(uncached_indices, features):
                if feature is not None:
                    self.cache.set(image_inputs[idx], feature)
                results[idx] = feature
        
        return results
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """提取文本特征（文本不做缓存）"""
        return self.embedder.embed_text(text)
    
    def embed_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        批量提取文本特征
        
        Args:
            texts: 文本列表
            
        Returns:
            文本特征矩阵 (len(texts), embedding_dim)
        """
        return self.embedder.embed_texts(texts)
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()


if __name__ == "__main__":
    # 测试
    embedder = CLIPEmbedderCached(cache_dir="./clip_cache")
    
    # 第一次提取（无缓存）
    import time
    start = time.time()
    feat1 = embedder.embed_image("test.jpg") if os.path.exists("test.jpg") else None
    print(f"第一次: {time.time()-start:.3f}s")
    
    # 第二次提取（有缓存）
    if os.path.exists("test.jpg"):
        start = time.time()
        feat2 = embedder.embed_image("test.jpg")
        print(f"第二次: {time.time()-start:.3f}s")
    
    print(embedder.get_cache_stats())
