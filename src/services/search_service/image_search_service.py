#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像搜索服务 - 基于Faiss向量检索实现以图搜图功能
"""

import os
import sys
import time
import threading
import numpy as np
import faiss
from PIL import Image
from typing import List, Dict, Optional, Tuple

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# 延迟导入PyTorch相关模块，避免启动时的锁竞争
torch = None
CLIPProcessor = None
CLIPModel = None

# 导入锁
torch_import_lock = threading.Lock()

from src.core.logging.global_logger import get_logger

logger = get_logger("image_search_service")

def _import_torch():
    """延迟导入PyTorch"""
    global torch, CLIPProcessor, CLIPModel
    with torch_import_lock:
        if torch is None:
            logger.info("延迟导入PyTorch...")
            import torch as _torch
            from transformers import CLIPProcessor as _CLIPProcessor
            from transformers import CLIPModel as _CLIPModel
            torch = _torch
            CLIPProcessor = _CLIPProcessor
            CLIPModel = _CLIPModel
            logger.info("PyTorch导入完成")


class ImageSearchService:
    """
    图像搜索服务
    使用CLIP模型提取图像特征，通过Faiss索引实现快速相似图像搜索
    """
    
    def __init__(self, index_path: str = "image_index", model_name: str = "openai/clip-vit-base-patch32"):
        """
        初始化图像搜索服务
        
        Args:
            index_path: 索引存储路径
            model_name: CLIP模型名称
        """
        self.index_path = index_path
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.index = None
        self.image_paths = []
        self.device = None
        
        # 延迟初始化 - 不在构造函数中加载模型，避免启动时的锁竞争
        self._model_initialized = False
        self._index_loaded = False
    
    def _ensure_initialized(self):
        """延迟初始化 - 确保模型和索引已加载"""
        if not self._model_initialized:
            # 延迟导入PyTorch
            _import_torch()
            
            # 设置设备
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
            
            # 加载模型
            self._load_model()
            self._model_initialized = True
        
        if not self._index_loaded:
            # 加载或创建索引
            self._load_or_create_index()
            self._index_loaded = True
    
    def _load_model(self):
        """加载CLIP模型"""
        try:
            logger.info(f"加载CLIP模型: {self.model_name}")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            logger.info(f"CLIP模型加载完成，使用设备: {self.device}")
        except Exception as e:
            logger.error(f"加载CLIP模型失败: {e}")
    
    def _load_or_create_index(self):
        """加载或创建Faiss索引"""
        os.makedirs(self.index_path, exist_ok=True)
        
        index_file = os.path.join(self.index_path, "image_index.faiss")
        paths_file = os.path.join(self.index_path, "image_paths.npy")
        
        if os.path.exists(index_file) and os.path.exists(paths_file):
            try:
                logger.info("加载已存在的索引...")
                self.index = faiss.read_index(index_file)
                self.image_paths = np.load(paths_file, allow_pickle=True).tolist()
                logger.info(f"索引加载完成，包含 {len(self.image_paths)} 张图像")
            except Exception as e:
                logger.error(f"加载索引失败，创建新索引: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """创建新的Faiss索引"""
        # 使用IVF索引，适合大规模数据
        dimension = 512  # CLIP特征维度
        nlist = 100  # 聚类数
        
        # 创建索引
        quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        self.index.nprobe = 10  # 查询时探索的聚类数
        
        # 标记为未训练
        self.image_paths = []
        logger.info("创建新的Faiss索引")
    
    def _extract_feature(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        提取图像特征
        
        Args:
            image: PIL图像
        
        Returns:
            特征向量 (512维)
        """
        try:
            if self.model is None or self.processor is None:
                logger.error("模型未加载")
                return None
            
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            
            # 归一化
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error(f"提取特征失败: {e}")
            return None
    
    def add_image(self, image_path: str) -> bool:
        """
        添加单张图像到索引
        
        Args:
            image_path: 图像路径
        
        Returns:
            是否成功
        """
        try:
            # 延迟初始化
            self._ensure_initialized()
            
            # 检查图像是否已存在
            if image_path in self.image_paths:
                logger.warning(f"图像已存在: {image_path}")
                return True
            
            # 读取图像
            image = Image.open(image_path).convert("RGB")
            
            # 提取特征
            feature = self._extract_feature(image)
            if feature is None:
                return False
            
            # 添加到索引
            if not self.index.is_trained:
                # 首次添加，先训练索引
                self.index.train(feature)
            
            self.index.add(feature)
            self.image_paths.append(image_path)
            
            logger.debug(f"添加图像: {image_path}")
            return True
        except Exception as e:
            logger.error(f"添加图像失败 {image_path}: {e}")
            return False
    
    def add_images(self, image_paths: List[str]) -> int:
        """
        批量添加图像到索引
        
        Args:
            image_paths: 图像路径列表
        
        Returns:
            成功添加的数量
        """
        success_count = 0
        
        for i, image_path in enumerate(image_paths):
            if self.add_image(image_path):
                success_count += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(image_paths)} 张图像")
        
        return success_count
    
    def build_index_from_dataset(self, dataset_dir: str) -> int:
        """
        从数据集目录构建索引
        
        Args:
            dataset_dir: 数据集目录，包含角色子目录
        
        Returns:
            成功添加的图像数量
        """
        logger.info(f"从数据集目录构建索引: {dataset_dir}")
        
        image_paths = []
        for role_name in os.listdir(dataset_dir):
            role_dir = os.path.join(dataset_dir, role_name)
            if not os.path.isdir(role_dir):
                continue
            
            for image_file in os.listdir(role_dir):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_paths.append(os.path.join(role_dir, image_file))
        
        logger.info(f"找到 {len(image_paths)} 张图像")
        return self.add_images(image_paths)
    
    def search(self, image: Image.Image, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        以图搜图 - 搜索相似图像
        
        Args:
            image: 查询图像
            top_k: 返回前k个结果
        
        Returns:
            相似图像路径和相似度列表 [(path, similarity), ...]
        """
        try:
            # 延迟初始化
            self._ensure_initialized()
            
            if len(self.image_paths) == 0:
                logger.warning("索引为空")
                return []
            
            # 提取查询图像特征
            query_feature = self._extract_feature(image)
            if query_feature is None:
                return []
            
            # 搜索相似图像
            distances, indices = self.index.search(query_feature, top_k)
            
            # 构建结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.image_paths):
                    # L2距离越小越相似，转换为相似度分数 (0-1)
                    max_distance = 4.0  # 归一化后的最大可能距离
                    similarity = max(0, 1 - distances[0][i] / max_distance)
                    results.append((self.image_paths[idx], float(similarity)))
            
            logger.info(f"搜索完成，找到 {len(results)} 个相似图像")
            return results
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def search_by_path(self, image_path: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        通过图像路径搜索相似图像
        
        Args:
            image_path: 查询图像路径
            top_k: 返回前k个结果
        
        Returns:
            相似图像路径和相似度列表
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self.search(image, top_k)
        except Exception as e:
            logger.error(f"读取图像失败 {image_path}: {e}")
            return []
    
    def save_index(self):
        """保存索引到磁盘"""
        try:
            os.makedirs(self.index_path, exist_ok=True)
            
            index_file = os.path.join(self.index_path, "image_index.faiss")
            paths_file = os.path.join(self.index_path, "image_paths.npy")
            
            faiss.write_index(self.index, index_file)
            np.save(paths_file, np.array(self.image_paths))
            
            logger.info(f"索引保存完成: {self.index_path}")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
    
    def get_index_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            "total_images": len(self.image_paths),
            "index_size": self.index.ntotal if self.index else 0,
            "is_trained": self.index.is_trained if self.index else False,
            "dimension": 512
        }


# 全局实例
_search_service = None


def get_image_search_service(index_path: str = "image_index") -> ImageSearchService:
    """获取图像搜索服务实例"""
    global _search_service
    if _search_service is None:
        _search_service = ImageSearchService(index_path=index_path)
    return _search_service


if __name__ == "__main__":
    # 测试图像搜索服务
    import argparse
    
    parser = argparse.ArgumentParser(description="图像搜索服务测试")
    parser.add_argument("--build", action="store_true", help="构建索引")
    parser.add_argument("--dataset", type=str, default="data/merged_english_dataset", help="数据集目录")
    parser.add_argument("--search", type=str, help="搜索图像路径")
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数")
    
    args = parser.parse_args()
    
    # 创建服务实例
    service = ImageSearchService()
    
    if args.build:
        # 构建索引
        logger.info("开始构建索引...")
        count = service.build_index_from_dataset(args.dataset)
        service.save_index()
        logger.info(f"索引构建完成，共添加 {count} 张图像")
    
    if args.search:
        # 搜索相似图像
        logger.info(f"搜索相似图像: {args.search}")
        results = service.search_by_path(args.search, args.top_k)
        
        print("\n搜索结果:")
        for i, (path, similarity) in enumerate(results, 1):
            print(f"{i}. {path} - 相似度: {similarity:.4f}")
