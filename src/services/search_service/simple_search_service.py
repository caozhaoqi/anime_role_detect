#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版图像搜索服务 - 不使用CLIP模型，避免macOS锁问题
使用预计算的特征或传统图像匹配算法
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image
import hashlib

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.core.logging.global_logger import get_logger

logger = get_logger("simple_search_service")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS不可用，将使用简单的距离计算")

class SimpleImageSearchService:
    """简化版图像搜索服务"""
    
    def __init__(self, index_path: str = None):
        self.index_path = index_path or os.path.join(project_root, "data", "search_index")
        self.index = None
        self.image_paths = []
        self.index_initialized = False
        
    def _compute_histogram(self, image: Image) -> np.ndarray:
        """计算图像的颜色直方图特征"""
        # 转换为HSV色彩空间
        hsv = image.convert('HSV')
        # 计算直方图
        h_hist = np.histogram(np.array(hsv.split()[0]), bins=8, range=(0, 256))[0]
        s_hist = np.histogram(np.array(hsv.split()[1]), bins=8, range=(0, 256))[0]
        v_hist = np.histogram(np.array(hsv.split()[2]), bins=8, range=(0, 256))[0]
        # 归一化并合并
        hist = np.concatenate([h_hist, s_hist, v_hist])
        return hist / hist.sum()
    
    def _compute_phash(self, image: Image) -> np.ndarray:
        """计算图像的感知哈希"""
        # 缩小到32x32
        img = image.resize((32, 32), Image.LANCZOS).convert('L')
        # 使用简单的均值哈希
        pixels = np.array(img)
        # 计算均值
        mean = pixels.mean()
        # 生成哈希
        hash_bits = (pixels > mean).astype(np.float32).flatten()
        return hash_bits
    
    def _compute_feature(self, image: Image) -> np.ndarray:
        """计算图像特征（组合多种特征）"""
        hist = self._compute_histogram(image)
        phash = self._compute_phash(image)
        return np.concatenate([hist, phash])
    
    def build_index_from_dataset(self, dataset_dir: str) -> int:
        """从数据集目录构建索引"""
        logger.info(f"从数据集目录构建索引: {dataset_dir}")
        
        if not os.path.exists(dataset_dir):
            logger.error(f"数据集目录不存在: {dataset_dir}")
            return 0
        
        features = []
        self.image_paths = []
        
        # 遍历数据集目录
        for role_name in os.listdir(dataset_dir):
            role_dir = os.path.join(dataset_dir, role_name)
            if not os.path.isdir(role_dir):
                continue
            
            for img_file in os.listdir(role_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    img_path = os.path.join(role_dir, img_file)
                    try:
                        # 打开图像并计算特征
                        image = Image.open(img_path).convert('RGB')
                        feature = self._compute_feature(image)
                        features.append(feature)
                        self.image_paths.append(img_path)
                    except Exception as e:
                        logger.warning(f"无法处理图像: {img_path}, 错误: {e}")
        
        if not features:
            logger.warning("未找到任何图像")
            return 0
        
        # 创建索引
        features = np.array(features).astype(np.float32)
        feature_dim = features.shape[1]
        
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(feature_dim)
            self.index.add(features)
            logger.info(f"使用FAISS索引，维度: {feature_dim}")
        else:
            self.index = features
            logger.info(f"使用简单索引，维度: {feature_dim}")
        
        self.index_initialized = True
        logger.info(f"索引构建完成，共添加 {len(self.image_paths)} 张图像")
        return len(self.image_paths)
    
    def search(self, image: Image, top_k: int = 10) -> list:
        """搜索相似图像"""
        if not self.index_initialized:
            logger.error("索引未初始化")
            return []
        
        # 计算查询图像特征
        query_feature = self._compute_feature(image).astype(np.float32)
        
        if FAISS_AVAILABLE:
            # 使用FAISS搜索
            query_feature = query_feature.reshape(1, -1)
            distances, indices = self.index.search(query_feature, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.image_paths):
                    similarity = 1.0 / (1.0 + distances[0][i])  # 转换为相似度
                    results.append((self.image_paths[idx], similarity))
        else:
            # 使用简单的L2距离计算
            query_feature = query_feature.reshape(1, -1)
            features = self.index
            
            # 计算距离
            distances = np.sqrt(np.sum((features - query_feature)**2, axis=1))
            
            # 获取前top_k个
            indices = np.argsort(distances)[:top_k]
            
            results = []
            for idx in indices:
                similarity = 1.0 / (1.0 + distances[idx])
                results.append((self.image_paths[idx], similarity))
        
        return results
    
    def search_by_path(self, image_path: str, top_k: int = 10) -> list:
        """通过图像路径搜索相似图像"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.search(image, top_k)
        except Exception as e:
            logger.error(f"无法读取图像: {image_path}, 错误: {e}")
            return []
    
    def save_index(self):
        """保存索引到文件"""
        os.makedirs(self.index_path, exist_ok=True)
        
        # 保存图像路径
        paths_file = os.path.join(self.index_path, "image_paths.pkl")
        with open(paths_file, 'wb') as f:
            pickle.dump(self.image_paths, f)
        
        # 保存索引
        if FAISS_AVAILABLE and self.index is not None:
            index_file = os.path.join(self.index_path, "faiss_index.index")
            faiss.write_index(self.index, index_file)
        elif self.index is not None:
            index_file = os.path.join(self.index_path, "simple_index.npy")
            np.save(index_file, self.index)
        
        logger.info(f"索引已保存到: {self.index_path}")
    
    def load_index(self) -> bool:
        """从文件加载索引"""
        if not os.path.exists(self.index_path):
            return False
        
        try:
            # 加载图像路径
            paths_file = os.path.join(self.index_path, "image_paths.pkl")
            with open(paths_file, 'rb') as f:
                self.image_paths = pickle.load(f)
            
            # 加载索引
            if FAISS_AVAILABLE:
                index_file = os.path.join(self.index_path, "faiss_index.index")
                if os.path.exists(index_file):
                    self.index = faiss.read_index(index_file)
                else:
                    return False
            else:
                index_file = os.path.join(self.index_path, "simple_index.npy")
                if os.path.exists(index_file):
                    self.index = np.load(index_file)
                else:
                    return False
            
            self.index_initialized = True
            logger.info(f"索引加载成功，共 {len(self.image_paths)} 张图像")
            return True
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False
    
    def get_index_stats(self) -> dict:
        """获取索引统计信息"""
        if not self.index_initialized:
            return {
                "status": "not_initialized",
                "total_images": 0,
                "index_dimension": 0
            }
        
        if FAISS_AVAILABLE:
            return {
                "status": "ready",
                "total_images": self.index.ntotal,
                "index_dimension": self.index.d
            }
        else:
            return {
                "status": "ready",
                "total_images": len(self.image_paths),
                "index_dimension": self.index.shape[1] if self.index is not None else 0
            }

# 创建全局服务实例
simple_search_service = SimpleImageSearchService()

def get_simple_search_service() -> SimpleImageSearchService:
    """获取简化版搜索服务实例"""
    return simple_search_service

if __name__ == "__main__":
    # 测试
    service = SimpleImageSearchService()
    dataset_dir = os.path.join(project_root, "data", "merged_english_dataset")
    count = service.build_index_from_dataset(dataset_dir)
    print(f"构建索引完成，共 {count} 张图像")
    
    stats = service.get_index_stats()
    print(f"索引统计: {stats}")
    
    # 测试搜索
    if count > 0:
        test_path = service.image_paths[0]
        results = service.search_by_path(test_path, top_k=5)
        print(f"\n搜索结果（以第一张图像为例）:")
        for path, similarity in results:
            role = os.path.basename(os.path.dirname(path))
            print(f"  - {role}/{os.path.basename(path)}: {similarity:.4f}")