#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量库检索系统

实现大规模角色库的特征存储和相似度检索，支持未见过角色的相似度匹配
"""

import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import logging
import json
from tqdm import tqdm

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入faiss
import faiss

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('vector_db')

# 导入torchvision模型
import torchvision.models as models

# 定义ArcFace模型用于特征提取
class ArcFaceModel(nn.Module):
    def __init__(self, num_classes=131, embedding_size=512):
        super(ArcFaceModel, self).__init__()
        # 使用EfficientNet-B0作为基础模型
        self.backbone = models.efficientnet_b0(pretrained=False)
        # 获取特征维度
        self.feature_dim = self.backbone.classifier[1].in_features
        # 替换分类头为特征提取层
        self.backbone.classifier = nn.Identity()
        # 添加特征投影层
        self.projection = nn.Linear(self.feature_dim, embedding_size)
    
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        # 投影到目标维度
        embeddings = self.projection(features)
        # L2归一化
        embeddings = nn.functional.normalize(embeddings, dim=1)
        return embeddings

class VectorDBSystem:
    def __init__(self, model_path, embedding_size=512, index_type='cosine'):
        """
        初始化向量库系统
        
        Args:
            model_path: 模型权重文件路径
            embedding_size: 特征维度
            index_type: 索引类型 ('cosine', 'l2')
        """
        self.model_path = model_path
        self.embedding_size = embedding_size
        self.index_type = index_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self._load_model()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 初始化FAISS索引
        self.index = None
        self.id_to_info = {}
        self.next_id = 0
    
    def _load_model(self):
        """
        加载特征提取模型
        """
        logger.info(f"加载特征提取模型: {self.model_path}")
        model = ArcFaceModel(embedding_size=self.embedding_size)
        
        # 加载权重
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            # 处理不同格式的权重文件
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            elif 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                # 尝试加载普通权重
                model.load_state_dict(state_dict)
            logger.info("模型权重加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            # 尝试使用简化的模型结构
            logger.info("尝试使用简化模型结构")
            # 创建简化模型
            model = nn.Sequential(
                models.efficientnet_b0(pretrained=False),
                nn.Linear(1000, self.embedding_size),
                nn.LayerNorm(self.embedding_size)
            )
            logger.info("使用简化模型结构")
        
        # 设置为评估模式
        model.eval()
        return model
    
    def _create_index(self):
        """
        创建FAISS索引
        """
        logger.info(f"创建FAISS索引，类型: {self.index_type}")
        
        if self.index_type == 'cosine':
            # 使用内积索引（余弦相似度）
            self.index = faiss.IndexFlatIP(self.embedding_size)
        else:
            # 使用L2距离索引
            self.index = faiss.IndexFlatL2(self.embedding_size)
        
        logger.info("FAISS索引创建成功")
    
    def extract_features(self, image_path):
        """
        提取单张图像的特征
        
        Args:
            image_path: 图像路径
        
        Returns:
            特征向量
        """
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            # 预处理
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                embedding = self.model(image_tensor)
            
            return embedding.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return None
    
    def batch_extract_features(self, image_dir, batch_size=32):
        """
        批量提取图像特征
        
        Args:
            image_dir: 图像目录
            batch_size: 批量大小
        
        Returns:
            特征列表和信息列表
        """
        logger.info(f"批量提取特征，目录: {image_dir}, 批量大小: {batch_size}")
        
        features = []
        info_list = []
        
        # 收集所有图像
        image_paths = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append(os.path.join(root, file))
        
        logger.info(f"找到 {len(image_paths)} 张图像")
        
        # 批量处理
        for i in tqdm(range(0, len(image_paths), batch_size), desc="提取特征"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                except Exception as e:
                    logger.warning(f"无法加载图像: {path}, 错误: {e}")
            
            if batch_images:
                # 创建批量张量
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # 提取特征
                with torch.no_grad():
                    batch_embeddings = self.model(batch_tensor)
                
                # 添加到列表
                features.extend(batch_embeddings.cpu().numpy())
                
                # 添加信息
                for path in batch_paths:
                    # 从路径中提取角色信息
                    parts = path.split(os.sep)
                    role_info = {
                        'path': path,
                        'role': parts[-2] if len(parts) >= 2 else 'unknown',
                        'series': parts[-3] if len(parts) >= 3 else 'unknown'
                    }
                    info_list.append(role_info)
        
        logger.info(f"批量特征提取完成，成功提取 {len(features)} 个特征")
        return features, info_list
    
    def add_to_index(self, features, info_list):
        """
        添加特征到索引
        
        Args:
            features: 特征列表
            info_list: 信息列表
        """
        if not features:
            logger.warning("没有特征需要添加到索引")
            return
        
        # 如果索引未创建，先创建
        if self.index is None:
            self._create_index()
        
        logger.info(f"添加 {len(features)} 个特征到索引")
        
        # 转换为numpy数组
        features_np = np.array(features, dtype=np.float32)
        
        # 添加到索引
        self.index.add(features_np)
        
        # 保存信息
        for i, info in enumerate(info_list):
            self.id_to_info[self.next_id + i] = info
        
        self.next_id += len(features)
        logger.info(f"索引更新完成，当前索引大小: {self.next_id}")
    
    def build_index(self, image_dir, batch_size=32):
        """
        构建向量索引
        
        Args:
            image_dir: 图像目录
            batch_size: 批量大小
        """
        # 批量提取特征
        features, info_list = self.batch_extract_features(image_dir, batch_size)
        
        # 添加到索引
        self.add_to_index(features, info_list)
        
        logger.info("向量索引构建完成")
    
    def search(self, query_embedding, k=5):
        """
        搜索相似向量
        
        Args:
            query_embedding: 查询特征向量
            k: 返回结果数量
        
        Returns:
            搜索结果
        """
        if self.index is None:
            logger.error("索引未初始化")
            return []
        
        # 确保特征向量维度正确
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # 搜索
        distances, indices = self.index.search(query_embedding, k)
        
        # 处理结果
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < 0:
                continue
            
            distance = distances[0][i]
            # 转换为相似度
            if self.index_type == 'cosine':
                similarity = distance
            else:
                # L2距离转换为相似度
                similarity = 1.0 / (1.0 + distance)
            
            info = self.id_to_info.get(idx, {})
            results.append({
                'id': idx,
                'similarity': similarity,
                'info': info
            })
        
        return results
    
    def search_by_image(self, image_path, k=5):
        """
        通过图像搜索相似角色
        
        Args:
            image_path: 查询图像路径
            k: 返回结果数量
        
        Returns:
            搜索结果
        """
        # 提取特征
        query_embedding = self.extract_features(image_path)
        if query_embedding is None:
            return []
        
        # 搜索
        results = self.search(query_embedding, k)
        
        logger.info(f"搜索完成，找到 {len(results)} 个相似角色")
        return results
    
    def save_index(self, index_path, info_path):
        """
        保存索引
        
        Args:
            index_path: 索引保存路径
            info_path: 信息保存路径
        """
        if self.index is None:
            logger.error("索引未初始化")
            return False
        
        try:
            # 保存FAISS索引
            faiss.write_index(self.index, index_path)
            logger.info(f"FAISS索引保存成功: {index_path}")
            
            # 保存信息
            info_data = {
                'id_to_info': self.id_to_info,
                'next_id': self.next_id,
                'embedding_size': self.embedding_size,
                'index_type': self.index_type
            }
            
            with open(info_path, 'wb') as f:
                pickle.dump(info_data, f)
            logger.info(f"索引信息保存成功: {info_path}")
            
            return True
        except Exception as e:
            logger.error(f"索引保存失败: {e}")
            return False
    
    def load_index(self, index_path, info_path):
        """
        加载索引
        
        Args:
            index_path: 索引路径
            info_path: 信息路径
        """
        try:
            # 加载FAISS索引
            self.index = faiss.read_index(index_path)
            logger.info(f"FAISS索引加载成功: {index_path}")
            
            # 加载信息
            with open(info_path, 'rb') as f:
                info_data = pickle.load(f)
            
            self.id_to_info = info_data['id_to_info']
            self.next_id = info_data['next_id']
            self.embedding_size = info_data['embedding_size']
            self.index_type = info_data['index_type']
            
            logger.info(f"索引信息加载成功: {info_path}")
            logger.info(f"索引包含 {self.next_id} 个向量")
            
            return True
        except Exception as e:
            logger.error(f"索引加载失败: {e}")
            return False
    
    def get_statistics(self):
        """
        获取索引统计信息
        """
        if self.index is None:
            return {
                'index_size': 0,
                'embedding_size': self.embedding_size,
                'index_type': self.index_type
            }
        
        return {
            'index_size': self.next_id,
            'embedding_size': self.embedding_size,
            'index_type': self.index_type,
            'dimension': self.embedding_size
        }

def main():
    parser = argparse.ArgumentParser(description='向量库检索系统')
    
    parser.add_argument('--model-path', type=str, 
                       default='models/character_classifier_best_improved.pth',
                       help='模型权重文件路径')
    parser.add_argument('--build-index', type=str,
                       help='构建索引的图像目录')
    parser.add_argument('--search-image', type=str,
                       help='搜索的查询图像')
    parser.add_argument('--index-path', type=str,
                       default='models/vector_index.faiss',
                       help='索引保存路径')
    parser.add_argument('--info-path', type=str,
                       default='models/vector_index_info.pkl',
                       help='索引信息保存路径')
    parser.add_argument('--k', type=int, default=5,
                       help='返回结果数量')
    parser.add_argument('--embedding-size', type=int, default=512,
                       help='特征维度')
    
    args = parser.parse_args()
    
    # 初始化系统
    system = VectorDBSystem(
        model_path=args.model_path,
        embedding_size=args.embedding_size
    )
    
    # 检查是否需要加载已有索引
    if os.path.exists(args.index_path) and os.path.exists(args.info_path):
        logger.info("加载已有索引")
        system.load_index(args.index_path, args.info_path)
    
    # 构建索引
    if args.build_index:
        system.build_index(args.build_index)
        # 保存索引
        system.save_index(args.index_path, args.info_path)
    
    # 搜索
    if args.search_image:
        results = system.search_by_image(args.search_image, args.k)
        
        # 打印结果
        print("\n" + "="*60)
        print("相似度检索结果")
        print("="*60)
        for i, result in enumerate(results, 1):
            similarity = result['similarity']
            info = result['info']
            print(f"{i}. 相似度: {similarity:.4f}")
            print(f"   角色: {info.get('role', 'unknown')}")
            print(f"   系列: {info.get('series', 'unknown')}")
            print(f"   路径: {info.get('path', 'unknown')}")
            print()
        print("="*60)
    
    # 打印统计信息
    stats = system.get_statistics()
    logger.info(f"索引统计信息: {stats}")

if __name__ == '__main__':
    main()
