#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理器

负责数据的预处理操作
"""

import os
import json
import numpy as np
import pandas as pd
from src.core.logging.global_logger import get_logger

logger = get_logger("preprocessing")


class DataPreprocessor:
    """
    数据预处理器类
    """
    
    def __init__(self):
        """
        初始化数据预处理器
        """
        pass
    
    def preprocess_csv(self, csv_path, output_path=None):
        """
        预处理CSV文件
        
        Args:
            csv_path: CSV文件路径
            output_path: 输出路径
        
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            logger.info(f"读取CSV文件: {csv_path}，共 {len(df)} 行")
            
            # 基本预处理
            df = self._basic_preprocess(df)
            
            # 保存预处理后的数据
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path, index=False)
                logger.info(f"保存预处理后的数据到: {output_path}")
            
            return df
        except Exception as e:
            logger.error(f"预处理CSV文件失败: {e}")
            return None
    
    def preprocess_json(self, json_path, output_path=None):
        """
        预处理JSON文件
        
        Args:
            json_path: JSON文件路径
            output_path: 输出路径
        
        Returns:
            dict: 预处理后的数据
        """
        try:
            # 读取JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"读取JSON文件: {json_path}")
            
            # 基本预处理
            data = self._basic_preprocess_json(data)
            
            # 保存预处理后的数据
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"保存预处理后的数据到: {output_path}")
            
            return data
        except Exception as e:
            logger.error(f"预处理JSON文件失败: {e}")
            return None
    
    def _basic_preprocess(self, df):
        """
        基本预处理
        
        Args:
            df: DataFrame
        
        Returns:
            pd.DataFrame: 预处理后的DataFrame
        """
        try:
            # 去除空值
            df = df.dropna()
            
            # 去除重复值
            df = df.drop_duplicates()
            
            # 重置索引
            df = df.reset_index(drop=True)
            
            return df
        except Exception as e:
            logger.error(f"基本预处理失败: {e}")
            return df
    
    def _basic_preprocess_json(self, data):
        """
        基本预处理JSON数据
        
        Args:
            data: JSON数据
        
        Returns:
            dict: 预处理后的数据
        """
        try:
            # 这里可以添加JSON数据的预处理逻辑
            return data
        except Exception as e:
            logger.error(f"基本预处理JSON数据失败: {e}")
            return data
    
    def normalize_data(self, data, feature_range=(0, 1)):
        """
        归一化数据
        
        Args:
            data: 数据
            feature_range: 特征范围
        
        Returns:
            np.array: 归一化后的数据
        """
        try:
            data = np.array(data)
            min_val = np.min(data)
            max_val = np.max(data)
            
            if max_val - min_val == 0:
                return data
            
            normalized = (data - min_val) / (max_val - min_val)
            normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
            
            return normalized
        except Exception as e:
            logger.error(f"归一化数据失败: {e}")
            return data
    
    def standardize_data(self, data):
        """
        标准化数据
        
        Args:
            data: 数据
        
        Returns:
            np.array: 标准化后的数据
        """
        try:
            data = np.array(data)
            mean = np.mean(data)
            std = np.std(data)
            
            if std == 0:
                return data
            
            standardized = (data - mean) / std
            
            return standardized
        except Exception as e:
            logger.error(f"标准化数据失败: {e}")
            return data
    
    def one_hot_encode(self, data, categories=None):
        """
        独热编码
        
        Args:
            data: 数据
            categories: 类别
        
        Returns:
            np.array: 独热编码后的数据
        """
        try:
            if categories is None:
                categories = np.unique(data)
            
            one_hot = np.zeros((len(data), len(categories)))
            for i, item in enumerate(data):
                if item in categories:
                    one_hot[i, np.where(categories == item)[0][0]] = 1
            
            return one_hot
        except Exception as e:
            logger.error(f"独热编码失败: {e}")
            return data
