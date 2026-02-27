#!/usr/bin/env python3
"""
数据集成测试
测试数据收集和预处理的集成
"""
import os
import sys
import unittest
import tempfile
import shutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.collection.data_collection import ImageCollector
from src.data.preprocessing.split_dataset import split_dataset

class TestDataIntegration(unittest.TestCase):
    """测试数据集成"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.collection_dir = os.path.join(self.temp_dir, 'collection')
        self.preprocessing_dir = os.path.join(self.temp_dir, 'preprocessing')
        
        # 创建目录
        os.makedirs(self.collection_dir)
        
        # 初始化收集器
        self.collector = ImageCollector()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_data_collection_and_preprocessing(self):
        """测试数据收集和预处理的集成"""
        # 模拟数据收集（实际测试中可以跳过，因为需要网络连接）
        # 这里我们直接创建测试目录结构
        test_class_dir = os.path.join(self.collection_dir, 'test_class')
        os.makedirs(test_class_dir)
        
        # 创建测试图像文件
        for i in range(5):
            with open(os.path.join(test_class_dir, f'test_{i}.jpg'), 'w') as f:
                f.write('test')
        
        # 测试数据预处理
        split_dataset(self.collection_dir, self.preprocessing_dir, val_size=0.2)
        
        # 检查输出目录结构
        train_dir = os.path.join(self.preprocessing_dir, 'train', 'test_class')
        val_dir = os.path.join(self.preprocessing_dir, 'val', 'test_class')
        
        self.assertTrue(os.path.exists(train_dir))
        self.assertTrue(os.path.exists(val_dir))
        
        # 检查文件数量
        train_files = os.listdir(train_dir)
        val_files = os.listdir(val_dir)
        
        self.assertEqual(len(train_files), 4)  # 80% of 5
        self.assertEqual(len(val_files), 1)    # 20% of 5

if __name__ == '__main__':
    unittest.main()
