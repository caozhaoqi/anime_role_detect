#!/usr/bin/env python3
"""
数据预处理模块单元测试
"""
import os
import sys
import unittest
import tempfile
import shutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.preprocessing.split_dataset import split_dataset

class TestDataPreprocessing(unittest.TestCase):
    """测试数据预处理模块"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.train_dir = os.path.join(self.temp_dir, 'train')
        self.val_dir = os.path.join(self.temp_dir, 'val')
        
        # 创建输入目录和测试数据
        os.makedirs(self.input_dir)
        os.makedirs(os.path.join(self.input_dir, 'test_class'))
        
        # 创建测试图像文件
        for i in range(5):
            with open(os.path.join(self.input_dir, 'test_class', f'test_{i}.jpg'), 'w') as f:
                f.write('test')
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_split_dataset(self):
        """测试数据集分割"""
        # 测试分割
        split_dataset(self.input_dir, self.train_dir, self.val_dir, val_ratio=0.2)
        
        # 检查输出目录结构
        train_class_dir = os.path.join(self.train_dir, 'test_class')
        val_class_dir = os.path.join(self.val_dir, 'test_class')
        
        self.assertTrue(os.path.exists(train_class_dir))
        self.assertTrue(os.path.exists(val_class_dir))
        
        # 检查文件数量
        train_files = os.listdir(train_class_dir)
        val_files = os.listdir(val_class_dir)
        
        self.assertEqual(len(train_files), 4)  # 80% of 5
        self.assertEqual(len(val_files), 1)    # 20% of 5

if __name__ == '__main__':
    unittest.main()
