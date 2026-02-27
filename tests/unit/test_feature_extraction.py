#!/usr/bin/env python3
"""
特征提取模块单元测试
"""
import os
import sys
import unittest
import numpy as np
from PIL import Image
import tempfile

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.feature_extraction.feature_extraction import FeatureExtraction

class TestFeatureExtraction(unittest.TestCase):
    """测试特征提取模块"""
    
    def setUp(self):
        """设置测试环境"""
        self.extractor = FeatureExtraction()
        
        # 创建测试图像
        self.test_image = Image.new('RGB', (224, 224), color='red')
        
        # 保存为临时文件
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        self.test_image.save(self.temp_file.name)
        self.temp_file.close()
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.extractor, FeatureExtraction)
        self.assertTrue(hasattr(self.extractor, 'model'))
    
    def test_extract_features(self):
        """测试特征提取"""
        # 提取特征
        feature = self.extractor.extract_features(self.test_image)
        
        # 检查特征形状
        self.assertEqual(feature.shape, (768,))  # CLIP特征维度
        
        # 检查特征值是否为数值
        self.assertTrue(np.issubdtype(feature.dtype, np.floating))

if __name__ == '__main__':
    unittest.main()
