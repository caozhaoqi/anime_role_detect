#!/usr/bin/env python3
"""
分类模块单元测试
"""
import os
import sys
import unittest
import numpy as np
import tempfile

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.classification.classification import Classification

class TestClassification(unittest.TestCase):
    """测试分类模块"""
    
    def setUp(self):
        """设置测试环境"""
        self.classifier = Classification(threshold=0.7)
        
        # 创建测试数据
        self.dim = 768  # CLIP特征维度
        self.num_roles = 2
        self.num_samples_per_role = 5
        
        # 生成随机特征向量
        self.features = np.random.randn(self.num_roles * self.num_samples_per_role, self.dim).astype(np.float32)
        # 归一化特征向量
        self.features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)
        
        # 生成角色名称映射
        self.role_names = []
        for i in range(self.num_roles):
            role_name = f"角色{i+1}"
            self.role_names.extend([role_name] * self.num_samples_per_role)
        
        # 构建索引
        self.classifier.build_index(self.features, self.role_names)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.classifier, Classification)
        self.assertEqual(self.classifier.threshold, 0.7)
    
    def test_build_index(self):
        """测试构建索引"""
        self.assertIsNotNone(self.classifier.index)
        self.assertEqual(len(self.classifier.role_mapping), self.num_roles * self.num_samples_per_role)
    
    def test_classify(self):
        """测试分类"""
        # 生成测试特征向量
        test_feature = np.random.randn(self.dim).astype(np.float32)
        test_feature = test_feature / np.linalg.norm(test_feature)
        
        # 分类
        role, similarity = self.classifier.classify(test_feature)
        
        # 检查结果
        self.assertIsInstance(role, str)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_batch_classify(self):
        """测试批量分类"""
        # 生成测试特征向量
        test_features = np.random.randn(3, self.dim).astype(np.float32)
        test_features = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)
        
        # 批量分类
        results = self.classifier.batch_classify(test_features)
        
        # 检查结果
        self.assertEqual(len(results), 3)
        for role, similarity in results:
            self.assertIsInstance(role, str)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)

if __name__ == '__main__':
    unittest.main()
