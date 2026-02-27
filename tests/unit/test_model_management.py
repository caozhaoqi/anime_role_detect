#!/usr/bin/env python3
"""
模型管理模块单元测试
"""
import os
import sys
import unittest
import tempfile
import torch
import torch.nn as nn

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.deployment.deploy_model import ModelManager
from src.models.training.train_simple import SimpleCharacterClassifier

class TestModelManagement(unittest.TestCase):
    """测试模型管理模块"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时模型文件
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_model.pth')
        
        # 创建简单模型
        num_classes = 2
        model = SimpleCharacterClassifier(num_classes=num_classes)
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_to_idx': {'class_0': 0, 'class_1': 1},
            'val_acc': 0.5
        }, self.model_path)
        
        # 初始化模型管理器
        self.model_configs = [
            {
                'name': 'test_model',
                'path': self.model_path,
                'type': 'simple'
            }
        ]
        
        self.model_manager = ModelManager(self.model_configs)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.model_manager, ModelManager)
        self.assertTrue('test_model' in self.model_manager.models)
    
    def test_get_model(self):
        """测试获取模型"""
        model_info = self.model_manager.get_model('test_model')
        self.assertIsInstance(model_info, dict)
        self.assertTrue('model' in model_info)
        self.assertTrue('class_to_idx' in model_info)
        self.assertTrue('idx_to_class' in model_info)
    
    def test_list_models(self):
        """测试列出模型"""
        models = self.model_manager.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]['name'], 'test_model')
    
    def test_predict(self):
        """测试预测"""
        # 创建测试图像
        from PIL import Image
        import io
        
        test_image = Image.new('RGB', (224, 224), color='red')
        image_content = io.BytesIO()
        test_image.save(image_content, format='JPEG')
        image_content.seek(0)
        
        # 预测
        result = self.model_manager.predict(image_content.getvalue(), model_name='test_model')
        
        # 检查结果
        self.assertIsInstance(result, dict)
        self.assertTrue('predicted_class' in result)
        self.assertTrue('confidence' in result)
        self.assertTrue('top5' in result)

if __name__ == '__main__':
    unittest.main()
