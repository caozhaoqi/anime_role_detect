#!/usr/bin/env python3
"""
API集成测试
测试FastAPI服务的功能
"""
import os
import sys
import unittest
import tempfile
import subprocess
import time
import requests
from PIL import Image
import io

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class TestAPIIntegration(unittest.TestCase):
    """测试API集成"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 启动FastAPI服务
        cls.server_process = subprocess.Popen(
            [sys.executable, 'src/models/deployment/deploy_model.py'],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        )
        # 等待服务启动
        time.sleep(5)
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        # 停止FastAPI服务
        cls.server_process.terminate()
        cls.server_process.wait()
    
    def test_root_endpoint(self):
        """测试根端点"""
        response = requests.get('http://localhost:8000/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['message'], '角色识别API服务运行中')
    
    def test_health_endpoint(self):
        """测试健康检查端点"""
        response = requests.get('http://localhost:8000/health')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
    
    def test_models_endpoint(self):
        """测试模型列表端点"""
        response = requests.get('http://localhost:8000/models')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('total', data)
        self.assertIn('models', data)
    
    def test_predict_endpoint(self):
        """测试预测端点"""
        # 创建测试图像
        test_image = Image.new('RGB', (224, 224), color='red')
        image_content = io.BytesIO()
        test_image.save(image_content, format='JPEG')
        image_content.seek(0)
        
        # 发送请求
        files = {'file': ('test.jpg', image_content, 'image/jpeg')}
        data = {'model': 'default'}
        response = requests.post('http://localhost:8000/predict', files=files, data=data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('predicted_class', data)
        self.assertIn('confidence', data)
        self.assertIn('top5', data)

if __name__ == '__main__':
    unittest.main()
