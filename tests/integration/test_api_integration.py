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
    
    BASE_URL = 'http://localhost:8080'
    
    def test_gateway_health(self):
        """测试API网关健康检查"""
        response = requests.get(f'{self.BASE_URL}/api/services')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('services', data)
    
    def test_model_service_health(self):
        """测试模型服务健康检查"""
        response = requests.get('http://localhost:8004/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
    
    def test_core_api_health(self):
        """测试核心API健康检查"""
        response = requests.get('http://localhost:8001/api/health')
        self.assertEqual(response.status_code, 200)
    
    def test_predict_endpoint(self):
        """测试预测端点"""
        # 创建测试图像
        test_image = Image.new('RGB', (224, 224), color='red')
        image_content = io.BytesIO()
        test_image.save(image_content, format='JPEG')
        image_content.seek(0)
        
        # 发送请求到模型服务
        files = {'file': ('test.jpg', image_content, 'image/jpeg')}
        response = requests.post('http://localhost:8004/api/model/predict', files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('role', data)

if __name__ == '__main__':
    unittest.main()
