#!/usr/bin/env python3
"""
数据收集模块单元测试
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.collection.data_collection import ImageCollector

class TestDataCollection(unittest.TestCase):
    """测试数据收集模块"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.collector = ImageCollector(output_dir=self.temp_dir)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.collector, ImageCollector)
        self.assertTrue(hasattr(self.collector, 'headers'))
        self.assertEqual(self.collector.output_dir, self.temp_dir)
    
    @patch('src.data.collection.data_collection.requests.get')
    def test_search_images(self, mock_get):
        """测试搜索图像"""
        # 模拟响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '<html><body><img src="http://example.com/test.jpg"></body></html>'
        mock_get.return_value = mock_response
        
        # 测试搜索
        results = self.collector.search_images('test', num_images=1)
        self.assertEqual(len(results), 1)
    
    @patch('src.data.collection.data_collection.requests.get')
    @patch('PIL.Image.open')
    def test_download_image(self, mock_open, mock_get):
        """测试下载图像"""
        # 模拟响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'test image data'
        mock_get.return_value = mock_response
        
        # 模拟图像打开
        mock_image = MagicMock()
        mock_open.return_value = mock_image
        mock_image.verify = MagicMock()
        
        # 测试下载
        save_path = os.path.join(self.temp_dir, 'test.jpg')
        result = self.collector.download_image('http://example.com/test.jpg', save_path)
        self.assertTrue(result)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()
