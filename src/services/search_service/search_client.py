#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索服务客户端 - 通过HTTP调用独立的搜索服务
"""

import os
import sys
import io
import requests

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.core.logging.global_logger import get_logger

logger = get_logger("search_client")

# 搜索服务配置
SEARCH_SERVICE_URL = os.environ.get('SEARCH_SERVICE_URL', 'http://localhost:8002')

class SearchServiceClient:
    """搜索服务客户端"""
    
    def __init__(self, service_url: str = None):
        self.service_url = service_url or SEARCH_SERVICE_URL
    
    def search_image(self, image_path: str, top_k: int = 10) -> dict:
        """
        搜索相似图像
        
        Args:
            image_path: 图像路径
            top_k: 返回前k个相似图像
        
        Returns:
            搜索结果
        """
        try:
            url = f"{self.service_url}/search/image"
            
            with open(image_path, 'rb') as f:
                files = {'file': f}
                params = {'top_k': top_k}
                response = requests.post(url, files=files, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result
                else:
                    logger.error(f"搜索服务返回错误: {result.get('error')}")
                    return {"success": False, "error": result.get('error')}
            else:
                logger.error(f"搜索服务请求失败: {response.status_code}")
                return {"success": False, "error": f"HTTP错误: {response.status_code}"}
        
        except Exception as e:
            logger.error(f"搜索服务调用失败: {e}")
            return {"success": False, "error": str(e)}
    
    def search_image_bytes(self, image_bytes: bytes, filename: str, top_k: int = 10) -> dict:
        """
        使用图像字节数据搜索相似图像
        
        Args:
            image_bytes: 图像字节数据
            filename: 文件名
            top_k: 返回前k个相似图像
        
        Returns:
            搜索结果
        """
        try:
            url = f"{self.service_url}/search/image"
            
            files = {'file': (filename, image_bytes, 'image/jpeg')}
            params = {'top_k': top_k}
            response = requests.post(url, files=files, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result
                else:
                    logger.error(f"搜索服务返回错误: {result.get('error')}")
                    return {"success": False, "error": result.get('error')}
            else:
                logger.error(f"搜索服务请求失败: {response.status_code}")
                return {"success": False, "error": f"HTTP错误: {response.status_code}"}
        
        except Exception as e:
            logger.error(f"搜索服务调用失败: {e}")
            return {"success": False, "error": str(e)}
    
    def build_index(self, dataset_dir: str) -> dict:
        """
        构建搜索索引
        
        Args:
            dataset_dir: 数据集目录
        
        Returns:
            构建结果
        """
        try:
            url = f"{self.service_url}/search/build-index"
            params = {'dataset_dir': dataset_dir}
            response = requests.post(url, params=params)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result
                else:
                    logger.error(f"构建索引失败: {result.get('error')}")
                    return {"success": False, "error": result.get('error')}
            else:
                logger.error(f"构建索引请求失败: {response.status_code}")
                return {"success": False, "error": f"HTTP错误: {response.status_code}"}
        
        except Exception as e:
            logger.error(f"构建索引调用失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_stats(self) -> dict:
        """
        获取搜索服务统计信息
        
        Returns:
            统计信息
        """
        try:
            url = f"{self.service_url}/search/stats"
            response = requests.get(url)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result.get('data', {})
                else:
                    logger.error(f"获取统计信息失败: {result.get('error')}")
                    return {}
            else:
                logger.error(f"获取统计信息请求失败: {response.status_code}")
                return {}
        
        except Exception as e:
            logger.error(f"获取统计信息调用失败: {e}")
            return {}
    
    def health_check(self) -> bool:
        """
        检查搜索服务健康状态
        
        Returns:
            True表示服务正常，False表示服务异常
        """
        try:
            url = f"{self.service_url}/health"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"搜索服务健康检查失败: {e}")
            return False

# 创建全局客户端实例
search_client = SearchServiceClient()

def get_search_client() -> SearchServiceClient:
    """获取搜索服务客户端实例"""
    return search_client
