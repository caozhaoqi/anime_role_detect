#!/usr/bin/env python3
"""ARD Client - Anime Role Detect 客户端模块"""

import requests
import json
import os
from typing import Dict, Any, Optional

class ARDClient:
    """ARD 服务客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def detect_role(self, image_path: str, model_name: str = "default") -> Dict[str, Any]:
        """
        检测图片中的动漫角色
        
        Args:
            image_path: 图片文件路径
            model_name: 模型名称
        
        Returns:
            检测结果字典
        """
        url = f"{self.base_url}/api/detect"
        
        try:
            with open(image_path, "rb") as f:
                files = {"image": f}
                data = {"model": model_name}
                response = self.session.post(url, files=files, data=data)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_models(self) -> Dict[str, Any]:
        """获取可用模型列表"""
        url = f"{self.base_url}/api/models"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """检查服务健康状态"""
        url = f"{self.base_url}/health"
        
        try:
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

if __name__ == "__main__":
    client = ARDClient()
    
    if client.health_check():
        print("ARD 服务运行正常")
        models = client.get_models()
        print(f"可用模型: {models}")
    else:
        print("ARD 服务不可用")
