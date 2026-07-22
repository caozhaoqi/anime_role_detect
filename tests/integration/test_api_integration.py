#!/usr/bin/env python3
"""
API集成测试
测试FastAPI服务的功能

需要服务运行中才能执行，否则自动跳过。
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


def _service_available(url: str, timeout: float = 2.0) -> bool:
    """检查服务是否可达"""
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code < 500
    except Exception:
        return False


class TestAPIIntegration(unittest.TestCase):
    """测试API集成"""

    BASE_URL = "http://localhost:8080"
    MODEL_SERVICE_URL = "http://localhost:8000"
    API_SERVICE_URL = "http://localhost:8001"

    @classmethod
    def setUpClass(cls):
        """检查服务是否运行，未运行则跳过需要服务的测试"""
        cls.gateway_ok = _service_available(f"{cls.BASE_URL}/api/services")
        cls.model_ok = _service_available(f"{cls.MODEL_SERVICE_URL}/api/health")
        cls.api_ok = _service_available(f"{cls.API_SERVICE_URL}/api/health")

    def test_gateway_health(self):
        """测试API网关健康检查"""
        if not self.gateway_ok:
            self.skipTest("API Gateway (8080) 未运行")
        response = requests.get(f"{self.BASE_URL}/api/services")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("services", data)

    def test_model_service_health(self):
        """测试模型服务健康检查"""
        if not self.model_ok:
            self.skipTest("Model Service (8000) 未运行")
        response = requests.get(f"{self.MODEL_SERVICE_URL}/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_core_api_health(self):
        """测试核心API健康检查"""
        if not self.api_ok:
            self.skipTest("API Service (8001) 未运行")
        response = requests.get(f"{self.API_SERVICE_URL}/api/health")
        self.assertEqual(response.status_code, 200)

    def test_predict_endpoint(self):
        """测试预测端点"""
        if not self.model_ok:
            self.skipTest("Model Service (8000) 未运行，跳过 predict 测试")
        # 创建测试图像
        test_image = Image.new("RGB", (224, 224), color="red")
        image_content = io.BytesIO()
        test_image.save(image_content, format="JPEG")
        image_content.seek(0)

        # 发送请求到模型服务
        files = {"file": ("test.jpg", image_content, "image/jpeg")}
        response = requests.post(
            f"{self.MODEL_SERVICE_URL}/api/model/predict", files=files
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("role", data)


if __name__ == "__main__":
    unittest.main()
