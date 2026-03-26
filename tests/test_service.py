#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试API服务是否正常响应
"""

import requests
import time

def test_health():
    """测试健康检查接口"""
    try:
        response = requests.get("http://127.0.0.1:8000/api/health", timeout=10)
        print(f"健康检查响应: {response.status_code}")
        print(f"响应内容: {response.json()}")
        return True
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

def test_classify():
    """测试分类接口"""
    try:
        # 使用一个测试图片
        test_image_path = "data/train/日奈/108093272_p0_master1200.jpg"
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://127.0.0.1:8000/api/classify", files=files, timeout=30)
            print(f"分类接口响应: {response.status_code}")
            print(f"响应内容: {response.json()}")
        return True
    except Exception as e:
        print(f"分类接口测试失败: {e}")
        return False

if __name__ == "__main__":
    print("测试API服务...")
    test_health()
    test_classify()
