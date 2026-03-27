#!/usr/bin/env python3
"""
测试SVG文件处理
"""

import requests
import os

# 测试图像路径
test_image_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train', '日奈', '日奈_1.svg')
print(f"使用测试图像: {test_image_path}")

# 确保文件存在
if not os.path.exists(test_image_path):
    print(f"测试图像不存在: {test_image_path}")
    exit(1)

# 构建请求
url = "http://localhost:8000/api/classify"

# 根据文件扩展名设置正确的Content-Type
content_type = 'image/jpeg'
if test_image_path.lower().endswith('.png'):
    content_type = 'image/png'
elif test_image_path.lower().endswith('.svg'):
    content_type = 'image/svg+xml'

print(f"Content-Type: {content_type}")

# 发送请求
try:
    with open(test_image_path, 'rb') as f:
        files = {'file': (os.path.basename(test_image_path), f, content_type)}
        data = {'model_name': 'default'}
        print("发送请求...")
        response = requests.post(url, files=files, data=data, timeout=60)
    
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
except Exception as e:
    print(f"请求异常: {e}")
