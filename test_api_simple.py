#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试API服务的脚本
"""

import requests
import time
import os

# 测试图片路径
test_image_path = "test_image.jpg"

# 确保测试图片存在
if not os.path.exists(test_image_path):
    print(f"测试图片 {test_image_path} 不存在，请先创建测试图片")
    exit(1)

# 测试API端点
api_url = "http://localhost:8000/api/classify"

# 准备请求数据
files = {
    'file': open(test_image_path, 'rb')
}
data = {
    'use_model': 'true',
    'use_attributes': 'true',
    'model_name': 'default',
    'cache_bypass': 'false'
}

print("测试API服务...")
print(f"请求URL: {api_url}")
print(f"测试图片: {test_image_path}")
print(f"图片大小: {os.path.getsize(test_image_path)} 字节")

# 发送请求
start_time = time.time()
try:
    print("发送请求...")
    response = requests.post(api_url, files=files, data=data, timeout=120)
    print(f"响应状态码: {response.status_code}")
    print(f"响应时间: {time.time() - start_time:.2f} 秒")
    print(f"响应内容: {response.json()}")
    
    # 检查响应是否包含预期的字段
    if 'detection_mode' in response.json():
        print(f"检测模式: {response.json()['detection_mode']}")
    else:
        print("响应中缺少detection_mode字段")
        
    if 'text_detections' in response.json():
        print(f"文本检测结果数量: {len(response.json()['text_detections'])}")
    else:
        print("响应中缺少text_detections字段")
        
    print("测试成功！")
except Exception as e:
    print(f"请求失败: {e}")
    print(f"请求时间: {time.time() - start_time:.2f} 秒")
    exit(1)
