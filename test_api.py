#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试API服务的功能
"""

import requests
import os
import sys

# 测试图片路径
test_image_path = "src/test/images/test1.jpg"

# 检查测试图片是否存在
if not os.path.exists(test_image_path):
    print(f"测试图片不存在: {test_image_path}")
    # 尝试使用其他测试图片
    test_image_path = "src/test/images/test2.jpg"
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
        # 尝试使用项目根目录下的其他图片
        for file in os.listdir('.'):
            if file.endswith('.jpg') or file.endswith('.png'):
                test_image_path = file
                print(f"使用图片: {test_image_path}")
                break
        else:
            print("没有找到测试图片")
            sys.exit(1)

print(f"使用测试图片: {test_image_path}")

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

# 发送请求
try:
    response = requests.post(api_url, files=files, data=data, timeout=60)
    print(f"响应状态码: {response.status_code}")
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
        
except Exception as e:
    print(f"请求失败: {e}")
    sys.exit(1)

print("\n测试完成!")
