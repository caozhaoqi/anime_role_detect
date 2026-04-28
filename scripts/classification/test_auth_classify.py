#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试带认证的图片分类
"""

import os
import requests
import json

# 获取认证token
def get_token():
    login_url = "http://localhost:8001/api/auth/login"
    data = {
        'username': 'admin',
        'password': 'admin123'
    }
    response = requests.post(login_url, data=data)
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            return result.get('data', {}).get('access_token')
    return None

# 测试单张图片分类
def test_classify(image_path, token):
    api_url = "http://localhost:8001/api/classify"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data = {
                'model_name': 'resnet18_loli8',
                'use_model': True,
                'use_attributes': False,
                'multi_role': False
            }
            headers = {
                'Authorization': f'Bearer {token}'
            }
            
            response = requests.post(api_url, files=files, data=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"成功: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return result
            else:
                print(f"失败: {response.status_code}")
                print(f"响应: {response.text}")
                return None
    except Exception as e:
        print(f"错误: {e}")
        return None

if __name__ == '__main__':
    # 获取token
    token = get_token()
    if not token:
        print("获取token失败")
        exit(1)
    
    print(f"Token: {token[:50]}...")
    
    # 测试分类
    image_dir = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/role_images/可莉"
    files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if files:
        image_path = os.path.join(image_dir, files[0])
        print(f"测试图片: {image_path}")
        result = test_classify(image_path, token)
    else:
        print("没有找到图片文件")