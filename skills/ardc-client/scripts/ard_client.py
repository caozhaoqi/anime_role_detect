#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anime Role Detect 核心客户端模块
"""
import os
import sys
import requests
from pathlib import Path

def load_env(env_path=None):
    """
    加载 .env 环境配置
    
    查找优先级：
    1. 显式 env_path 参数
    2. ARDC_PROJECT_ENV_FILE 环境变量
    3. <cwd>/.env.<ARDC_ENV> → <cwd>/config/.env.<ARDC_ENV>
    4. <cwd>/.env → <cwd>/config/.env
    """
    env_name = os.environ.get('ARDC_ENV', '')
    project_env_file = os.environ.get('ARDC_PROJECT_ENV_FILE', '')
    
    # 初始化环境变量
    env = {
        'ARDC_BASE_URL': os.environ.get('ARDC_BASE_URL', ''),
        'ARDC_API_KEY': os.environ.get('ARDC_API_KEY', ''),
        'ARDC_TOKEN': os.environ.get('ARDC_TOKEN', ''),
    }
    
    # 构建查找路径
    env_paths = []
    if env_path:
        env_paths.append(env_path)
    elif project_env_file:
        env_paths.append(project_env_file)
    else:
        if env_name:
            env_paths.append(f'.env.{env_name}')
            env_paths.append(f'config/.env.{env_name}')
        env_paths.append('.env')
        env_paths.append('config/.env')
    
    # 按优先级查找并加载
    for candidate in env_paths:
        if os.path.exists(candidate):
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        k, v = line.split('=', 1)
                        env[k.strip()] = v.strip().strip('"').strip("'")
            print(f'[ard_client] env loaded from {os.path.abspath(candidate)}', file=sys.stderr)
            break
    
    return env

def call_api(base_url, token, endpoint, body=None):
    """
    统一 API 调用接口
    
    Args:
        base_url: 服务地址
        token: API Token
        endpoint: API 端点
        body: 请求体
    
    Returns:
        JSON 响应
    """
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
    
    try:
        resp = requests.post(
            f"{base_url}{endpoint}",
            json=body or {},
            headers=headers,
            timeout=(10, 60),
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API {endpoint} 调用失败：{e}")

def classify_image(base_url, token, image_path, model='efficientnet_b3'):
    """
    图片分类 API 调用
    
    Args:
        image_path: 图片路径
        model: 模型名称
    
    Returns:
        分类结果
    """
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'model': model}
        
        headers = {'Authorization': f'Bearer {token}'}
        
        resp = requests.post(
            f"{base_url}/api/classify",
            files=files,
            data=data,
            headers=headers,
        )
        
        return resp.json()