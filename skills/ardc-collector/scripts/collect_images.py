#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量采集角色图片
"""
import sys
import os
import argparse

# 添加 ardc-client 到路径
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__),
    '..', '..', 'ardc-client', 'scripts'
))

from ard_client import load_env, call_api

def collect_images(role, count=100, output_dir='./data'):
    """
    批量采集角色图片
    
    Args:
        role: 角色名
        count: 目标数量
        output_dir: 输出目录
    """
    env = load_env()
    base_url = env.get('ARDC_BASE_URL', 'http://localhost:8080')
    token = env.get('ARDC_TOKEN', '')
    
    print(f"开始采集 {role} 的图片，目标数量：{count}")
    
    # 调用采集 API
    result = call_api(base_url, token, '/api/collect', {
        'role': role,
        'count': count,
        'output_dir': output_dir,
    })
    
    print(f"采集完成：{result}")
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量采集角色图片')
    parser.add_argument('--role', required=True, help='角色名')
    parser.add_argument('--count', type=int, default=100, help='目标数量')
    parser.add_argument('--output', default='./data', help='输出目录')
    
    args = parser.parse_args()
    collect_images(args.role, args.count, args.output)
