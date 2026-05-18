#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 ardc-client 环境加载
"""
import sys
import os

# 添加 ardc-client 到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(
    project_root,
    'skills', 'ardc-client', 'scripts'
))

from ard_client import load_env

def test_load_env():
    """测试环境加载"""
    print("测试环境加载...")
    print("=" * 60)
    
    env = load_env()
    
    print(f"\n加载的环境配置:")
    for key, value in env.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✓ 环境加载测试完成")

if __name__ == '__main__':
    test_load_env()
