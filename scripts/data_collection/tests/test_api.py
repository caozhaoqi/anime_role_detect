#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试API可用性
"""

import requests
import json

def test_waifu_api():
    """测试waifu.pics API"""
    print("测试waifu.pics API...")
    try:
        # 测试基本API
        url = "https://waifu.pics/api/sfw"
        response = requests.get(url, timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"成功: {json.dumps(data, indent=2)}")
        else:
            print(f"失败: {response.text}")
    except Exception as e:
        print(f"错误: {e}")

def test_sdvv50():
    """测试sd.vv50.de"""
    print("\n测试sd.vv50.de...")
    try:
        url = "https://sd.vv50.de/search?q=arona"
        response = requests.get(url, timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应长度: {len(response.text)} 字符")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    test_waifu_api()
    test_sdvv50()
