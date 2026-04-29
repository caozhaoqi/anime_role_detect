#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试维基百科API
"""

import requests

def test_wikipedia_api():
    """
    测试维基百科API
    """
    # 测试中文维基百科
    print("=" * 80)
    print("测试中文维基百科API")
    print("=" * 80)
    
    zh_api_url = "https://zh.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": "可莉",
        "format": "json",
        "exintro": True,
        "explaintext": True
    }
    
    try:
        response = requests.get(zh_api_url, params=params, timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应长度: {len(response.text)}")
        print(f"响应前500字符: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"JSON数据: {data}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试英文维基百科
    print("\n" + "=" * 80)
    print("测试英文维基百科API")
    print("=" * 80)
    
    en_api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": "Klee",
        "format": "json",
        "exintro": True,
        "explaintext": True
    }
    
    try:
        response = requests.get(en_api_url, params=params, timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应长度: {len(response.text)}")
        print(f"响应前500字符: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"JSON数据: {data}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    test_wikipedia_api()
