#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试下载失败原因
"""

import requests
import sys
from urllib.parse import urlparse

def test_url(url):
    """测试单个URL"""
    try:
        response = requests.get(url, timeout=10, stream=True)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            content_length = response.headers.get('content-length', 'unknown')
            return {'status': 'success', 'code': 200, 'content_type': content_type, 'content_length': content_length}
        else:
            return {'status': 'failed', 'code': response.status_code, 'reason': f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {'status': 'failed', 'code': None, 'reason': str(e)}

def main():
    # 读取URL文件
    if len(sys.argv) < 2:
        print("Usage: python3 debug_download.py <url_file>")
        sys.exit(1)
    
    url_file = sys.argv[1]
    print(f"调试URL文件: {url_file}")
    
    with open(url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip().startswith('http')]
    
    print(f"总URL数: {len(urls)}")
    print("=" * 60)
    
    # 测试前20个URL
    results = []
    for i, url in enumerate(urls[:20], 1):
        print(f"测试 {i:2d}/{len(urls[:20])}: {url[:60]}...")
        result = test_url(url)
        results.append(result)
        print(f"  状态: {result['status']}, 原因: {result.get('reason', 'N/A')}")
    
    # 统计
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    print("=" * 60)
    print(f"成功: {success_count}/{len(results)}")
    print(f"失败: {failed_count}/{len(results)}")

if __name__ == '__main__':
    main()
