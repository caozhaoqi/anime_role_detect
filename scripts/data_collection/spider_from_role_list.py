#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据角色名单文件批量采集图片URL
"""

import os
import requests
import time
import urllib.parse

API_BASE = "http://localhost:33333/api/v1.2.5.260305/sis"
ROLE_FILE = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt"

def wait_for_spider():
    """等待爬虫完成"""
    print("  等待爬虫完成...")
    max_wait = 180  # 最多等待3分钟
    wait_time = 0
    while wait_time < max_wait:
        try:
            status = requests.get(f"{API_BASE}/spider/status", timeout=10)
            data = status.json()
            if data.get('code') == 0 and not data.get('data', {}).get('is_running', True):
                print("  ✅ 爬虫完成")
                return True
            time.sleep(3)
            wait_time += 3
        except Exception as e:
            print(f"  ❌ 状态查询失败: {str(e)}")
            return False
    print("  ⚠️ 等待超时")
    return False

def spider_single_role(role_name):
    """调用爬虫API采集角色URL"""
    encoded_name = urllib.parse.quote(role_name)
    url = f"{API_BASE}/spider_start/single?key_word={encoded_name}"
    try:
        response = requests.post(url, timeout=30)
        result = response.json()
        if result.get('code') == 0:
            print(f"  🕷️ 启动采集: {result.get('msg')}")
            return wait_for_spider()
        else:
            print(f"  ❌ 采集失败: {result.get('msg')}")
            return False
    except Exception as e:
        print(f"  ❌ 采集异常: {str(e)}")
        return False

def main():
    print("=" * 70)
    print("🚀 根据角色名单批量采集图片URL")
    print(f"角色文件: {ROLE_FILE}")
    print(f"API: {API_BASE}")
    print("=" * 70)
    
    # 检查爬虫服务
    try:
        response = requests.get(f"{API_BASE}/spider/status", timeout=5)
        print("✅ 爬虫服务连接成功")
    except Exception as e:
        print(f"❌ 爬虫服务未运行: {str(e)}")
        print("请先启动爬虫服务:")
        print("cd spider_image_system && python3 -m src.run.sis_main_process")
        return
    
    # 读取角色名单
    if not os.path.exists(ROLE_FILE):
        print(f"❌ 角色文件不存在: {ROLE_FILE}")
        return
    
    with open(ROLE_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"\n📋 共读取到 {len(lines)} 个角色")
    
    success_count = 0
    fail_count = 0
    
    for i, line in enumerate(lines, 1):
        parts = line.split()
        role_name = parts[0]
        print(f"\n[{i}/{len(lines)}] {role_name}")
        
        # 调用爬虫API
        if spider_single_role(role_name):
            success_count += 1
        else:
            fail_count += 1
        
        # 添加间隔，避免请求过快
        time.sleep(2)
    
    print("\n" + "=" * 70)
    print(f"采集完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    print("=" * 70)

if __name__ == '__main__':
    main()
