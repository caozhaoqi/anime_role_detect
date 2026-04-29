#!/usr/bin/env python3
"""
批量爬取缺失角色的二次元图片
"""
import requests
import time
import urllib.parse

# 缺失角色列表
missing_roles = [
    '纳西妲',
    '可莉',
    '蕾贝',
    '迪奥娜',
    '阿洛娜',
    '普拉娜',
    '希格雯',
    '瑶瑶'
]

# API基础URL
BASE_URL = 'http://localhost:33333/api/v1.2.5.260305/sis'

def spider_single_role(keyword):
    """爬取单个角色"""
    encoded_keyword = urllib.parse.quote(keyword)
    url = f'{BASE_URL}/spider_start/single?key_word={encoded_keyword}'
    
    try:
        response = requests.post(url)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                print(f"✅ 开始爬取角色: {keyword}")
                return True
            else:
                print(f"❌ 爬取失败 [{keyword}]: {data.get('msg')}")
                return False
        else:
            print(f"❌ 请求失败 [{keyword}]: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 爬取异常 [{keyword}]: {e}")
        return False

def download_all_images():
    """下载所有已爬取的图片"""
    url = f'{BASE_URL}/download_all_image/start/'
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == 0:
                print("✅ 开始下载图片")
                return True
            else:
                print(f"❌ 下载失败: {data.get('msg')}")
                return False
        else:
            print(f"❌ 请求失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 下载异常: {e}")
        return False

def main():
    print("=== 开始批量爬取缺失角色 ===")
    print(f"待爬取角色数: {len(missing_roles)}")
    
    for i, role in enumerate(missing_roles):
        print(f"\n[{i+1}/{len(missing_roles)}] 处理角色: {role}")
        
        # 爬取角色图片URL
        if spider_single_role(role):
            # 等待爬取完成
            time.sleep(15)
            
            # 下载图片
            download_all_images()
            time.sleep(10)
    
    print("\n=== 批量爬取完成 ===")

if __name__ == "__main__":
    main()
