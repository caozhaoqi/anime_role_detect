#!/usr/bin/env python3
"""仅采集URL（不下载图片）"""
import requests
import time
import sys
sys.path.insert(0, 'spider_image_system/src/run')
from constants import PINYIN_MAPPING

# 需要采集的角色
LOW_COUNT_ROLES = [
    {'name': '芙丽希娅', 'current_count': 5},
    {'name': '洛茜', 'current_count': 11},
    {'name': '克萝萝', 'current_count': 12},
    {'name': '德丽莎', 'current_count': 16}
]

# API配置
API_BASE = 'http://localhost:33334/api/v1.2.5.260305/sis'

def spider_url(role_name):
    """采集单个角色的URL"""
    pinyin = PINYIN_MAPPING.get(role_name)
    if not pinyin:
        print(f"❌ 未找到 {role_name} 的拼音映射")
        return False
    
    url = f"{API_BASE}/spider_start/single?key_word={role_name}"
    
    print(f"🚀 开始采集 {role_name} ({pinyin}) 的URL...")
    
    try:
        response = requests.post(url, timeout=300)
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                print(f"✅ {role_name} URL采集任务已启动")
                return True
            else:
                print(f"❌ {role_name} URL采集失败: {result.get('msg', '未知错误')}")
                return False
        else:
            print(f"❌ {role_name} URL采集请求失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {role_name} URL采集异常: {e}")
        return False

def wait_for_spider():
    """等待爬虫完成"""
    while True:
        try:
            response = requests.get(f"{API_BASE}/spider/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                if status.get("code") == 0:
                    data = status.get("data", {})
                    if data.get("is_running") == False:
                        print("✅ URL采集完成")
                        break
                    else:
                        keyword = data.get("current_keyword", "")
                        count = data.get("current_count", 0)
                        print(f"⏳ 正在采集: {keyword}, 当前进度: {count}")
        except Exception as e:
            print(f"⚠️ 检查状态异常: {e}")
        time.sleep(5)

def check_url_count():
    """检查采集后的URL数量"""
    print("\n📊 URL采集结果统计:")
    for role in LOW_COUNT_ROLES:
        pinyin = PINYIN_MAPPING.get(role['name'])
        if pinyin:
            url_file = f'spider_image_system/data/href_url/{pinyin}_url.txt'
            if os.path.exists(url_file):
                with open(url_file, 'r', encoding='utf-8') as f:
                    urls = [line.strip() for line in f if line.strip()]
                    urls = list(set(urls))  # 去重
                    print(f"   {role['name']}: {len(urls)} 个URL")
            else:
                print(f"   {role['name']}: 无URL文件")

def main():
    global os
    import os
    
    print("=" * 60)
    print("🔗 开始采集URL")
    print("=" * 60)
    
    # 逐个采集角色URL
    for role in LOW_COUNT_ROLES:
        print(f"\n📋 {role['name']}: 当前 {role['current_count']} 张图片")
        
        if spider_url(role['name']):
            wait_for_spider()
            time.sleep(2)
    
    # 检查结果
    check_url_count()
    
    print("\n" + "=" * 60)
    print("📊 URL采集任务完成")
    print("=" * 60)

if __name__ == '__main__':
    main()
