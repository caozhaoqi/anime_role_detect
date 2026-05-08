#!/usr/bin/env python3
"""为图片不足20张的角色采集非R18图片"""
import requests
import time
import configparser
import sys
sys.path.insert(0, 'spider_image_system/src/run')
from constants import PINYIN_MAPPING

# 需要采集的角色（图片不足20张）
LOW_COUNT_ROLES = [
    {'name': '芙丽希娅', 'current_count': 5},
    {'name': '洛茜', 'current_count': 11},
    {'name': '克萝萝', 'current_count': 12},
    {'name': '德丽莎', 'current_count': 16}
]

# API配置 - 使用端口33334
API_BASE = 'http://localhost:33334/api/v1.2.5.260305/sis'

def set_r18_mode(enable=False):
    """设置爬虫模式为非R18"""
    config_path = 'spider_image_system/src/run/config/config.ini'
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    
    if 'automatic_config' in config:
        config['automatic_config']['r18_mode'] = 'True' if enable else 'False'
        with open(config_path, 'w', encoding='utf-8') as f:
            config.write(f)
        print(f"✅ 配置文件R18模式已设置为: {enable}")
        return True
    else:
        print("❌ 配置文件格式错误")
        return False

def spider_role(role_name):
    """采集单个角色的图片"""
    pinyin = PINYIN_MAPPING.get(role_name)
    if not pinyin:
        print(f"❌ 未找到 {role_name} 的拼音映射")
        return 0
    
    url = f"{API_BASE}/spider_start/single?key_word={role_name}"
    
    print(f"🚀 开始采集 {role_name} ({pinyin})...")
    
    try:
        response = requests.post(url, timeout=300)
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                print(f"✅ {role_name} 采集任务已启动")
                return 1
            else:
                print(f"❌ {role_name} 采集失败: {result.get('msg', '未知错误')}")
                return 0
        else:
            print(f"❌ {role_name} 采集请求失败: {response.status_code}")
            return 0
    except requests.exceptions.Timeout:
        print(f"⏰ {role_name} 采集超时")
        return 0
    except Exception as e:
        print(f"❌ {role_name} 采集异常: {e}")
        return 0

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
                        print("✅ 爬虫已完成")
                        break
                    else:
                        keyword = data.get("current_keyword", "")
                        count = data.get("current_count", 0)
                        print(f"⏳ 正在采集: {keyword}, 当前进度: {count}")
                else:
                    print(f"⚠️ 状态查询失败: {status.get('msg', '未知错误')}")
        except Exception as e:
            print(f"⚠️ 检查状态异常: {e}")
        time.sleep(5)

def main():
    print("=" * 60)
    print("📷 开始采集图片不足20张的角色")
    print("=" * 60)
    
    # 设置为非R18模式
    if not set_r18_mode(False):
        print("❌ 无法设置非R18模式")
    
    # 逐个采集角色
    for role in LOW_COUNT_ROLES:
        print(f"\n📋 {role['name']}: 当前 {role['current_count']} 张")
        
        # 启动采集
        if spider_role(role['name']):
            # 等待完成
            wait_for_spider()
            time.sleep(2)  # 间隔2秒
    
    print("\n" + "=" * 60)
    print("📊 采集任务已全部提交")
    print("=" * 60)

if __name__ == '__main__':
    main()
