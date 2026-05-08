#!/usr/bin/env python3
import requests
import time

API_BASE = "http://localhost:33333/api/v1.2.5.260305/sis"

def spider_single_role(role_name):
    """调用API爬取单个角色的URL"""
    url = f"{API_BASE}/spider_start/single?key_word={role_name}"
    try:
        response = requests.post(url, timeout=30)
        result = response.json()
        if result.get('code') == 200:
            print(f"✅ 开始爬取 {role_name}: {result.get('msg')}")
            return True
        else:
            print(f"❌ 爬取 {role_name} 失败: {result.get('msg')}")
            return False
    except Exception as e:
        print(f"❌ 爬取 {role_name} 异常: {str(e)}")
        return False

def check_spider_status():
    """检查爬虫状态"""
    url = f"{API_BASE}/spider/status"
    try:
        response = requests.get(url, timeout=10)
        result = response.json()
        if result.get('code') == 200:
            data = result.get('data', {})
            return data.get('is_running', False), data.get('current_keyword', '')
        return False, ''
    except Exception as e:
        print(f"⚠️ 检查状态异常: {str(e)}")
        return False, ''

def wait_for_spider():
    """等待当前爬虫任务完成"""
    while True:
        is_running, keyword = check_spider_status()
        if not is_running:
            break
        print(f"⏳ 正在爬取 {keyword}，等待完成...")
        time.sleep(10)
    print("✅ 当前爬虫任务完成")

def main():
    print("=" * 60)
    print("🚀 调用爬虫API为角色补充URL")
    print("=" * 60)
    
    # 需要爬取URL的角色
    roles_to_spider = [
        '洛可可',    # URL缺失
        '月千夜',    # 需要3张
        '爱丽儿',    # 需要9张
        '小闪',      # 需要21张
        '釉壶',      # 需要21张
        '克萝萝',    # 需要30张
        '芙丽希娅',  # 需要39张
    ]
    
    for role in roles_to_spider:
        # 等待前一个任务完成
        wait_for_spider()
        
        # 开始爬取
        success = spider_single_role(role)
        if success:
            # 等待一段时间让爬虫启动
            time.sleep(5)
    
    # 等待最后一个任务完成
    wait_for_spider()
    
    print("\n" + "=" * 60)
    print("🎉 URL爬取任务全部完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
