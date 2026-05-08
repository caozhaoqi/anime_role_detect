#!/usr/bin/env python3
import requests
import time

API_BASE = "http://localhost:33333/api/v1.2.5.260305/sis"

def check_status():
    """检查爬虫状态"""
    try:
        response = requests.get(f"{API_BASE}/spider/status", timeout=10)
        data = response.json()
        return data['data']['is_running'], data['data']['current_keyword']
    except Exception as e:
        print(f"⚠️ 检查状态失败: {e}")
        return False, ""

def spider_role(role_name):
    """爬取单个角色的URL"""
    try:
        response = requests.post(f"{API_BASE}/spider_start/single?key_word={role_name}", timeout=30)
        result = response.json()
        return result.get('code') == 0
    except Exception as e:
        print(f"❌ 爬取 {role_name} 失败: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 批量爬取角色URL (非R18模式)")
    print("=" * 60)
    
    roles = [
        '洛可可',    # URL缺失
        '月千夜',    # 需要补充
        '爱丽儿',    # 需要补充
        '小闪',      # 需要补充
        '釉壶',      # 需要补充
        '克萝萝',    # 需要补充
        '芙丽希娅',  # 需要补充
    ]
    
    for role in roles:
        # 等待前一个任务完成
        is_running, keyword = check_status()
        while is_running:
            print(f"⏳ 等待 {keyword} 完成...")
            time.sleep(10)
            is_running, keyword = check_status()
        
        # 开始爬取
        print(f"\n📡 开始爬取: {role}")
        success = spider_role(role)
        if success:
            print(f"✅ 任务已提交")
        else:
            print(f"❌ 任务提交失败")
        
        # 等待启动
        time.sleep(5)
    
    # 等待最后一个任务完成
    print("\n⏳ 等待最后一个任务完成...")
    is_running, keyword = check_status()
    while is_running:
        print(f"   {keyword} 正在爬取中...")
        time.sleep(10)
        is_running, keyword = check_status()
    
    print("\n" + "=" * 60)
    print("🎉 所有爬取任务完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
