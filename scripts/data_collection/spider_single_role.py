#!/usr/bin/env python3
import requests
import time
from pathlib import Path

API_BASE = "http://localhost:33333/api/v1.2.5.260305/sis"

def reset_and_wait():
    """重置爬虫状态并等待"""
    resp = requests.post(f"{API_BASE}/spider/reset")
    print(f"重置: {resp.json()}")
    time.sleep(3)

def wait_for_completion(timeout=600):
    """等待爬虫完成"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = requests.get(f"{API_BASE}/spider/status")
        data = status.json()
        if data.get('code') == 0:
            is_running = data['data']['is_running']
            keyword = data['data'].get('current_keyword', '')
            count = data['data'].get('current_count', 0)
            print(f"  状态: {'运行中' if is_running else '空闲'}, 关键词: {keyword}, 计数: {count}")
            if not is_running:
                return True
        time.sleep(10)
    return False

def spider_role(role_name):
    """爬取单个角色"""
    print(f"\n{'='*60}")
    print(f"📡 爬取角色: {role_name}")
    print(f"{'='*60}")

    # 重置状态
    reset_and_wait()

    # 启动爬虫
    resp = requests.post(f"{API_BASE}/spider_start/single?key_word={role_name}")
    print(f"启动: {resp.json()}")

    # 等待完成
    if resp.json().get('code') == 0:
        print("等待爬取完成...")
        success = wait_for_completion()
        if success:
            print("✅ 爬取完成")
        else:
            print("⚠️ 等待超时")
    else:
        print(f"❌ 启动失败")

    time.sleep(2)

def main():
    # 按URL数量从少到多排序
    roles_to_spider = [
        ('釉壶', 'you4hu2'),
        ('芙丽希娅', 'fu2li4xi1ya4'),
        ('克萝萝', 'ke4luo2luo2'),
        ('小闪', 'xiao3shan3'),
        ('爱丽儿', 'ai4li4er3'),
    ]

    for role_name, role_pinyin in roles_to_spider:
        # 检查当前URL数量
        url_file = Path(f"spider_image_system/data/img_url/{role_pinyin}_img.txt")
        current_count = 0
        if url_file.exists():
            with open(url_file, 'r', encoding='utf-8') as f:
                current_count = len([line for line in f if line.strip()])

        print(f"\n当前 {role_name} URL数量: {current_count}")

        if current_count < 50:
            spider_role(role_name)

            # 检查爬取后的URL数量
            with open(url_file, 'r', encoding='utf-8') as f:
                new_count = len([line for line in f if line.strip()])
            print(f"爬取后 {role_name} URL数量: {new_count} (新增 {new_count - current_count})")
        else:
            print(f"✅ {role_name} URL已足够 ({current_count})")

    print(f"\n{'='*60}")
    print("🎉 所有需要爬取的角色完成！")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
