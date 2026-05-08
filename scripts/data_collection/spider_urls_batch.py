#!/usr/bin/env python3
import os
import sys
import time
import requests
from pathlib import Path

sys.path.append('spider_image_system/src')

API_BASE = "http://localhost:33333/api/v1.2.5.260305/sis"

def get_status():
    """获取爬虫状态"""
    try:
        resp = requests.get(f"{API_BASE}/spider/status", timeout=10)
        data = resp.json()
        if data.get('code') == 0:
            return data['data']
        return None
    except:
        return None

def reset_spider():
    """重置爬虫状态"""
    try:
        requests.get(f"{API_BASE}/spider/reset", timeout=10)
    except:
        pass

def start_spider(keyword):
    """启动爬虫"""
    try:
        resp = requests.post(f"{API_BASE}/spider_start/single?key_word={keyword}", timeout=30)
        return resp.json()
    except Exception as e:
        return {"code": -1, "msg": str(e)}

def wait_for_completion(timeout=600):
    """等待爬虫完成"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = get_status()
        if status:
            print(f"  状态: {'运行中' if status['is_running'] else '空闲'}, "
                  f"关键词: {status.get('current_keyword', '')}, "
                  f"计数: {status.get('current_count', 0)}")
            if not status['is_running']:
                return True
        time.sleep(15)
    return False

def check_url_file(role_pinyin):
    """检查URL文件"""
    url_file = Path(f"spider_image_system/data/img_url/{role_pinyin}_img.txt")
    if url_file.exists():
        with open(url_file, 'r', encoding='utf-8') as f:
            count = len([line for line in f if line.strip()])
        return count
    return 0

def main():
    print("=" * 70)
    print("🚀 批量爬取角色URL (非R18模式)")
    print("=" * 70)
    print(f"API地址: {API_BASE}")
    print()

    roles = [
        ('洛可可', 'luo4ke4ke4'),
        ('月千夜', 'yue4qian1ye4'),
        ('爱丽儿', 'ai4li4er3'),
        ('小闪', 'xiao3shan3'),
        ('釉壶', 'you4hu2'),
        ('克萝萝', 'ke4luo2luo2'),
        ('芙丽希娅', 'fu2li4xi1ya4'),
    ]

    for role_name, role_pinyin in roles:
        print(f"\n{'='*70}")
        print(f"📡 开始爬取: {role_name} ({role_pinyin})")
        print(f"{'='*70}")

        current_count = check_url_file(role_pinyin)
        print(f"  当前URL数量: {current_count}")

        # 重置状态
        reset_spider()
        time.sleep(2)

        # 启动爬虫
        result = start_spider(role_name)
        print(f"  启动结果: {result.get('msg', result)}")

        if result.get('code') == 0:
            # 等待爬取完成
            print("  等待爬取完成...")
            success = wait_for_completion(timeout=600)
            if success:
                new_count = check_url_file(role_pinyin)
                print(f"  ✅ 爬取完成! URL数量: {new_count} (新增 {new_count - current_count})")
            else:
                print(f"  ⚠️ 等待超时")
        else:
            print(f"  ❌ 启动失败: {result.get('msg')}")

        time.sleep(3)

    print(f"\n{'='*70}")
    print("🎉 批量爬取完成!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
