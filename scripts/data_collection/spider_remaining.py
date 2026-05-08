#!/usr/bin/env python3
import requests
import time
from pathlib import Path

# 正确的API路径 - 需要包含 /sis 前缀
API_BASE = 'http://localhost:33333/api/v1.2.5.260305/sis'

def reset_spider():
    """重置爬虫状态"""
    try:
        resp = requests.post(f"{API_BASE}/spider/reset")
        return resp.json()
    except Exception as e:
        print(f"重置失败: {e}")
        return None

def spider_single_role(role_name, role_pinyin):
    """爬取单个角色的URL"""
    print(f"\n{'='*60}")
    print(f"📡 爬取角色: {role_name}")
    print(f"{'='*60}")
    
    # 重置状态
    reset_spider()
    time.sleep(2)
    
    # 启动爬虫
    try:
        resp = requests.post(f"{API_BASE}/spider_start/single?key_word={role_name}")
        print(f"HTTP状态码: {resp.status_code}")
        try:
            result = resp.json()
            print(f"启动结果: {result}")
        except:
            print(f"响应内容: {resp.text[:500]}")
        
        if resp.status_code == 200:
            print("等待爬取完成...")
            # 等待一段时间让爬虫完成
            for i in range(60):
                time.sleep(2)
                status = check_spider_status()
                if status and status.get('status') == 'idle':
                    print("✅ 爬取完成")
                    break
                elif i % 10 == 0:
                    print(f"等待中... ({i*2}秒)")
            
            # 检查URL文件是否更新
            url_file = Path(f'spider_image_system/data/img_url/{role_pinyin}_img.txt')
            if url_file.exists():
                with open(url_file) as f:
                    urls = [l.strip() for l in f if l.strip()]
                print(f"📊 {role_name} 共获取 {len(urls)} 个URL")
                return len(urls)
            else:
                print(f"❌ {role_name} URL文件未生成")
                return 0
        else:
            print(f"❌ 启动失败: HTTP {resp.status_code}")
            return 0
    except Exception as e:
        print(f"❌ 爬取异常: {e}")
        return 0

def check_spider_status():
    """检查爬虫状态"""
    try:
        resp = requests.get(f"{API_BASE}/spider/status")
        return resp.json()
    except Exception as e:
        return None

def main():
    print("=" * 70)
    print("🚀 批量采集剩余角色URL")
    print("=" * 70)
    
    # 剩余需要采集的角色
    roles = [
        ('月千夜', 'yue4qian1ye4'),   # 47张 → 需3张
        ('爱丽儿', 'ai4li4er3'),      # 41张 → 需9张
        ('小闪', 'xiao3shan3'),       # 29张 → 需21张
        ('釉壶', 'you4hu2'),          # 29张 → 需21张
        ('克萝萝', 'ke4luo2luo2'),    # 20张 → 需30张
        ('芙丽希娅', 'fu2li4xi1ya4'),  # 11张 → 需39张
    ]
    
    total_urls_added = 0
    
    for role_name, role_pinyin in roles:
        urls_added = spider_single_role(role_name, role_pinyin)
        total_urls_added += urls_added
        time.sleep(3)  # 间隔避免频繁请求
    
    print("\n" + "=" * 70)
    print(f"🎉 采集完成！共获取 {total_urls_added} 个新URL")
    print("=" * 70)

if __name__ == '__main__':
    main()
