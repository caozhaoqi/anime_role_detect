#!/usr/bin/env python3
"""
开启R18模式重新采集URL
注意：采集后需要过滤NSFW内容
"""
import requests
import time
from pathlib import Path

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
    print(f"📡 爬取角色: {role_name} (R18模式)")
    print(f"{'='*60}")
    
    # 获取当前URL数量
    url_file = Path(f'spider_image_system/data/img_url/{role_pinyin}_img.txt')
    old_count = 0
    if url_file.exists():
        with open(url_file) as f:
            old_count = len([l for l in f if l.strip()])
    print(f"📊 当前URL数量: {old_count}")
    
    # 重置状态
    reset_spider()
    time.sleep(2)
    
    # 启动爬虫
    try:
        resp = requests.post(f"{API_BASE}/spider_start/single?key_word={role_name}")
        print(f"HTTP状态码: {resp.status_code}")
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"启动结果: {result}")
            
            print("等待爬取完成...")
            # 等待一段时间让爬虫完成
            for i in range(90):  # 最多等待3分钟
                time.sleep(2)
                status = check_spider_status()
                if status and status.get('status') == 'idle':
                    print("✅ 爬取完成")
                    break
                elif i % 15 == 0:
                    print(f"等待中... ({i*2}秒)")
            
            # 检查URL文件是否更新
            if url_file.exists():
                with open(url_file) as f:
                    urls = [l.strip() for l in f if l.strip()]
                new_count = len(urls)
                added = new_count - old_count
                print(f"📊 {role_name}: 原{old_count}个 → 现{new_count}个 (新增{added}个)")
                return added
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
    print("🚀 R18模式批量采集URL")
    print("⚠️  注意：采集后需要过滤NSFW内容")
    print("=" * 70)
    
    # 需要补充URL的角色
    roles = [
        ('月千夜', 'yue4qian1ye4'),   # 需3张
        ('爱丽儿', 'ai4li4er3'),      # 需9张
        ('小闪', 'xiao3shan3'),       # 需21张
        ('釉壶', 'you4hu2'),          # 需21张
        ('克萝萝', 'ke4luo2luo2'),    # 需30张
        ('芙丽希娅', 'fu2li4xi1ya4'),  # 需39张
    ]
    
    total_added = 0
    
    for role_name, role_pinyin in roles:
        added = spider_single_role(role_name, role_pinyin)
        total_added += added
        time.sleep(5)  # 间隔避免频繁请求
    
    print("\n" + "=" * 70)
    print(f"🎉 R18采集完成！共新增 {total_added} 个URL")
    print("⚠️  请记得：")
    print("   1. 将r18_mode改回False")
    print("   2. 运行NSFW检测过滤敏感内容")
    print("=" * 70)

if __name__ == '__main__':
    main()
