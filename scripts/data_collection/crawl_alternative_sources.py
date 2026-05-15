#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从其他二次元网站采集图片URL
支持: Zerochan, Yande.re, Danbooru
"""
import requests
import re
import os
import time
from urllib.parse import quote

# 需要采集的角色
ROLES = [
    {"cn_name": "姬坂乃爱", "en_name": "Himesaka", "keywords": ["himesaka noa", "姫坂乃愛"]},
    {"cn_name": "小鸟游星野", "en_name": "Hoshino", "keywords": ["hoshino (blue archive)", "小鳥遊星野"]}
]

# 输出目录
OUTPUT_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/img_url'

def crawl_zerochan(keyword, max_count=100):
    """从Zerochan采集"""
    urls = []
    page = 1
    collected = 0
    
    while collected < max_count:
        url = f"https://www.zerochan.net/{quote(keyword)}?p={page}"
        try:
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }, timeout=15)
            
            if response.status_code != 200:
                break
            
            # 提取图片URL
            matches = re.findall(r'<a href="(/full/[^"]+)"', response.text)
            for match in matches:
                if collected >= max_count:
                    break
                full_url = f"https://www.zerochan.net{match}"
                urls.append(full_url)
                collected += 1
            
            page += 1
            time.sleep(2)
            
        except Exception as e:
            print(f"  Zerochan采集失败: {e}")
            break
    
    return urls

def crawl_yandere(keyword, max_count=100):
    """从Yande.re采集"""
    urls = []
    page = 1
    collected = 0
    
    while collected < max_count:
        url = f"https://yande.re/post?tags={quote(keyword)}&page={page}"
        try:
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }, timeout=15)
            
            if response.status_code != 200:
                break
            
            # 提取图片URL
            matches = re.findall(r'<a class="directlink" href="([^"]+)"', response.text)
            for match in matches:
                if collected >= max_count:
                    break
                urls.append(match)
                collected += 1
            
            page += 1
            time.sleep(2)
            
        except Exception as e:
            print(f"  Yande.re采集失败: {e}")
            break
    
    return urls

def crawl_danbooru(keyword, max_count=100):
    """从Danbooru采集"""
    urls = []
    page = 1
    collected = 0
    
    while collected < max_count:
        url = f"https://danbooru.donmai.us/posts?tags={quote(keyword)}&page={page}"
        try:
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }, timeout=15)
            
            if response.status_code != 200:
                break
            
            # 提取图片URL
            matches = re.findall(r'original-url="([^"]+)"', response.text)
            for match in matches:
                if collected >= max_count:
                    break
                urls.append(match)
                collected += 1
            
            page += 1
            time.sleep(2)
            
        except Exception as e:
            print(f"  Danbooru采集失败: {e}")
            break
    
    return urls

def save_urls(role_name, pinyin, urls):
    """保存URL到文件"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 读取已存在的URL
    existing_urls = set()
    file_path = os.path.join(OUTPUT_DIR, f'{pinyin}_img.txt')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_urls = set(line.strip() for line in f if line.strip())
    
    # 添加新URL
    new_urls = [u for u in urls if u not in existing_urls]
    
    if new_urls:
        with open(file_path, 'a') as f:
            for url in new_urls:
                f.write(url + '\n')
    
    return len(new_urls)

def main():
    print("🚀 从其他二次元网站采集URL")
    print("=" * 60)
    
    for role in ROLES:
        cn_name = role["cn_name"]
        en_name = role["en_name"]
        keywords = role["keywords"]
        
        print(f"\n📥 采集: {cn_name} ({en_name})")
        
        all_urls = []
        
        for keyword in keywords:
            print(f"  关键词: {keyword}")
            
            # Zerochan
            print("  ├─ Zerochan...", end=" ")
            urls = crawl_zerochan(keyword, max_count=50)
            all_urls.extend(urls)
            print(f"获取 {len(urls)} 个")
            
            # Yande.re
            print("  ├─ Yande.re...", end=" ")
            urls = crawl_yandere(keyword, max_count=50)
            all_urls.extend(urls)
            print(f"获取 {len(urls)} 个")
            
            # Danbooru
            print("  └─ Danbooru...", end=" ")
            urls = crawl_danbooru(keyword, max_count=50)
            all_urls.extend(urls)
            print(f"获取 {len(urls)} 个")
        
        # 去重
        all_urls = list(set(all_urls))
        print(f"  总计: {len(all_urls)} 个URL")
        
        # 保存
        pinyin = {
            '姬坂乃爱': 'ji1ban3nai3ai4',
            '小鸟游星野': 'xiao3niao3you2xing1ye3'
        }.get(cn_name, en_name.lower())
        
        added = save_urls(en_name, pinyin, all_urls)
        print(f"  ✅ 新增 {added} 个URL")
    
    print("\n" + "=" * 60)
    print("✅ 采集完成")

if __name__ == "__main__":
    main()