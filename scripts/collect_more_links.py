#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据auto_spider_img中的角色名抓取更多图片链接
使用项目现有的spider_image_system架构
"""

import os
import time
import random
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# 配置参数
AUTO_SPIDER_DIR = "./auto_spider_img"
SPIDER_DATA_DIR = "./spider_image_system/data"
HREF_URL_DIR = os.path.join(SPIDER_DATA_DIR, "href_url")
IMG_URL_DIR = os.path.join(SPIDER_DATA_DIR, "img_url")

# 支持的数据源
SEARCH_ENGINES = [
    {
        "name": "sd.vv50",
        "search_url": "https://sd.vv50.de/search?q={}",
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    }
]

def load_keywords_from_file(file_path):
    """从文件中加载关键词"""
    keywords = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                keyword = line.strip()
                if keyword:
                    keywords.append(keyword)
    return keywords

def load_all_keywords():
    """加载所有关键词文件中的角色名"""
    all_keywords = set()
    
    # 遍历auto_spider_img目录中的所有txt文件
    for filename in os.listdir(AUTO_SPIDER_DIR):
        if filename.endswith('.txt'):
            file_path = os.path.join(AUTO_SPIDER_DIR, filename)
            keywords = load_keywords_from_file(file_path)
            all_keywords.update(keywords)
    
    # 过滤空字符串
    all_keywords = [kw for kw in all_keywords if kw]
    
    print(f"共加载 {len(all_keywords)} 个角色关键词")
    return all_keywords

def search_images(keyword, engine):
    """搜索图片并提取链接"""
    try:
        # 构建搜索URL
        search_url = engine["search_url"].format(keyword)
        
        # 发送请求
        response = requests.get(
            search_url,
            headers=engine["headers"],
            timeout=30
        )
        
        if response.status_code == 200:
            # 解析页面
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取图片链接
            img_links = []
            
            # 查找所有img标签
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src') or img.get('data-original')
                if src:
                    # 确保是完整URL
                    if not src.startswith('http'):
                        if src.startswith('//'):
                            src = 'https:' + src
                        else:
                            continue
                    # 过滤非图片链接
                    if any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        img_links.append(src)
            
            return img_links
        else:
            print(f"搜索失败: {response.status_code} for {keyword}")
            return []
            
    except Exception as e:
        print(f"搜索错误: {e} for {keyword}")
        return []

def process_role(keyword):
    """处理单个角色"""
    print(f"\n处理角色: {keyword}")
    
    # 生成角色ID（拼音转换）
    role_id = keyword.lower().replace(' ', '').replace('·', '').replace('之', '')
    
    # 创建输出文件
    href_file = os.path.join(HREF_URL_DIR, f"{role_id}_url.txt")
    img_file = os.path.join(IMG_URL_DIR, f"{role_id}_img.txt")
    
    # 已存在的链接
    existing_links = set()
    if os.path.exists(img_file):
        with open(img_file, 'r', encoding='utf-8') as f:
            existing_links = set(line.strip() for line in f if line.strip())
    
    print(f"  已存在 {len(existing_links)} 个链接")
    
    # 搜索图片
    all_img_links = []
    for engine in SEARCH_ENGINES:
        print(f"  使用 {engine['name']} 搜索...")
        img_links = search_images(keyword, engine)
        all_img_links.extend(img_links)
        time.sleep(random.uniform(1, 3))  # 避免请求过快
    
    # 去重
    new_links = set(all_img_links) - existing_links
    
    print(f"  新找到 {len(new_links)} 个链接")
    
    # 保存新链接
    if new_links:
        # 保存图片链接
        with open(img_file, 'a', encoding='utf-8') as f:
            for link in new_links:
                f.write(link + '\n')
        
        print(f"  已保存到 {img_file}")
    
    return len(new_links)

def main():
    """主函数"""
    print("=" * 80)
    print("根据auto_spider_img中的角色名抓取更多图片链接")
    print("=" * 80)
    
    # 确保目录存在
    os.makedirs(HREF_URL_DIR, exist_ok=True)
    os.makedirs(IMG_URL_DIR, exist_ok=True)
    
    # 加载所有关键词
    keywords = load_all_keywords()
    
    if not keywords:
        print("没有找到关键词")
        return
    
    print(f"开始抓取 {len(keywords)} 个角色的图片链接...")
    
    # 批量处理
    total_new_links = 0
    
    # 使用线程池加速
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_role, keyword): keyword for keyword in keywords}
        
        for future in futures:
            keyword = futures[future]
            try:
                new_links = future.result()
                total_new_links += new_links
                print(f"  {keyword}: +{new_links} 个链接")
            except Exception as e:
                print(f"  {keyword}: 处理失败 - {e}")
    
    print("\n" + "=" * 80)
    print("抓取完成")
    print("=" * 80)
    print(f"总角色数: {len(keywords)}")
    print(f"新增链接数: {total_new_links}")
    print("=" * 80)

if __name__ == "__main__":
    main()
