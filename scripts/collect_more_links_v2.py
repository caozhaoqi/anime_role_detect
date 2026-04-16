#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据auto_spider_img中的角色名抓取更多图片链接
使用项目现有的spider_image_system架构
"""

import os
import time
import random
import json
from pathlib import Path

# 配置参数
AUTO_SPIDER_DIR = "./auto_spider_img"
SPIDER_DATA_DIR = "./spider_image_system/data"
HREF_URL_DIR = os.path.join(SPIDER_DATA_DIR, "href_url")
IMG_URL_DIR = os.path.join(SPIDER_DATA_DIR, "img_url")

# 支持的数据源
SEARCH_CONFIG = {
    "base_url": "https://sd.vv50.de",
    "search_path": "/search?q={}",
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1"
    }
}

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

def generate_role_id(keyword):
    """生成角色ID"""
    # 简单的拼音转换（这里使用简化版本）
    pinyin_map = {
        '琴': 'qin',
        '安柏': 'anbai',
        '丽莎': 'lisha',
        '芭芭拉': 'babaala',
        '可莉': 'keli',
        '诺艾尔': 'nuoaiier',
        '菲谢尔': 'feixieer',
        '砂糖': 'shatang',
        '莫娜': 'mona',
        '迪奥娜': 'diiona',
        '罗莎莉亚': 'luoshalia',
        '优菈': 'youlia',
        '埃洛伊': 'ailuoyi',
        '闲云': 'xianyun',
        '瑶瑶': 'yaoyao',
        '夜兰': 'yelan',
        '申鹤': 'shenhe',
        '云堇': 'yunjin',
        '北斗': 'beidou',
        '凝光': 'ningguang'
    }
    
    # 如果在映射中找到，使用映射值
    if keyword in pinyin_map:
        return pinyin_map[keyword]
    
    # 否则使用原关键词的小写形式
    return keyword.lower().replace(' ', '').replace('·', '').replace('之', '')

def create_spider_config(keyword):
    """创建爬虫配置"""
    role_id = generate_role_id(keyword)
    
    # 创建配置目录
    config_dir = os.path.join(SPIDER_DATA_DIR, "config")
    os.makedirs(config_dir, exist_ok=True)
    
    # 创建配置文件
    config = {
        "keyword": keyword,
        "role_id": role_id,
        "search_url": SEARCH_CONFIG["base_url"] + SEARCH_CONFIG["search_path"].format(keyword),
        "headers": SEARCH_CONFIG["headers"],
        "max_images": 100,
        "output_dir": IMG_URL_DIR
    }
    
    config_file = os.path.join(config_dir, f"{role_id}_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    return config

def use_existing_spider_system():
    """使用现有的spider_image_system"""
    print("使用现有的spider_image_system进行爬虫...")
    
    # 检查spider_image_system是否存在
    if not os.path.exists("./spider_image_system"):
        print("错误: spider_image_system 目录不存在")
        return False
    
    # 检查是否有运行脚本
    run_script = "./spider_image_system/src/run/sh/run.sh"
    if os.path.exists(run_script):
        print(f"发现运行脚本: {run_script}")
        print("请手动运行spider_image_system来抓取更多链接")
        print("步骤:")
        print("1. cd spider_image_system")
        print("2. ./src/run/sh/run.sh")
        print("3. 在界面中输入角色名进行搜索")
    else:
        print("未找到运行脚本")
    
    return True

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
    
    print(f"找到 {len(keywords)} 个角色关键词")
    
    # 显示前20个关键词
    print("\n前20个角色关键词:")
    for i, keyword in enumerate(keywords[:20], 1):
        print(f"  {i}. {keyword}")
    
    # 使用现有的spider_image_system
    use_existing_spider_system()
    
    # 创建配置文件
    print("\n为每个角色创建爬虫配置...")
    for keyword in keywords:
        config = create_spider_config(keyword)
        print(f"  创建配置: {keyword} -> {config['role_id']}")
    
    print("\n" + "=" * 80)
    print("配置创建完成")
    print("=" * 80)
    print("请使用spider_image_system来抓取更多链接")
    print("或使用以下命令启动爬虫:")
    print("cd spider_image_system && ./src/run/sh/run.sh")
    print("=" * 80)

if __name__ == "__main__":
    main()
