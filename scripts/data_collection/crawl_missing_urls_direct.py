#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接调用爬虫代码采集数据不足角色的URL
"""
import os
import sys
import time
from pathlib import Path

# 添加爬虫系统路径
sys.path.append(str(Path(__file__).parent.parent.parent / 'spider_image_system/src'))

from ui_event.get_url import spider_artworks_url
from run import constants

# 需要采集URL的角色
MISSING_ROLES = [
    {"cn_name": "姬坂乃爱", "en_name": "Himesaka", "needed": 79},
    {"cn_name": "科谢尼娅", "en_name": "Koshenia", "needed": 28},
    {"cn_name": "克拉拉", "en_name": "Clara", "needed": 13},
    {"cn_name": "小鸟游星野", "en_name": "Hoshino", "needed": 7}
]

def main():
    print("📡 开始直接采集角色URL")
    print("=" * 60)
    
    # 设置爬虫模式
    constants.SpiderConfig.spider_mode = 'manual'
    constants.SpiderConfig.max_urls_per_keyword = 200  # 每个角色最多采集200个URL
    
    for role in MISSING_ROLES:
        cn_name = role["cn_name"]
        en_name = role["en_name"]
        needed = role["needed"]
        
        print(f"\n📥 准备采集: {cn_name} ({en_name})")
        print(f"   需要补充: {needed} 张图片")
        
        # 重置爬虫状态
        constants.SpiderConfig.stop_spider_url_flag = False
        
        # 执行爬虫
        try:
            print(f"🚀 开始爬取...")
            spider_artworks_url(None, cn_name)
            print(f"✅ {cn_name} 采集完成")
        except Exception as e:
            print(f"❌ {cn_name} 采集失败: {e}")
        
        # 等待3秒再处理下一个
        time.sleep(3)
    
    print("\n" + "=" * 60)
    print("✅ URL采集任务完成")
    
    # 统计结果
    print("\n📊 采集结果统计:")
    for role in MISSING_ROLES:
        cn_name = role["cn_name"]
        pinyin = constants.get_pinyin(cn_name)
        url_file = Path(__file__).parent.parent.parent / 'spider_image_system' / 'data' / 'img_url' / f'{pinyin}_img.txt'
        
        if url_file.exists():
            with open(url_file, 'r', encoding='utf-8') as f:
                count = len([line for line in f if line.strip()])
            print(f"  {cn_name}: {count} 个URL")
        else:
            print(f"  {cn_name}: 未找到URL文件")

if __name__ == "__main__":
    main()