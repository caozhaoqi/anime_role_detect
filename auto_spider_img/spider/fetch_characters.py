#!/usr/bin/env python3
"""
多源角色列表采集脚本 (并发优化版)
支持：萌娘百科、百度百科、Fandom
特性：多线程并发、请求重试、更强的解析逻辑
"""
import os
import requests
import re
import time
import urllib.parse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def clean_character_name(name):
    """清洗角色名称"""
    if not name:
        return None
    
    # 移除括号及其内容
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'（.*?）', '', name)
    name = re.sub(r'\[.*?\]', '', name)
    
    # 移除特殊字符
    name = name.strip()
    
    # 过滤无效名称
    if len(name) < 2 or len(name) > 20:
        return None
        
    # 排除关键词
    exclude_words = ["列表", "角色", "人物", "配音", "CV", "演员", "介绍", "概览", "相关", "制作", "音乐", "首页", "模板", "分类", "编辑", "更多", "查看", "收起", "图鉴", "攻略", "声优"]
    if any(word in name for word in exclude_words):
        return None
        
    # 排除纯数字或纯符号
    if re.match(r'^[\d\W]+$', name):
        return None
        
    return name

def fetch_url(url, retries=3):
    """带重试的请求函数"""
    for i in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                return response
        except requests.RequestException:
            time.sleep(1)
    return None

def fetch_from_moegirl(anime_name):
    """从萌娘百科获取"""
    print(f"  [萌娘百科] 搜索: {anime_name}")
    characters = set()
    
    try:
        # 1. 尝试直接访问 "作品名/角色列表" 页面
        sub_page_url = f"https://zh.moegirl.org.cn/{urllib.parse.quote(anime_name)}/角色列表"
        response = fetch_url(sub_page_url)
        
        if response:
            print(f"  [萌娘百科] 发现角色列表子页面: {sub_page_url}")
            soup = BeautifulSoup(response.text, 'html.parser')
            # 在角色列表页，通常是 h2/h3 标题或者表格第一列
            for link in soup.select('div.mw-parser-output a'):
                name = clean_character_name(link.text)
                if name: characters.add(name)
        
        # 2. 如果子页面没找到或结果太少，访问主页面
        if len(characters) < 5:
            search_url = f"https://zh.moegirl.org.cn/index.php?search={urllib.parse.quote(anime_name)}&title=Special:%E6%90%9C%E7%B4%A2&profile=default&fulltext=1"
            response = fetch_url(search_url)
            
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                page_url = None
                search_results = soup.select('.mw-search-result-heading a')
                if search_results:
                    page_url = "https://zh.moegirl.org.cn" + search_results[0].get('href')
                elif "搜索" not in soup.title.string:
                    page_url = response.url
                    
                if page_url:
                    print(f"  [萌娘百科] 访问主条目: {page_url}")
                    response = fetch_url(page_url)
                    if response:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # 策略A: 查找 navbox (导航模板)
                        navboxes = soup.select('table.navbox')
                        for navbox in navboxes:
                            if any(k in navbox.text for k in ["角色", "人物", "登场"]):
                                for link in navbox.select('a'):
                                    name = clean_character_name(link.text)
                                    if name: characters.add(name)

                        # 策略B: 查找正文中的列表
                        headers = soup.find_all(['h2', 'h3'], string=re.compile(r'角色|人物|登场'))
                        for header in headers:
                            for sibling in header.next_siblings:
                                if sibling.name in ['h2', 'h3']:
                                    break
                                if sibling.name in ['ul', 'ol', 'div', 'table']:
                                    if hasattr(sibling, 'select'):
                                        for link in sibling.select('a'):
                                            name = clean_character_name(link.text)
                                            if name: characters.add(name)
                                    
    except Exception as e:
        print(f"  [萌娘百科] 错误: {e}")
        
    return characters

def fetch_from_baike(anime_name):
    """从百度百科获取"""
    print(f"  [百度百科] 搜索: {anime_name}")
    characters = set()
    
    try:
        url = f"https://baike.baidu.com/item/{urllib.parse.quote(anime_name)}"
        response = fetch_url(url)
        
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 策略1: 查找所有 dt (定义项) 和 b (加粗)
            candidates = soup.select('dt, b, a')
            for cand in candidates:
                parent = cand.find_parent()
                if parent and ("角色" in parent.text or "人物" in parent.text):
                    name = clean_character_name(cand.text)
                    if name: characters.add(name)
                    
            # 策略2: 查找表格
            tables = soup.select('table')
            for table in tables:
                if "角色" in table.text or "配音" in table.text:
                    for row in table.select('tr'):
                        cols = row.select('td')
                        if cols:
                            name1 = clean_character_name(cols[0].text)
                            if name1: characters.add(name1)
                            if len(cols) > 1:
                                name2 = clean_character_name(cols[1].text)
                                if name2: characters.add(name2)

    except Exception as e:
        print(f"  [百度百科] 错误: {e}")
        
    return characters

def fetch_from_fandom(anime_name):
    """从Fandom获取 (针对特定大IP)"""
    print(f"  [Fandom] 搜索: {anime_name}")
    characters = set()
    
    try:
        if "原神" in anime_name:
            url = "https://genshin-impact.fandom.com/wiki/Characters"
            print(f"  [Fandom] 访问原神Wiki: {url}")
            response = fetch_url(url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.select('table.article-table td:nth-of-type(2) a'):
                    name = clean_character_name(link.text)
                    if name: characters.add(name)
                
        elif "崩坏" in anime_name and "星穹" in anime_name:
             url = "https://honkai-star-rail.fandom.com/wiki/Characters"
             print(f"  [Fandom] 访问星穹铁道Wiki: {url}")
             response = fetch_url(url)
             if response:
                 soup = BeautifulSoup(response.text, 'html.parser')
                 for link in soup.select('div.character-icon a'):
                     name = clean_character_name(link.get('title', ''))
                     if name: characters.add(name)
                     
        elif "鸣潮" in anime_name:
            url = "https://wutheringwaves.fandom.com/wiki/Resonators"
            print(f"  [Fandom] 访问鸣潮Wiki: {url}")
            response = fetch_url(url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.select('div.wds-tab__content a'):
                    name = clean_character_name(link.get('title', ''))
                    if name: characters.add(name)

    except Exception as e:
        print(f"  [Fandom] 错误: {e}")
        
    return characters

def process_anime(anime, output_dir):
    """处理单个作品"""
    print(f"\n=== 处理: {anime} ===")
    all_characters = set()
    
    # 并发请求多个来源
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(fetch_from_moegirl, anime): "Moegirl",
            executor.submit(fetch_from_baike, anime): "Baike",
            executor.submit(fetch_from_fandom, anime): "Fandom"
        }
        
        for future in as_completed(futures):
            source = futures[future]
            try:
                chars = future.result()
                print(f"  > {source} 找到 {len(chars)} 个")
                all_characters.update(chars)
            except Exception as e:
                print(f"  > {source} 失败: {e}")
    
    # 保存
    if all_characters:
        sorted_chars = sorted(list(all_characters))
        print(f"[{anime}] 总计找到 {len(sorted_chars)} 个唯一角色")
        
        safe_filename = re.sub(r'[\\/*?:"<>|]', '_', anime)
        output_path = os.path.join(output_dir, f"{safe_filename}.txt")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for char in sorted_chars:
                f.write(f"{char}\n")
        print(f"已保存到: {output_path}")
    else:
        print(f"[{anime}] 未找到任何角色")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "anime_set.txt")
    output_dir = os.path.join(base_dir, "characters")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(input_file):
        # 尝试在当前目录查找
        if os.path.exists("auto_spider_img/anime_set.txt"):
            input_file = "auto_spider_img/anime_set.txt"
            output_dir = "auto_spider_img/characters"
        else:
            print(f"输入文件不存在: {input_file}")
            return

    with open(input_file, 'r', encoding='utf-8') as f:
        anime_list = [line.strip() for line in f if line.strip()]
    
    # 串行处理每个作品，避免对单一网站并发过高被封
    # 但每个作品内部是并发请求不同来源的
    for anime in anime_list:
        process_anime(anime, output_dir)
        time.sleep(1) # 礼貌延时

if __name__ == "__main__":
    main()
