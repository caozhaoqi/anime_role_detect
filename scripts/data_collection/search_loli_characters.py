#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过网络搜索判断角色是否为萝莉
"""

import os
import re
import time
import logging
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('search_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 搜索配置
SEARCH_TIMEOUT = 10  # 搜索超时时间（秒）
SEARCH_DELAY = 2  # 搜索延迟时间（秒）
SEARCH_MAX_RETRIES = 3  # 搜索最大重试次数

# 搜索关键词模板
SEARCH_TEMPLATES = [
    "{character} 萝莉 分类",
    "{character} 年龄 动漫",
    "{character} 角色设定",
    "{character} loli 判断"
]

# 萝莉相关关键词
LOLI_KEYWORDS = [
    "萝莉", "loli", "幼女", "小学生", "初中生", "12岁", "13岁", "14岁",
    "可爱", "娇小", "天真", "活泼", "萌"
]

# 非萝莉相关关键词
NON_LOLI_KEYWORDS = [
    "御姐", "成年", "高中生", "大学生", "20岁", "成熟", "性感", "高挑"
]

def search_character_info(character):
    """搜索角色信息
    
    Args:
        character: 角色名
        
    Returns:
        str: 搜索结果文本
    """
    search_results = []
    
    for template in SEARCH_TEMPLATES:
        query = template.format(character=character)
        logger.info(f"搜索: {query}")
        
        for retry in range(SEARCH_MAX_RETRIES):
            try:
                # 使用百度搜索
                url = f"https://www.baidu.com/s?wd={query}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                }
                response = requests.get(url, headers=headers, timeout=SEARCH_TIMEOUT)
                response.encoding = 'utf-8'
                
                # 解析搜索结果
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 获取搜索结果
                results = soup.find_all('div', class_='result c-container')
                for result in results[:3]:  # 只取前3个结果
                    title = result.find('h3').text.strip() if result.find('h3') else ''
                    content = result.find('div', class_='c-abstract').text.strip() if result.find('div', class_='c-abstract') else ''
                    search_results.append(f"{title}\n{content}")
                
                break  # 成功搜索，跳出重试循环
            except Exception as e:
                logger.error(f"搜索失败 ({retry+1}/{SEARCH_MAX_RETRIES}): {e}")
                time.sleep(SEARCH_DELAY)
        
        time.sleep(SEARCH_DELAY)  # 搜索之间的延迟
    
    return "\n\n".join(search_results)

def analyze_search_results(character, search_results):
    """分析搜索结果，判断角色是否为萝莉
    
    Args:
        character: 角色名
        search_results: 搜索结果文本
        
    Returns:
        bool: 是否为萝莉
    """
    if not search_results:
        logger.info(f"角色 {character} 未找到搜索结果，默认为: 否")
        return False
    
    # 统计关键词出现次数
    loli_count = 0
    non_loli_count = 0
    
    for keyword in LOLI_KEYWORDS:
        loli_count += search_results.count(keyword)
    
    for keyword in NON_LOLI_KEYWORDS:
        non_loli_count += search_results.count(keyword)
    
    # 分析年龄信息
    age_pattern = re.compile(r'(\d+)岁')
    ages = age_pattern.findall(search_results)
    young_age = False
    for age in ages:
        try:
            age_num = int(age)
            if age_num <= 14:
                young_age = True
                break
        except ValueError:
            pass
    
    # 判断结果
    if young_age or loli_count > non_loli_count:
        logger.info(f"角色 {character} 被识别为: 是")
        return True
    else:
        logger.info(f"角色 {character} 被识别为: 否")
        return False

def search_loli_character(character):
    """通过网络搜索判断角色是否为萝莉
    
    Args:
        character: 角色名
        
    Returns:
        bool: 是否为萝莉
    """
    logger.info(f"开始搜索角色: {character}")
    
    # 搜索角色信息
    search_results = search_character_info(character)
    logger.info(f"搜索结果:\n{search_results[:500]}...")  # 只显示前500个字符
    
    # 分析搜索结果
    is_loli = analyze_search_results(character, search_results)
    
    return is_loli

def load_characters_from_directory(directory):
    """从目录中加载所有角色
    
    Args:
        directory: 目录路径
        
    Returns:
        list: 角色名列表
    """
    characters = []
    
    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            character = line.strip()
                            if character:
                                characters.append(character)
                except Exception as e:
                    logger.error(f"读取文件 {file_path} 失败: {e}")
    
    # 去重
    characters = list(set(characters))
    logger.info(f"共加载 {len(characters)} 个角色")
    return characters

def main():
    """主函数"""
    print("========================================")
    print("通过网络搜索判断角色是否为萝莉")
    print("========================================")
    
    # 选择角色来源
    print("请选择角色来源:")
    print("1. 从指定文件读取角色")
    print("2. 从 auto_spider_img 目录读取角色")
    
    choice = input("请输入选择 (1/2): ").strip()
    
    if choice == "1":
        file_path = input("请输入角色文件路径: ").strip()
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在")
            return
        
        # 读取角色
        characters = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    character = line.strip()
                    if character:
                        characters.append(character)
        except Exception as e:
            print(f"读取文件失败: {e}")
            return
    elif choice == "2":
        directory = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img"
        characters = load_characters_from_directory(directory)
    else:
        print("无效选择")
        return
    
    if not characters:
        print("未找到角色")
        return
    
    print(f"共加载 {len(characters)} 个角色")
    
    # 询问是否只使用网络搜索
    use_only_search = input("是否只使用网络搜索 (y/n): ").strip().lower() == "y"
    
    # 处理角色
    loli_characters = []
    non_loli_characters = []
    
    for i, character in enumerate(characters, 1):
        print(f"处理角色 {i}/{len(characters)}: {character}")
        is_loli = search_loli_character(character)
        
        if is_loli:
            loli_characters.append(character)
        else:
            non_loli_characters.append(character)
        
        # 处理完一个角色后休息一下，避免请求过于频繁
        time.sleep(SEARCH_DELAY)
    
    # 保存结果
    output_dir = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/lolis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存萝莉角色
    loli_file = os.path.join(output_dir, "loli_characters_search.txt")
    with open(loli_file, 'w', encoding='utf-8') as f:
        for character in loli_characters:
            f.write(f"{character}\n")
    
    # 保存非萝莉角色
    non_loli_file = os.path.join(output_dir, "non_loli_characters_search.txt")
    with open(non_loli_file, 'w', encoding='utf-8') as f:
        for character in non_loli_characters:
            f.write(f"{character}\n")
    
    # 输出结果
    print("\n========================================")
    print("分类结果:")
    print(f"萝莉角色数量: {len(loli_characters)}")
    print(f"非萝莉角色数量: {len(non_loli_characters)}")
    
    if loli_characters:
        print("\n前20个萝莉角色:")
        for i, character in enumerate(loli_characters[:20], 1):
            print(f"  {i}. {character}")
    
    print(f"\n分类结果已保存到: {output_dir}")
    print("========================================")
    print("分类完成")
    print("========================================")
    print(f"萝莉角色列表已保存到: {loli_file}")
    print("分类方法通过网络搜索实现")
    print("========================================")

if __name__ == "__main__":
    main()
