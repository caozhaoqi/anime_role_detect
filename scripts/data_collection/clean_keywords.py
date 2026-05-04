#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理关键词文件，只保留loli-role.txt中的角色"""

from pathlib import Path

ROLE_FILE = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/loli-role.txt')
KEYWORD_FILE = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/spider_image_system/data/auto_spider_img/spider_img_keyword.txt')

def main():
    print("=== 清理关键词文件 ===")
    
    # 读取loli-role.txt中的角色
    roles = []
    with open(ROLE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ')
                chinese_name = parts[0]
                roles.append(chinese_name)
    
    print(f"loli-role.txt 中的角色数: {len(roles)}")
    print(f"角色列表: {', '.join(roles[:10])}...")
    
    # 读取原有关键词文件
    old_keywords = []
    if KEYWORD_FILE.exists():
        with open(KEYWORD_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    old_keywords.append(line)
    
    print(f"\n原有关键词数: {len(old_keywords)}")
    
    # 写入新的关键词文件（只包含名单中的角色）
    with open(KEYWORD_FILE, 'w', encoding='utf-8') as f:
        for role in roles:
            f.write(role + '\n')
    
    print(f"\n✅ 已更新关键词文件")
    print(f"   新关键词数: {len(roles)}")
    print(f"   文件路径: {KEYWORD_FILE}")

if __name__ == '__main__':
    main()
