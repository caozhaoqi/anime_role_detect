#!/usr/bin/env python3
"""
从 keywords 目录提取所有角色并整理成标准格式
输出: 中文名 作品名
"""

import os
from pathlib import Path
from collections import defaultdict

KEYWORDS_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/auto_spider_img/keywords")
OUTPUT_FILE = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/all_characters_formatted.txt")

# 作品名映射 - 从文件名推断作品
WORK_MAPPING = {
    '0307': '原神',
    '1_genshin': '原神',
    'genshin': '原神',
    '1_p_top': 'TOP榜',
    '3_star_rail': '崩坏星穹铁道',
    'star_rail': '崩坏星穹铁道',
    '6_honkai3': '崩坏3',
    'honkai3': '崩坏3',
    'blda': '蔚蓝档案',
    'cbjq': '重装战区',
    'ht': '海绵宝宝',
    'll': '鸣潮',
    'lsxy': '两生花',
    'mc': '鸣潮',
    'new': '新番',
    'qlwh': '七濑物语',
    'spider_img': '综合',
    'zzz': '绝区零',
    'p_top': 'TOP榜',
}

# 收集所有角色
all_chars = defaultdict(set)  # 角色名 -> 作品集合

for txt_file in KEYWORDS_DIR.glob("*.txt"):
    if txt_file.name.startswith('.'):
        continue
    
    # 获取作品名
    work_name = None
    file_stem = txt_file.stem.replace('_spider_img_keyword', '').replace('_chinese', '')
    for key, name in WORK_MAPPING.items():
        if key in file_stem.lower():
            work_name = name
            break
    
    if not work_name:
        work_name = file_stem
    
    # 读取角色
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                char_name = line.strip()
                if char_name and len(char_name) >= 2 and not char_name.startswith('#'):
                    all_chars[char_name].add(work_name)
    except Exception as e:
        print(f"读取 {txt_file} 失败: {e}")

# 按作品分类整理
work_chars = defaultdict(list)
for char_name, works in all_chars.items():
    for work in works:
        work_chars[work].append(char_name)

# 输出按作品分类的格式
output_lines = []
for work in sorted(work_chars.keys(), key=lambda x: -len(work_chars[x])):
    chars = sorted(work_chars[work])
    output_lines.append(f"# {work} ({len(chars)}个角色)")
    for char in chars:
        output_lines.append(f"{char}")
    output_lines.append("")  # 空行分隔

# 写入文件
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"=== 角色统计 ===")
print(f"总角色数: {len(all_chars)}")
print(f"输出文件: {OUTPUT_FILE}")
print()

print("=== 按作品分布 ===")
for work, chars in sorted(work_chars.items(), key=lambda x: -len(x[1])):
    print(f"{work}: {len(chars)} 个角色")

print()
print("=== 完整列表预览 ===")
print('\n'.join(output_lines[:50]))
print("...")
