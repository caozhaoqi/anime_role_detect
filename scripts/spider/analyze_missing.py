#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析未采集的角色
"""

import json
import os

def main():
    # 加载角色列表
    with open('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/roles.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    characters = data.get('characters', [])
    print(f'总角色数: {len(characters)}')
    
    # 检查已采集的角色目录
    image_dir = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/merged_images'
    collected = set()
    if os.path.exists(image_dir):
        for name in os.listdir(image_dir):
            if os.path.isdir(os.path.join(image_dir, name)):
                collected.add(name)
    
    print(f'已采集角色数: {len(collected)}')
    
    # 找出未采集的角色
    missing = []
    for char in characters:
        chinese_name = char['chinese_name']
        danbooru_tag = char.get('danbooru_tag', '')
        # 检查中文名或标签名是否存在
        if chinese_name not in collected and danbooru_tag not in collected:
            missing.append({
                'chinese_name': chinese_name,
                'danbooru_tag': danbooru_tag,
                'work_title': char.get('work_title', '')
            })
    
    print(f'\n未采集的角色 ({len(missing)} 个):')
    for char in missing:
        print(f"  {char['chinese_name']} - {char['work_title']} ({char['danbooru_tag']})")
    
    # 保存未采集角色列表
    output_file = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/missing_characters.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(missing, f, indent=2, ensure_ascii=False)
    print(f'\n已保存未采集角色列表到: {output_file}')

if __name__ == '__main__':
    main()
