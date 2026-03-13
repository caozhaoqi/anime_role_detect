#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将生成的标签文件转换为模型训练所需的标注格式
"""

import os
import json
import argparse

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from core.logging.global_logger import get_logger

logger = get_logger("convert_tags_to_annotations")


def convert_tags_to_annotations(tags_file, output_file):
    """将生成的标签文件转换为模型训练所需的标注格式
    
    Args:
        tags_file: 生成的标签文件路径
        output_file: 输出的标注文件路径
    """
    # 读取标签文件
    with open(tags_file, 'r', encoding='utf-8') as f:
        tags_data = json.load(f)
    
    # 提取所有标签
    all_tags = set()
    for item in tags_data:
        all_tags.update(item['tags'])
    
    # 构建标签到索引的映射
    tag_to_idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    logger.info(f'共提取到 {len(tag_to_idx)} 个唯一标签')
    
    # 转换标注格式
    annotations = []
    for item in tags_data:
        # 从图像路径中提取角色名称
        image_path = item['image_path']
        character = os.path.dirname(image_path)
        
        # 将标签转换为数值表示
        attribute_labels = [0] * len(tag_to_idx)
        for tag in item['tags']:
            if tag in tag_to_idx:
                attribute_labels[tag_to_idx[tag]] = 1
        
        # 添加到标注列表
        annotations.append({
            'character': character,
            'image_path': image_path,
            'attribute_labels': attribute_labels
        })
    
    # 保存标注文件
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    
    logger.info(f'转换完成，共处理 {len(annotations)} 张图像')
    logger.info(f'标注文件保存到: {output_file}')
    
    # 保存标签映射
    tag_map_file = os.path.join(os.path.dirname(output_file), 'tag_to_idx.json')
    with open(tag_map_file, 'w', encoding='utf-8') as f:
        json.dump(tag_to_idx, f, ensure_ascii=False, indent=2)
    logger.info(f'标签映射文件保存到: {tag_map_file}')


def main():
    parser = argparse.ArgumentParser(description='将生成的标签文件转换为模型训练所需的标注格式')
    parser.add_argument('--tags-file', type=str, default='data/tags.json', help='生成的标签文件路径')
    parser.add_argument('--output-file', type=str, default='data/annotations.json', help='输出的标注文件路径')
    
    args = parser.parse_args()
    
    convert_tags_to_annotations(args.tags_file, args.output_file)


if __name__ == '__main__':
    main()
