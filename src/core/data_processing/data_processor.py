#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的数据处理脚本
"""

import os
import sys
import argparse
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from core.utils.utils import load_json, save_json, create_directory, list_files
from core.logging.global_logger import get_logger

logger = get_logger("data_processing")


def process_dataset(data_dir, output_file):
    """处理数据集，生成标注文件
    
    Args:
        data_dir: 数据目录
        output_file: 输出文件路径
    """
    annotations = []
    
    # 遍历数据目录
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                # 获取相对路径
                relative_path = os.path.relpath(os.path.join(root, file), data_dir)
                # 获取类别名称
                class_name = os.path.basename(root)
                
                # 创建标注
                annotation = {
                    'image_path': relative_path,
                    'character': class_name,
                    'attribute_labels': []  # 占位符，后续会填充
                }
                annotations.append(annotation)
    
    # 保存标注
    save_json(annotations, output_file)
    logger.info(f"数据集处理完成，保存到: {output_file}")


def integrate_attributes(annotations_file, attributes_file, output_file):
    """整合属性标注
    
    Args:
        annotations_file: 标注文件路径
        attributes_file: 属性配置文件路径
        output_file: 输出文件路径
    """
    # 加载标注
    annotations = load_json(annotations_file)
    # 加载属性配置
    attributes = load_json(attributes_file)
    
    # 整合属性
    for annotation in annotations:
        character = annotation['character']
        if character in attributes:
            # 转换属性为标签
            attribute_labels = []
            
            # 头发颜色
            hair_color = attributes[character].get('hair_color', 'blue')
            hair_color_map = {'blue': 0, 'black': 1, 'brown': 2, 'blonde': 3, 'red': 4, 'green': 5, 'purple': 6}
            attribute_labels.append(hair_color_map.get(hair_color, 0))
            
            # 眼睛颜色
            eye_color = attributes[character].get('eye_color', 'blue')
            eye_color_map = {'blue': 0, 'black': 1, 'brown': 2, 'red': 3, 'green': 4, 'purple': 5, 'yellow': 6}
            attribute_labels.append(eye_color_map.get(eye_color, 0))
            
            # 是否有光环
            has_halo = attributes[character].get('has_halo', False)
            attribute_labels.append(1 if has_halo else 0)
            
            # 服装类型
            outfit = attributes[character].get('outfit', 'school_uniform')
            outfit_map = {'school_uniform': 0, 'dress': 1, 'casual': 2, 'uniform': 3, 'swimsuit': 4}
            attribute_labels.append(outfit_map.get(outfit, 0))
            
            # 发型
            hair_style = attributes[character].get('hair_style', 'twintails')
            hair_style_map = {'twintails': 0, 'long_hair': 1, 'short_hair': 2, 'bun': 3, 'ponytail': 4}
            attribute_labels.append(hair_style_map.get(hair_style, 0))
            
            # 配饰
            accessories = attributes[character].get('accessories', [])
            accessories_count = len(accessories)
            attribute_labels.append(min(accessories_count, 5))  # 最多5个配饰
            
            annotation['attribute_labels'] = attribute_labels
    
    # 保存整合后的标注
    save_json(annotations, output_file)
    logger.info(f"属性整合完成，保存到: {output_file}")


def integrate_tags(annotations_file, tags_file, output_file):
    """整合标签
    
    Args:
        annotations_file: 标注文件路径
        tags_file: 标签文件路径
        output_file: 输出文件路径
    """
    # 加载标注
    annotations = load_json(annotations_file)
    # 加载标签
    tags_data = load_json(tags_file)
    
    # 创建标签映射
    tag_map = {item['image_path']: item['tags'] for item in tags_data}
    
    # 整合标签
    for annotation in annotations:
        image_path = annotation['image_path']
        if image_path in tag_map:
            annotation['tags'] = tag_map[image_path]
    
    # 保存整合后的标注
    save_json(annotations, output_file)
    logger.info(f"标签整合完成，保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='统一的数据处理脚本')
    parser.add_argument('--data-dir', type=str, default='data/downloaded_images', help='数据目录')
    parser.add_argument('--annotations-file', type=str, default='data/annotations.json', help='标注文件路径')
    parser.add_argument('--attributes-file', type=str, default='config/character_attributes.json', help='属性配置文件路径')
    parser.add_argument('--tags-file', type=str, default=None, help='标签文件路径')
    parser.add_argument('--output-file', type=str, default='data/processed_annotations.json', help='输出文件路径')
    parser.add_argument('--process-dataset', action='store_true', help='处理数据集')
    parser.add_argument('--integrate-attributes', action='store_true', help='整合属性')
    parser.add_argument('--integrate-tags', action='store_true', help='整合标签')
    
    args = parser.parse_args()
    
    # 处理数据集
    if args.process_dataset:
        process_dataset(args.data_dir, args.annotations_file)
    
    # 整合属性
    if args.integrate_attributes:
        integrate_attributes(args.annotations_file, args.attributes_file, args.output_file)
    
    # 整合标签
    if args.integrate_tags and args.tags_file:
        integrate_tags(args.annotations_file, args.tags_file, args.output_file)
    
    if not any([args.process_dataset, args.integrate_attributes, args.integrate_tags]):
        parser.print_help()


if __name__ == '__main__':
    main()