#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为训练数据创建属性标注
"""

import os
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('create_attribute_annotations')


def load_character_attributes(config_path=None):
    """加载角色属性定义
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
    """
    if config_path is None:
        # 尝试多个可能的配置文件路径
        possible_paths = [
            'character_attributes.json',
            '../config/character_attributes.json',
            '../../config/character_attributes.json',
            os.path.join(os.path.dirname(__file__), 'character_attributes.json'),
            os.path.join(os.path.dirname(__file__), '..', 'config', 'character_attributes.json')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path is None or not os.path.exists(config_path):
        logger.warning("无法找到角色属性配置文件，将使用空配置")
        return {"characters": {}, "attribute_mappings": {}, "attribute_order": []}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_annotations(data_dir, output_file, config_path=None):
    """为训练数据创建属性标注
    
    Args:
        data_dir: 数据目录
        output_file: 输出标注文件路径
        config_path: 配置文件路径
    """
    # 加载角色属性定义
    attribute_data = load_character_attributes(config_path)
    characters = attribute_data.get('characters', {})
    attribute_order = attribute_data.get('attribute_order', [])
    attribute_mappings = attribute_data.get('attribute_mappings', {})
    
    annotations = []
    skipped_characters = []
    
    # 遍历每个角色目录
    for character in os.listdir(data_dir):
        character_dir = os.path.join(data_dir, character)
        if not os.path.isdir(character_dir):
            continue
        
        # 检查角色是否在属性定义中
        if character not in characters:
            skipped_characters.append(character)
            logger.warning(f"角色 {character} 不在属性定义中，跳过")
            continue
        
        # 获取角色属性
        character_attrs = characters[character]
        
        # 遍历角色目录下的所有图像
        for img_name in os.listdir(character_dir):
            img_path = os.path.join(character_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            
            # 检查文件扩展名
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                continue
            
            # 生成属性标签
            attribute_labels = []
            for attr_name in attribute_order:
                attr_value = character_attrs.get(attr_name, "unknown")
                if attr_name in attribute_mappings:
                    # 转换属性值为索引
                    if isinstance(attr_value, bool):
                        attr_value = str(attr_value).lower()
                    mapping = attribute_mappings.get(attr_name, {})
                    if attr_value in mapping:
                        attribute_labels.append(mapping[attr_value])
                    else:
                        logger.warning(f"属性值 {attr_value} 不在映射中，使用默认值")
                        attribute_labels.append(0)
                else:
                    attribute_labels.append(0)
            
            # 添加标注
            annotation = {
                'image_path': os.path.relpath(img_path, data_dir),
                'character': character,
                'attributes': character_attrs,
                'attribute_labels': attribute_labels
            }
            annotations.append(annotation)
    
    # 保存标注结果
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    logger.info(f"标注完成，共处理 {len(annotations)} 张图像")
    if skipped_characters:
        logger.warning(f"跳过的角色: {skipped_characters}")
    logger.info(f"标注结果保存到: {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='为训练数据创建属性标注')
    parser.add_argument('--data-dir', type=str, default='../data/downloaded_images', help='数据目录')
    parser.add_argument('--output-file', type=str, default='attribute_annotations.json', help='输出文件路径')
    parser.add_argument('--config', type=str, default=None, help='属性配置文件路径')
    
    args = parser.parse_args()
    
    create_annotations(args.data_dir, args.output_file, args.config)


if __name__ == '__main__':
    main()