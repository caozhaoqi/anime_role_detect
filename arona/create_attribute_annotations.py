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


def load_character_attributes():
    """加载角色属性定义"""
    with open('character_attributes.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def create_annotations(data_dir, output_file):
    """为训练数据创建属性标注
    
    Args:
        data_dir: 数据目录
        output_file: 输出标注文件路径
    """
    # 加载角色属性定义
    attribute_data = load_character_attributes()
    characters = attribute_data['characters']
    attribute_order = attribute_data['attribute_order']
    attribute_mappings = attribute_data['attribute_mappings']
    
    annotations = []
    
    # 遍历每个角色目录
    for character in os.listdir(data_dir):
        character_dir = os.path.join(data_dir, character)
        if not os.path.isdir(character_dir):
            continue
        
        # 检查角色是否在属性定义中
        if character not in characters:
            logger.warning(f"角色 {character} 不在属性定义中，跳过")
            continue
        
        # 获取角色属性
        character_attributes = characters[character]['attributes']
        
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
                attr_value = character_attributes.get(attr_name)
                if attr_name in attribute_mappings:
                    # 转换属性值为索引
                    if isinstance(attr_value, bool):
                        attr_value = str(attr_value).lower()
                    if attr_value in attribute_mappings[attr_name]:
                        attribute_labels.append(attribute_mappings[attr_name][attr_value])
                    else:
                        logger.warning(f"属性值 {attr_value} 不在映射中，使用默认值")
                        attribute_labels.append(0)
                else:
                    attribute_labels.append(0)
            
            # 添加标注
            annotation = {
                'image_path': os.path.relpath(img_path, data_dir),
                'character': character,
                'attributes': character_attributes,
                'attribute_labels': attribute_labels
            }
            annotations.append(annotation)
    
    # 保存标注结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    logger.info(f"标注完成，共处理 {len(annotations)} 张图像")
    logger.info(f"标注结果保存到: {output_file}")


def main():
    # 数据目录
    data_dir = '../data/downloaded_images'
    # 输出文件
    output_file = 'attribute_annotations.json'
    
    create_annotations(data_dir, output_file)


if __name__ == '__main__':
    main()