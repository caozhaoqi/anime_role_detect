#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成WD Vit V3 Tagger生成的标签到现有属性标注系统
"""

import os
import json
import argparse
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('integrate_tags')


def load_wd_tags(tags_dir, image_path):
    """加载WD Vit V3 Tagger生成的标签
    
    Args:
        tags_dir: 标签目录
        image_path: 图片路径（相对于下载目录的路径）
    
    Returns:
        list: 标签列表
    """
    tag_file = os.path.join(tags_dir, os.path.splitext(image_path)[0] + '.json')
    
    if os.path.exists(tag_file):
        try:
            with open(tag_file, 'r', encoding='utf-8') as f:
                tag_data = json.load(f)
                return tag_data.get('tags', [])
        except Exception as e:
            logger.error(f"加载标签文件失败: {tag_file}, 错误: {e}")
    
    return []


def integrate_tags(annotations_file, tags_dir, output_file):
    """集成WD标签到现有标注文件
    
    Args:
        annotations_file: 现有标注文件
        tags_dir: WD标签目录
        output_file: 输出文件
    """
    # 加载现有标注
    if not os.path.exists(annotations_file):
        logger.error(f"标注文件不存在: {annotations_file}")
        return
    
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    logger.info(f"加载了 {len(annotations)} 条标注")
    
    # 处理每条标注
    for item in tqdm(annotations, desc="集成标签"):
        image_path = item.get('image_path')
        if image_path:
            # 加载WD标签
            wd_tags = load_wd_tags(tags_dir, image_path)
            # 添加WD标签到标注中
            item['wd_tags'] = wd_tags
            item['wd_tag_count'] = len(wd_tags)
    
    # 保存集成后的标注
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    logger.info(f"集成完成，结果保存在: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='集成WD Vit V3 Tagger标签到现有标注系统')
    parser.add_argument('--annotations', type=str, default='attribute_annotations.json', help='现有标注文件')
    parser.add_argument('--tags-dir', type=str, default='../data/image_tags', help='WD标签目录')
    parser.add_argument('--output', type=str, default='attribute_annotations_with_wd_tags.json', help='输出文件')
    
    args = parser.parse_args()
    
    integrate_tags(args.annotations, args.tags_dir, args.output)


if __name__ == '__main__':
    main()