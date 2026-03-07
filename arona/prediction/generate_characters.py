#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色图片生成脚本
用于生成蔚蓝档案阿罗娜和普拉娜的角色图片
"""

import os
import requests
import argparse
import logging
from PIL import Image
from io import BytesIO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('generate_characters')


def generate_character_image(character_name, output_dir, num_images=5):
    """生成角色图片
    
    Args:
        character_name: 角色名称
        output_dir: 输出目录
        num_images: 生成的图片数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 角色描述
    character_descriptions = {
        'arona': "Arona from Blue Archive, blue short hair, halo, school uniform, cute, anime style",
        'plana': "Plana from Blue Archive, black long hair, halo, school uniform, elegant, anime style"
    }
    
    if character_name not in character_descriptions:
        logger.error(f"不支持的角色: {character_name}")
        return
    
    description = character_descriptions[character_name]
    logger.info(f"开始生成 {character_name} 的图片...")
    
    for i in range(num_images):
        try:
            # 使用Trae API生成图片
            prompt = f"{description}, high quality, detailed, anime, 4k"
            url = f"https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt={prompt}&image_size=portrait_4_3"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # 保存图片
            image = Image.open(BytesIO(response.content))
            output_path = os.path.join(output_dir, f"{character_name}_{i+1}.png")
            image.save(output_path)
            logger.info(f"生成图片: {output_path}")
            
        except Exception as e:
            logger.error(f"生成图片时出错: {e}")
            continue
    
    logger.info(f"{character_name} 图片生成完成")


def main():
    parser = argparse.ArgumentParser(description='角色图片生成脚本')
    parser.add_argument('--character', type=str, choices=['arona', 'plana', 'both'], 
                       default='both', help='要生成的角色')
    parser.add_argument('--num-images', type=int, default=5, help='每个角色生成的图片数量')
    parser.add_argument('--output-dir', type=str, default='data/generated', help='输出目录')
    
    args = parser.parse_args()
    
    if args.character == 'both' or args.character == 'arona':
        generate_character_image('arona', os.path.join(args.output_dir, '蔚蓝档案_阿罗娜'), args.num_images)
    
    if args.character == 'both' or args.character == 'plana':
        generate_character_image('plana', os.path.join(args.output_dir, '蔚蓝档案_普拉娜'), args.num_images)


if __name__ == '__main__':
    main()
