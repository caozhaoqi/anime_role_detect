#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 SVG 图像转换为 PNG 格式
"""

import os
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('convert_svg_to_png')

def convert_svg_to_png(input_dir, output_dir):
    """将目录中的所有 SVG 图像转换为 PNG 格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(input_dir):
        # 计算相对路径，保持目录结构
        relative_path = os.path.relpath(root, input_dir)
        if relative_path == '.':
            relative_path = ''
        
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)
        
        for file in files:
            if file.lower().endswith('.svg'):
                input_path = os.path.join(root, file)
                output_file = os.path.splitext(file)[0] + '.png'
                output_path = os.path.join(output_subdir, output_file)
                
                try:
                    # 使用 ImageMagick 将 SVG 转换为 PNG
                    subprocess.run(
                        ['convert', input_path, '-resize', '256x256', output_path],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    logger.info(f'转换成功: {input_path} -> {output_path}')
                except subprocess.CalledProcessError as e:
                    logger.error(f'转换失败: {input_path}')
                    logger.error(f'错误信息: {e.stderr}')

if __name__ == '__main__':
    input_dir = 'data/train'
    output_dir = 'data/train_png'
    
    logger.info(f'开始将 SVG 图像转换为 PNG 格式...')
    logger.info(f'输入目录: {input_dir}')
    logger.info(f'输出目录: {output_dir}')
    
    convert_svg_to_png(input_dir, output_dir)
    
    logger.info('转换完成！')
