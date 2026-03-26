#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将SVG图片转换为PNG格式，用于测试
"""

import os
import sys
from PIL import Image
import cairosvg

def convert_svg_to_png(svg_dir, png_dir):
    """将SVG目录下的所有SVG文件转换为PNG格式"""
    if not os.path.exists(svg_dir):
        print(f"SVG目录不存在: {svg_dir}")
        return
    
    # 创建PNG目录
    os.makedirs(png_dir, exist_ok=True)
    
    # 获取所有SVG文件
    svg_files = [f for f in os.listdir(svg_dir) if f.lower().endswith('.svg')]
    
    if not svg_files:
        print(f"没有找到SVG文件在目录: {svg_dir}")
        return
    
    print(f"找到 {len(svg_files)} 个SVG文件")
    
    converted_count = 0
    for svg_file in svg_files:
        svg_path = os.path.join(svg_dir, svg_file)
        png_file = os.path.splitext(svg_file)[0] + '.png'
        png_path = os.path.join(png_dir, png_file)
        
        try:
            # 使用cairosvg转换SVG到PNG
            cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=224, output_height=224)
            print(f"✓ 转换成功: {svg_file} -> {png_file}")
            converted_count += 1
        except Exception as e:
            print(f"✗ 转换失败: {svg_file}, 错误: {e}")
    
    print(f"\n转换完成: {converted_count}/{len(svg_files)} 个文件成功转换")

def main():
    """主函数"""
    data_dir = 'data/train'
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    # 获取所有角色目录
    role_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"发现 {len(role_dirs)} 个角色目录")
    
    # 为每个角色目录转换SVG到PNG
    for role_name in role_dirs:
        svg_dir = os.path.join(data_dir, role_name)
        png_dir = os.path.join(data_dir, role_name + '_png')
        
        print(f"\n处理角色: {role_name}")
        convert_svg_to_png(svg_dir, png_dir)
    
    print("\n所有转换完成！")

if __name__ == "__main__":
    main()
