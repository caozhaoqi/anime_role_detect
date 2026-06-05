#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据清洗模块（简化版，不使用CLIP）
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.cleaner import QualityFilter, CharacterCropper


def test_simple_modules():
    """测试简单的清洗模块"""
    # 找一个测试图片
    test_image = None
    data_dir = Path("data/final_dataset")
    for char_dir in data_dir.iterdir():
        if char_dir.is_dir():
            for img_file in char_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.png']:
                    test_image = str(img_file)
                    break
            if test_image:
                break
    
    if not test_image:
        print("❌ 未找到测试图片")
        return
    
    print(f"📷 测试图片: {test_image}")
    
    # 测试QualityFilter
    print('\n🎯 测试 QualityFilter...')
    filter = QualityFilter()
    quality_ok, info = filter.filter(test_image)
    print(f'  质量过滤: {quality_ok}')
    print(f'  详细信息: {info}')
    
    # 测试CharacterCropper
    print('\n🎯 测试 CharacterCropper...')
    cropper = CharacterCropper()
    # 测试边界框裁剪
    bbox = (100, 100, 300, 300)
    result = cropper.crop_character(test_image, bbox)
    print(f'  裁剪结果: {result}')
    
    # 测试角色占比计算
    ratio = cropper.calculate_character_ratio(test_image, bbox)
    print(f'  角色占比: {ratio:.2%}')
    
    print('\n✅ 简单模块测试通过！')


if __name__ == "__main__":
    test_simple_modules()
