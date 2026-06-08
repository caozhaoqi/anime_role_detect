#!/usr/bin/env python3
"""检测数据目录中所有图片是否可以正常打开"""

import os
from pathlib import Path
from PIL import Image
import io

def is_valid_image(file_path):
    """
    检查图片文件是否可以正常打开
    
    Returns:
        (是否有效, 错误信息或图片信息)
    """
    try:
        # 先检查文件大小
        if os.path.getsize(file_path) == 0:
            return False, "空文件"
        
        # 尝试打开图片
        with Image.open(file_path) as img:
            # 尝试读取图片数据
            img.verify()
            
            # 重新打开以获取图片信息
            img = Image.open(file_path)
            img.load()
            
            # 获取图片信息
            info = {
                'format': img.format,
                'size': img.size,
                'mode': img.mode
            }
            return True, info
            
    except Exception as e:
        return False, str(e)

def check_all_images(data_dir):
    """检查目录中所有图片"""
    data_path = Path(data_dir)
    
    all_images = []
    for char_dir in data_path.iterdir():
        if not char_dir.is_dir():
            continue
        
        images = list(char_dir.glob('*'))
        for img_path in images:
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.gif'):
                all_images.append((char_dir.name, img_path))
    
    print(f'📁 开始检测 {len(all_images)} 张图片...')
    
    valid_count = 0
    invalid_count = 0
    invalid_images = []
    
    for i, (char_name, img_path) in enumerate(all_images, 1):
        if i % 50 == 0:
            print(f'   已检测 {i}/{len(all_images)} 张图片...')
        
        is_valid, info = is_valid_image(img_path)
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            invalid_images.append({
                'char_name': char_name,
                'file_name': img_path.name,
                'error': info
            })
    
    print(f'\n📊 检测结果:')
    print(f'   ✅ 有效图片: {valid_count} 张')
    print(f'   ❌ 无效图片: {invalid_count} 张')
    
    if invalid_images:
        print(f'\n❌ 无效图片列表:')
        for item in invalid_images[:20]:  # 最多显示20个
            print(f'   {item["char_name"]}/{item["file_name"]}: {item["error"]}')
        
        if len(invalid_images) > 20:
            print(f'   ... 还有 {len(invalid_images) - 20} 个无效图片')
    
    return valid_count, invalid_count, invalid_images

if __name__ == '__main__':
    data_dir = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset'
    check_all_images(data_dir)