#!/usr/bin/env python3
"""手动下载图片不足20张的角色图片"""
import os
import sys
sys.path.insert(0, 'spider_image_system/src/run')
from constants import PINYIN_MAPPING

# 需要下载的角色
LOW_COUNT_ROLES = [
    '芙丽希娅',
    '洛茜', 
    '克萝萝',
    '德丽莎'
]

def download_role_images(role_name):
    """下载单个角色的图片"""
    pinyin = PINYIN_MAPPING.get(role_name)
    if not pinyin:
        print(f"❌ 未找到 {role_name} 的拼音映射")
        return 0
    
    # 检查URL文件是否存在
    url_file = f'spider_image_system/data/href_url/{pinyin}_url.txt'
    if not os.path.exists(url_file):
        print(f"❌ {role_name} 的URL文件不存在")
        return 0
    
    # 读取URL
    with open(url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # 去重
    urls = list(set(urls))
    print(f"📥 {role_name} 有 {len(urls)} 个URL")
    
    # 目标目录
    target_dir = f'data/organized_images/{pinyin}'
    os.makedirs(target_dir, exist_ok=True)
    
    # 使用爬虫系统的下载功能
    from image.spider_img_save import download_img_txt
    
    # 临时修改配置来下载特定角色
    import constants
    original_url_path = constants.data_url_path
    
    try:
        # 下载图片
        print(f"🚀 开始下载 {role_name} 的图片...")
        # 直接调用下载函数
        download_img_txt(url_file, target_dir)
        print(f"✅ {role_name} 下载完成")
        return len(urls)
    except Exception as e:
        print(f"❌ {role_name} 下载失败: {e}")
        return 0
    finally:
        constants.data_url_path = original_url_path

def main():
    print("=" * 60)
    print("📷 手动下载图片不足20张的角色")
    print("=" * 60)
    
    for role in LOW_COUNT_ROLES:
        print(f"\n📋 {role}")
        download_role_images(role)
    
    print("\n" + "=" * 60)
    print("📊 下载任务完成")
    print("=" * 60)

if __name__ == '__main__':
    main()
