#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的角色数据采集脚本

为低准确率角色补充采集图像数据
"""

import os
import requests
import time
import random
import argparse
from PIL import Image
from io import BytesIO

# 配置
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 低准确率角色列表
LOW_ACCURACY_CHARACTERS = [
    # 准确率为0的角色
    {"series": "demon_slayer", "name": "炭治郎", "target_count": 100},
    {"series": "demon_slayer", "name": "祢豆子", "target_count": 100},
    {"series": "honkai_star_rail", "name": "丹恒", "target_count": 100},
    {"series": "honkai_star_rail", "name": "姬子", "target_count": 100},
    {"series": "honkai_star_rail", "name": "瓦尔特", "target_count": 100},
    {"series": "tokyo_ghoul", "name": "董香", "target_count": 100},
    {"series": "tokyo_ghoul", "name": "金木研", "target_count": 100},
    # 准确率低于60%的角色
    {"series": "honkai_impact_3", "name": "琪亚娜·卡斯兰娜", "target_count": 80},
    {"series": "honkai_impact_3", "name": "雷电芽衣", "target_count": 80},
    {"series": "honkai_star_rail", "name": "三月七", "target_count": 80},
    # 准确率在60-80%之间的角色
    {"series": "attack_on_titan", "name": "艾伦", "target_count": 60},
    {"series": "honkai_impact_3", "name": "琪亚娜", "target_count": 60},
    {"series": "one_piece", "name": "路飞", "target_count": 60},
    {"series": "dragon_ball", "name": "孙悟空", "target_count": 60},
    {"series": "naruto", "name": "鸣人", "target_count": 60},
    {"series": "genshin_impact", "name": "雷电将军", "target_count": 60}
]

def validate_image(content):
    """验证图像是否有效"""
    try:
        img = Image.open(BytesIO(content))
        img.verify()
        return True
    except:
        return False

def download_image(url, save_path):
    """下载图像"""
    try:
        # 随机延迟，避免被封禁
        time.sleep(random.uniform(0.5, 2.0))
        
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            content = response.content
            if validate_image(content):
                # 保存图像
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(content)
                return True
    except Exception as e:
        print(f"下载失败: {url}, 错误: {e}")
    return False

def collect_character_images(series, character_name, target_count):
    """为单个角色采集图像"""
    # 创建角色目录
    character_dir = os.path.join('../data', 'train', f"{series}_{character_name}")
    os.makedirs(character_dir, exist_ok=True)
    
    # 获取已有图像数量
    existing_images = [f for f in os.listdir(character_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    existing_count = len(existing_images)
    
    if existing_count >= target_count:
        print(f"{series}_{character_name} 已有 {existing_count} 张图像，达到目标数量")
        return existing_count
    
    needed_count = target_count - existing_count
    print(f"开始为 {series}_{character_name} 采集 {needed_count} 张图像")
    
    # 简单的图像源（实际项目中可以使用更复杂的图像搜索API）
    # 这里使用placeholder.com作为示例，实际采集时需要替换为真实的图像源
    base_url = "https://via.placeholder.com/512"
    
    downloaded_count = 0
    for i in range(needed_count):
        # 生成唯一的URL以获取不同的占位图像
        url = f"{base_url}?text={series}_{character_name}_{i}"
        save_path = os.path.join(character_dir, f"{series}_{character_name}_{existing_count + i:04d}.jpg")
        
        if download_image(url, save_path):
            downloaded_count += 1
            print(f"已下载 {downloaded_count}/{needed_count} 张图像")
        
        # 每下载5张图像后休息一下
        if (i + 1) % 5 == 0:
            time.sleep(random.uniform(2.0, 3.0))
    
    final_count = existing_count + downloaded_count
    print(f"{series}_{character_name} 采集完成，当前共有 {final_count} 张图像")
    return final_count

def main():
    parser = argparse.ArgumentParser(description='简单的角色数据采集脚本')
    parser.add_argument('--target-count', type=int, default=60, help='每个角色的目标图像数')
    args = parser.parse_args()
    
    total_collected = 0
    
    for character in LOW_ACCURACY_CHARACTERS:
        series = character['series']
        character_name = character['name']
        target_count = character.get('target_count', args.target_count)
        
        print(f"\n=== 采集角色: {series}_{character_name} ===")
        count = collect_character_images(series, character_name, target_count)
        total_collected += count
        
        # 角色之间增加延迟
        time.sleep(random.uniform(3.0, 5.0))
    
    print(f"\n=== 采集完成 ===")
    print(f"总共采集了 {total_collected} 张图像")

if __name__ == '__main__':
    main()