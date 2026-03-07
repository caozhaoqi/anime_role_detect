#!/usr/bin/env python3
"""
蔚蓝档案阿罗娜和普拉娜专用采集脚本
针对这两个角色进行优化采集
"""
import os
import sys
import shutil
import requests
import time
import random
from PIL import Image
from io import BytesIO

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "image/*"
}

# 阿罗娜和普拉娜的搜索标签配置
CHARACTERS_CONFIG = {
    "蔚蓝档案_阿罗娜": {
        "tags": [
            "arona_(blue_archive)",
            "アロナ(ブルーアーカイブ)",
            "blue_archive_arona",
            "arona_blue_archive_solo"
        ],
        "target_count": 100,
        "description": "阿罗娜 - 蔚蓝档案看板娘，蓝色短发，有光环"
    },
    "蔚蓝档案_普拉娜": {
        "tags": [
            "plana_(blue_archive)",
            "プラナ(ブルーアーカイブ)",
            "blue_archive_plana",
            "plana_blue_archive_solo"
        ],
        "target_count": 100,
        "description": "普拉娜 - 蔚蓝档案角色，黑色长发，有光环"
    }
}


def validate_image(content):
    """验证图像是否有效"""
    try:
        img = Image.open(BytesIO(content))
        img.verify()
        return True
    except:
        return False


def download_image(url, save_path, min_size=(512, 512)):
    """下载图像并进行验证"""
    try:
        # 随机延迟，避免被封禁
        time.sleep(random.uniform(0.5, 1.5))
        
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            content = response.content
            
            # 验证图像
            if not validate_image(content):
                return False
            
            # 检查图像尺寸
            try:
                img = Image.open(BytesIO(content))
                if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                    print(f"  图像尺寸太小: {img.size}")
                    return False
            except:
                return False
            
            # 保存图像
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"  下载失败: {e}")
    return False


def collect_from_danbooru(character_name, tags, target_count, output_dir):
    """从Danbooru采集图像"""
    print(f"\n=== 从Danbooru采集 {character_name} ===")
    
    collected = 0
    existing_files = set()
    
    for tag in tags:
        if collected >= target_count:
            break
        
        print(f"\n搜索标签: {tag}")
        
        try:
            import urllib.parse
            encoded_tag = urllib.parse.quote(tag)
            api_url = f"https://danbooru.donmai.us/posts.json?limit=100&tags={encoded_tag}+rating:s"
            
            response = requests.get(api_url, headers=HEADERS, timeout=15)
            if response.status_code != 200:
                print(f"  API请求失败: {response.status_code}")
                continue
            
            posts = response.json()
            print(f"  找到 {len(posts)} 个结果")
            
            for post in posts:
                if collected >= target_count:
                    break
                
                if 'file_url' not in post:
                    continue
                
                image_url = post['file_url']
                
                # 去重检查
                if image_url in existing_files:
                    continue
                existing_files.add(image_url)
                
                # 生成保存路径
                file_ext = os.path.splitext(image_url)[1]
                if not file_ext:
                    file_ext = ".jpg"
                
                save_path = os.path.join(output_dir, f"{character_name}_{collected:04d}{file_ext}")
                
                if download_image(image_url, save_path):
                    collected += 1
                    print(f"  已下载 {collected}/{target_count}: {os.path.basename(save_path)}")
                
                # 每下载5张休息一下
                if collected % 5 == 0:
                    time.sleep(random.uniform(2, 3))
            
        except Exception as e:
            print(f"  采集出错: {e}")
        
        # 标签之间延迟
        time.sleep(random.uniform(2, 4))
    
    return collected


def collect_from_safebooru(character_name, tags, target_count, output_dir):
    """从Safebooru采集图像"""
    print(f"\n=== 从Safebooru采集 {character_name} ===")
    
    collected = 0
    existing_files = set()
    
    for tag in tags:
        if collected >= target_count:
            break
        
        print(f"\n搜索标签: {tag}")
        
        try:
            import urllib.parse
            encoded_tag = urllib.parse.quote(tag)
            api_url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&limit=100&tags={encoded_tag}"
            
            response = requests.get(api_url, headers=HEADERS, timeout=15)
            if response.status_code != 200:
                print(f"  API请求失败: {response.status_code}")
                continue
            
            # 解析XML响应
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            posts = root.findall('.//post')
            print(f"  找到 {len(posts)} 个结果")
            
            for post in posts:
                if collected >= target_count:
                    break
                
                image_url = post.get('file_url')
                if not image_url:
                    continue
                
                # 去重检查
                if image_url in existing_files:
                    continue
                existing_files.add(image_url)
                
                # 生成保存路径
                file_ext = os.path.splitext(image_url)[1]
                if not file_ext:
                    file_ext = ".jpg"
                
                save_path = os.path.join(output_dir, f"{character_name}_{collected:04d}{file_ext}")
                
                if download_image(image_url, save_path):
                    collected += 1
                    print(f"  已下载 {collected}/{target_count}: {os.path.basename(save_path)}")
                
                # 每下载5张休息一下
                if collected % 5 == 0:
                    time.sleep(random.uniform(2, 3))
            
        except Exception as e:
            print(f"  采集出错: {e}")
        
        # 标签之间延迟
        time.sleep(random.uniform(2, 4))
    
    return collected


def collect_character_data(character_name, config, output_base_dir):
    """采集单个角色的数据"""
    print(f"\n{'='*60}")
    print(f"开始采集角色: {character_name}")
    print(f"描述: {config['description']}")
    print(f"目标数量: {config['target_count']} 张")
    print(f"{'='*60}")
    
    # 创建输出目录
    output_dir = os.path.join(output_base_dir, character_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查已有数据
    existing_images = [f for f in os.listdir(output_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    existing_count = len(existing_images)
    
    print(f"已有数据: {existing_count} 张")
    
    if existing_count >= config['target_count']:
        print(f"✅ {character_name} 数据已充足，跳过采集")
        return existing_count
    
    needed_count = config['target_count'] - existing_count
    print(f"需要补充: {needed_count} 张")
    
    tags = config['tags']
    total_collected = existing_count
    
    # 从Danbooru采集
    danbooru_count = collect_from_danbooru(
        character_name, 
        tags, 
        needed_count, 
        output_dir
    )
    total_collected += danbooru_count
    print(f"\n从Danbooru采集了 {danbooru_count} 张")
    
    # 如果还不够，从Safebooru补充
    if total_collected < config['target_count']:
        remaining = config['target_count'] - total_collected
        print(f"\n还需要 {remaining} 张，尝试从Safebooru补充...")
        
        safebooru_count = collect_from_safebooru(
            character_name,
            tags,
            remaining,
            output_dir
        )
        total_collected += safebooru_count
        print(f"\n从Safebooru采集了 {safebooru_count} 张")
    
    print(f"\n{'='*60}")
    print(f"{character_name} 采集完成")
    print(f"总计: {total_collected} 张图像")
    print(f"{'='*60}")
    
    return total_collected


def main():
    """主函数"""
    print("="*60)
    print("蔚蓝档案阿罗娜和普拉娜专用采集脚本")
    print("="*60)
    
    # 配置
    output_base_dir = "../data/train"
    
    results = {}
    
    # 采集每个角色
    for character_name, config in CHARACTERS_CONFIG.items():
        count = collect_character_data(character_name, config, output_base_dir)
        results[character_name] = count
        
        # 角色之间延迟
        time.sleep(random.uniform(5, 8))
    
    # 汇总报告
    print("\n" + "="*60)
    print("采集完成汇总")
    print("="*60)
    for character, count in results.items():
        target = CHARACTERS_CONFIG[character]['target_count']
        status = "✅ 完成" if count >= target else "⚠️ 部分完成"
        print(f"{character}: {count}/{target} 张 {status}")
    print("="*60)


if __name__ == "__main__":
    main()
