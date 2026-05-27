#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充采集图片数量不足的角色
针对 final_dataset 中图片少于目标数量的角色进行补充采集
"""

import os
import sys
import subprocess
import time
import shutil
from pypinyin import lazy_pinyin, Style

# 配置
FINAL_DATASET = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset'
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/spider_image_system/data/img_url'
DOWNLOAD_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/downloaded_images'
ROLE_LIST_FILE = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/auto_spider_img/loli-role.txt'

MIN_IMAGES = 30  # 最少需要30张图片

def parse_role_list():
    """解析角色列表，建立映射"""
    role_mapping = {}  # 英文名称 -> 中文名称
    pinyin_mapping = {}  # 拼音 -> 英文名称
    
    with open(ROLE_LIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                chinese_name = parts[0]
                english_name = parts[2]
                pinyin = ''.join(lazy_pinyin(chinese_name, style=Style.TONE3))
                role_mapping[english_name] = chinese_name
                pinyin_mapping[pinyin] = english_name
    
    return role_mapping, pinyin_mapping

def get_low_count_roles(min_images=MIN_IMAGES):
    """获取图片数量不足的角色"""
    low_count = []
    
    for role_name in sorted(os.listdir(FINAL_DATASET)):
        role_dir = os.path.join(FINAL_DATASET, role_name)
        if not os.path.isdir(role_dir) or role_name.startswith('.'):
            continue
        
        img_count = len([f for f in os.listdir(role_dir) 
                        if f.lower().endswith('.jpg')])
        
        if img_count < min_images:
            low_count.append({
                'name': role_name,
                'count': img_count,
                'needed': min_images - img_count
            })
    
    low_count.sort(key=lambda x: x['count'])
    return low_count

def run_spider_for_role(chinese_name):
    """调用爬虫采集角色URL"""
    print(f"\n🔍 开始采集 {chinese_name} 的URL...")
    
    spider_script = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/data_collection/spider_single_role.py'
    
    if os.path.exists(spider_script):
        try:
            cmd = ['python3', spider_script, '--role', chinese_name]
            print(f"执行: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                if result.stdout:
                    print(result.stdout.strip())
                print(f"✅ {chinese_name} URL采集完成")
                return True
            else:
                print(f"⚠️ URL采集可能有问题")
                if result.stderr:
                    print(f"错误: {result.stderr[:200]}")
                return False
        except subprocess.TimeoutExpired:
            print(f"⏰ {chinese_name} URL采集超时")
            return False
        except Exception as e:
            print(f"❌ {chinese_name} URL采集失败: {e}")
            return False
    else:
        print(f"❌ 爬虫脚本不存在: {spider_script}")
        return False

def download_images_for_role(role_name, pinyin):
    """下载角色图片"""
    print(f"\n📥 开始下载 {role_name} 的图片...")
    
    url_file = os.path.join(URL_DIR, f"{pinyin}_img.txt")
    if not os.path.exists(url_file):
        print(f"❌ 未找到URL文件: {url_file}")
        return False
    
    download_script = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/data_collection/download_images.py'
    
    if os.path.exists(download_script):
        try:
            cmd = ['python3', download_script, '--role', role_name]
            print(f"执行: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        print(lines[-1])  # 打印最后一行
                print(f"✅ {role_name} 图片下载完成")
                return True
            else:
                print(f"⚠️ 图片下载可能有问题")
                if result.stderr:
                    print(f"错误: {result.stderr[:200]}")
                return False
        except subprocess.TimeoutExpired:
            print(f"⏰ {role_name} 图片下载超时")
            return False
        except Exception as e:
            print(f"❌ {role_name} 图片下载失败: {e}")
            return False
    else:
        print(f"❌ 下载脚本不存在: {download_script}")
        return False

def sync_to_final_dataset(role_name, pinyin):
    """将新下载的图片同步到final_dataset"""
    print(f"\n📤 同步到 final_dataset...")
    
    # 查找最新的下载批次
    batches = sorted([d for d in os.listdir(DOWNLOAD_DIR) if d.startswith('batch_')])
    if not batches:
        print(f"❌ 未找到下载批次")
        return False
    
    latest_batch = batches[-1]
    source_dir = os.path.join(DOWNLOAD_DIR, latest_batch, pinyin)
    
    if not os.path.exists(source_dir):
        print(f"❌ 未找到源目录: {source_dir}")
        return False
    
    target_dir = os.path.join(FINAL_DATASET, role_name)
    os.makedirs(target_dir, exist_ok=True)
    
    copied = 0
    for filename in os.listdir(source_dir):
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, filename)
        
        if os.path.isfile(src_path) and filename.lower().endswith('.jpg'):
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                copied += 1
    
    print(f"✅ 同步完成，新增 {copied} 张图片")
    return True

def main():
    print("="*70)
    print("🔄 补充采集图片数量不足的角色")
    print("="*70)
    
    role_mapping, pinyin_mapping = parse_role_list()
    
    # 获取需要补充的角色
    low_count_roles = get_low_count_roles()
    
    if not low_count_roles:
        print("🎉 所有角色图片数量都已达标！")
        return
    
    print(f"\n发现 {len(low_count_roles)} 个角色图片数量不足 ({MIN_IMAGES}张以下):")
    print("-"*70)
    for role in low_count_roles:
        chinese_name = role_mapping.get(role['name'], role['name'])
        print(f"  {role['name']:<15} ({chinese_name}): {role['count']} 张 (需要 {role['needed']} 张)")
    print("-"*70)
    
    # 逐个处理
    for role in low_count_roles:
        english_name = role['name']
        chinese_name = role_mapping.get(english_name, english_name)
        
        # 计算拼音（用于查找URL文件）
        pinyin = ''.join(lazy_pinyin(chinese_name, style=Style.TONE3))
        
        print(f"\n{'='*70}")
        print(f"处理: {english_name} ({chinese_name})")
        print(f"当前: {role['count']} 张, 需要补充: {role['needed']} 张")
        print("="*70)
        
        # 1. 检查是否已有URL文件
        url_file = os.path.join(URL_DIR, f"{pinyin}_img.txt")
        url_count = 0
        if os.path.exists(url_file):
            with open(url_file, 'r', encoding='utf-8') as f:
                url_count = len([l for l in f if l.strip()])
        
        # 2. 如果URL不足，采集URL
        if url_count < role['needed'] * 2:  # 需要多采集一些备用
            print(f"\n1️⃣ 步骤一: 采集URL")
            success = run_spider_for_role(chinese_name)
            if not success:
                print(f"⚠️ URL采集失败，跳过该角色")
                continue
            time.sleep(3)
        
        # 3. 下载图片
        print(f"\n2️⃣ 步骤二: 下载图片")
        download_images_for_role(chinese_name, pinyin)
        time.sleep(2)
        
        # 4. 同步到final_dataset
        print(f"\n3️⃣ 步骤三: 同步到数据集")
        sync_to_final_dataset(english_name, pinyin)
    
    print("\n" + "="*70)
    print("✅ 补充采集任务完成")
    print("="*70)

if __name__ == '__main__':
    main()
