#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选多角色图片和无角色图片的脚本
参考项目中的人脸检测代码实现
"""
import os
import cv2
import json
from pathlib import Path

# 配置
DATA_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset')
OUTPUT_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/filtered_results')
MIN_FACES_FOR_MULTI = 2  # 超过这个数量视为多角色
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

def detect_faces(image_path):
    """
    使用OpenCV检测图像中的人脸数量
    
    Args:
        image_path: 图像路径
        
    Returns:
        int: 检测到的人脸数量
    """
    try:
        # 加载人脸分类器
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            return -1  # 无法读取
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        # 参数调整：scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=3, 
            minSize=(20, 20),
            maxSize=(200, 200)
        )
        
        return len(faces)
    
    except Exception as e:
        print(f"❌ 检测失败 {image_path}: {e}")
        return -1

def filter_multi_and_no_role_images(data_dir, output_dir):
    """
    筛选多角色图片和无角色图片
    
    Args:
        data_dir: 数据集目录
        output_dir: 输出目录
    """
    # 创建输出目录
    multi_role_dir = output_dir / 'multi_role'
    no_role_dir = output_dir / 'no_role'
    multi_role_dir.mkdir(parents=True, exist_ok=True)
    no_role_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计结果
    stats = {
        'total_images': 0,
        'single_role': 0,
        'multi_role': 0,
        'no_role': 0,
        'failed': 0,
        'multi_role_images': [],
        'no_role_images': []
    }
    
    print("=" * 80)
    print("🔍 开始筛选多角色和无角色图片")
    print("=" * 80)
    
    # 遍历所有角色目录
    for role_dir in sorted(data_dir.iterdir()):
        if not role_dir.is_dir() or role_dir.name.startswith('.'):
            continue
        
        role_name = role_dir.name
        print(f"\n📂 处理角色: {role_name}")
        
        # 遍历角色目录下的所有图片
        for img_file in role_dir.glob('*.jpg'):
            stats['total_images'] += 1
            
            # 检测人脸数量
            face_count = detect_faces(img_file)
            
            if face_count == -1:
                stats['failed'] += 1
                print(f"  ❌ 无法读取: {img_file.name}")
            elif face_count == 0:
                stats['no_role'] += 1
                stats['no_role_images'].append({
                    'role': role_name,
                    'filename': img_file.name,
                    'path': str(img_file)
                })
                print(f"  ⚠️ 无角色: {img_file.name}")
            elif face_count >= MIN_FACES_FOR_MULTI:
                stats['multi_role'] += 1
                stats['multi_role_images'].append({
                    'role': role_name,
                    'filename': img_file.name,
                    'path': str(img_file),
                    'face_count': face_count
                })
                print(f"  ⚠️ 多角色({face_count}人): {img_file.name}")
            else:
                stats['single_role'] += 1
        
        # 进度显示
        if stats['total_images'] % 100 == 0:
            print(f"\n📊 已处理 {stats['total_images']} 张图片")
    
    # 保存结果
    with open(output_dir / 'filter_results.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 输出统计报告
    print("\n" + "=" * 80)
    print("✅ 筛选完成")
    print("=" * 80)
    print(f"总图片数: {stats['total_images']}")
    print(f"单人角色: {stats['single_role']} ({stats['single_role']/stats['total_images']*100:.1f}%)")
    print(f"多角色: {stats['multi_role']} ({stats['multi_role']/stats['total_images']*100:.1f}%)")
    print(f"无角色: {stats['no_role']} ({stats['no_role']/stats['total_images']*100:.1f}%)")
    print(f"检测失败: {stats['failed']}")
    
    # 保存详细列表
    if stats['multi_role_images']:
        with open(output_dir / 'multi_role_list.txt', 'w', encoding='utf-8') as f:
            for item in stats['multi_role_images']:
                f.write(f"{item['face_count']}人 - {item['role']}/{item['filename']}\n")
        print(f"\n📄 多角色图片列表已保存到: {output_dir / 'multi_role_list.txt'}")
    
    if stats['no_role_images']:
        with open(output_dir / 'no_role_list.txt', 'w', encoding='utf-8') as f:
            for item in stats['no_role_images']:
                f.write(f"{item['role']}/{item['filename']}\n")
        print(f"📄 无角色图片列表已保存到: {output_dir / 'no_role_list.txt'}")
    
    return stats

def move_filtered_images(stats, data_dir, output_dir):
    """
    将筛选出的图片移动到对应目录
    
    Args:
        stats: 统计结果
        data_dir: 数据集目录
        output_dir: 输出目录
    """
    multi_role_dir = output_dir / 'multi_role'
    no_role_dir = output_dir / 'no_role'
    
    print("\n📦 开始移动图片...")
    
    # 移动多角色图片
    for item in stats['multi_role_images']:
        src_path = data_dir / item['role'] / item['filename']
        dst_path = multi_role_dir / f"{item['role']}_{item['filename']}"
        try:
            os.rename(src_path, dst_path)
        except Exception as e:
            print(f"❌ 移动失败 {src_path}: {e}")
    
    # 移动无角色图片
    for item in stats['no_role_images']:
        src_path = data_dir / item['role'] / item['filename']
        dst_path = no_role_dir / f"{item['role']}_{item['filename']}"
        try:
            os.rename(src_path, dst_path)
        except Exception as e:
            print(f"❌ 移动失败 {src_path}: {e}")
    
    print(f"✅ 已移动 {len(stats['multi_role_images'])} 张多角色图片")
    print(f"✅ 已移动 {len(stats['no_role_images'])} 张无角色图片")

if __name__ == "__main__":
    # 执行筛选
    stats = filter_multi_and_no_role_images(DATA_DIR, OUTPUT_DIR)
    
    # 询问是否移动图片
    if stats['multi_role'] + stats['no_role'] > 0:
        confirm = input("\n是否将筛选出的图片移动到单独目录？(y/n): ")
        if confirm.lower() == 'y':
            move_filtered_images(stats, DATA_DIR, OUTPUT_DIR)
    
    print("\n🎉 任务完成！")