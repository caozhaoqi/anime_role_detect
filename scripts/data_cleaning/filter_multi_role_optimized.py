#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版多角色检测脚本 - 修复多进程pickle问题
"""
import os
import cv2
import json
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import urllib.request

# 配置
DATA_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/combined_dataset')
OUTPUT_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/filtered_results_optimized')
MIN_FACES_FOR_MULTI = 2

# 动漫人脸检测器URL
ANIME_CASCADE_URL = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
CASCADE_DIR = Path(__file__).parent / "cascades"

# 全局变量，用于每个进程
_detector = None

def download_anime_cascade():
    """下载动漫人脸检测器"""
    CASCADE_DIR.mkdir(exist_ok=True)
    cascade_path = CASCADE_DIR / "lbpcascade_animeface.xml"
    
    if not cascade_path.exists():
        print(f"正在下载动漫人脸检测器...")
        try:
            urllib.request.urlretrieve(ANIME_CASCADE_URL, str(cascade_path))
            print(f"已下载动漫人脸检测器: {cascade_path}")
        except Exception as e:
            print(f"下载失败: {e}")
            return None
    return cascade_path

def init_worker():
    """
    每个进程初始化时调用，加载检测器
    🔥 关键修复：cv2.CascadeClassifier无法被pickle，
    因此每个进程需要独立加载
    """
    global _detector
    
    cascade_path = download_anime_cascade()
    
    if cascade_path and cascade_path.exists():
        try:
            _detector = cv2.CascadeClassifier(str(cascade_path))
            if _detector.empty():
                _detector = None
            return
        except Exception as e:
            pass
    
    # 降级到默认Haar检测器
    _detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces_single(image_path):
    """
    处理单张图片 - 供多进程调用
    
    🔥 优化点：
    1. cv2.equalizeHist(gray) - 直方图均衡化，增强人脸特征
    2. scaleFactor=1.1 - 动漫角色比例多变，需要更小步进
    """
    global _detector
    
    try:
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            return str(image_path), 0, False
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 🔥 直方图均衡化 - 关键优化
        # 动漫图片的对比度分布与真人照片完全不同
        # 通过直方图均衡化，可以让色彩平淡或过曝的背景中，人脸特征更加突出
        gray = cv2.equalizeHist(gray)
        
        # 🔥 scaleFactor=1.1 - 关键优化
        # 动漫识别需要比真人识别更小的步进
        # 动漫角色比例多变，这个值可以平衡速度与精度
        faces = _detector.detectMultiScale(
            gray,
            scaleFactor=1.1,        # 更小的步进，适应动漫角色比例多变
            minNeighbors=4,         # 更高的邻居阈值，减少误检
            minSize=(20, 20),       # 最小人脸尺寸
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return str(image_path), len(faces), True
        
    except Exception as e:
        return str(image_path), 0, False

def filter_multi_role_images(data_dir, output_dir):
    """
    筛选多角色和无角色图片 - 优化版
    """
    # 创建输出目录
    multi_role_dir = output_dir / 'multi_role'
    no_role_dir = output_dir / 'no_role'
    multi_role_dir.mkdir(parents=True, exist_ok=True)
    no_role_dir.mkdir(parents=True, exist_ok=True)
    
    # 先下载检测器（主进程）
    download_anime_cascade()
    
    # 收集所有图片文件
    image_files = []
    for role_dir in data_dir.iterdir():
        if not role_dir.is_dir() or role_dir.name.startswith('.'):
            continue
        
        role_name = role_dir.name
        for img_file in role_dir.glob('*.jpg'):
            image_files.append((role_name, img_file))
    
    total_files = len(image_files)
    print(f"📂 共找到 {total_files} 张图片")
    print("=" * 80)
    
    # 🔥 多进程处理 - 关键优化
    # OpenCV的识别是CPU密集型任务，Python的GIL会限制多线程
    # 使用ProcessPoolExecutor可以真正利用多核CPU
    stats = {
        'total_images': 0,
        'single_role': 0,
        'multi_role': 0,
        'no_role': 0,
        'failed': 0,
        'multi_role_images': [],
        'no_role_images': []
    }
    
    print(f"🚀 启动多进程检测 (CPU核心全部利用)...")
    
    with ProcessPoolExecutor(initializer=init_worker) as executor:
        # 提交所有任务
        futures = {}
        for role_name, img_file in image_files:
            # 只传递路径，不传递检测器对象
            future = executor.submit(detect_faces_single, img_file)
            futures[future] = (role_name, img_file)
        
        # 处理完成的任务
        for i, future in enumerate(as_completed(futures), 1):
            role_name, img_file = futures[future]
            path, face_count, success = future.result()
            
            stats['total_images'] += 1
            
            if not success:
                stats['failed'] += 1
            elif face_count == 0:
                stats['no_role'] += 1
                stats['no_role_images'].append({
                    'role': role_name,
                    'filename': img_file.name,
                    'path': str(img_file)
                })
            elif face_count >= MIN_FACES_FOR_MULTI:
                stats['multi_role'] += 1
                stats['multi_role_images'].append({
                    'role': role_name,
                    'filename': img_file.name,
                    'path': str(img_file),
                    'face_count': face_count
                })
            else:
                stats['single_role'] += 1
            
            # 进度显示
            if i % 500 == 0:
                print(f"⏳ 已处理: {i}/{total_files}")
    
    # 保存结果
    with open(output_dir / 'filter_results.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 输出统计报告
    print("\n" + "=" * 80)
    print("✅ 检测完成")
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
        print(f"\n📄 多角色图片列表已保存")
    
    if stats['no_role_images']:
        with open(output_dir / 'no_role_list.txt', 'w', encoding='utf-8') as f:
            for item in stats['no_role_images']:
                f.write(f"{item['role']}/{item['filename']}\n")
        print(f"📄 无角色图片列表已保存")
    
    return stats

def move_filtered_images(stats, data_dir, output_dir):
    """
    移动筛选出的图片到对应目录
    
    🔥 归档策略优化：
    将文件名改成 {role_name}_{original_filename}
    这是因为不同角色的目录下可能有同名的图片，移动到同一个目录时会发生覆盖
    """
    multi_role_dir = output_dir / 'multi_role'
    no_role_dir = output_dir / 'no_role'
    
    print("\n📦 开始移动图片...")
    
    # 移动多角色图片
    for item in stats['multi_role_images']:
        src_path = data_dir / item['role'] / item['filename']
        # 🔥 使用 {role_name}_{original_filename} 避免覆盖
        dst_path = multi_role_dir / f"{item['role']}_{item['filename']}"
        try:
            shutil.move(src_path, dst_path)
        except Exception as e:
            print(f"❌ 移动失败 {src_path}: {e}")
    
    # 移动无角色图片
    for item in stats['no_role_images']:
        src_path = data_dir / item['role'] / item['filename']
        dst_path = no_role_dir / f"{item['role']}_{item['filename']}"
        try:
            shutil.move(src_path, dst_path)
        except Exception as e:
            print(f"❌ 移动失败 {src_path}: {e}")
    
    print(f"✅ 已移动 {len(stats['multi_role_images'])} 张多角色图片")
    print(f"✅ 已移动 {len(stats['no_role_images'])} 张无角色图片")

if __name__ == "__main__":
    # 执行筛选
    stats = filter_multi_role_images(DATA_DIR, OUTPUT_DIR)
    
    # 询问是否移动图片
    if stats['multi_role'] + stats['no_role'] > 0:
        confirm = input("\n是否将筛选出的图片移动到单独目录？(y/n): ")
        if confirm.lower() == 'y':
            move_filtered_images(stats, DATA_DIR, OUTPUT_DIR)
    
    print("\n🎉 任务完成！")
    print("\n💡 优化要点总结：")
    print("  1. cv2.equalizeHist(gray) - 提升动漫人脸检测效果")
    print("  2. scaleFactor=1.1 - 平衡速度与精度")
    print("  3. ProcessPoolExecutor + initializer - 真正多核并行")
    print("  4. {role_name}_{filename} - 避免文件覆盖")