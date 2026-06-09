#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 final_dataset 中的图片合成为测试视频
用于验证视频识别功能是否正常工作
"""

import os
import sys
import cv2
import random
from pathlib import Path
from PIL import Image
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def collect_images(dataset_dir, max_images_per_role=5):
    """
    从数据集中收集图片
    
    Args:
        dataset_dir: 数据集目录
        max_images_per_role: 每个角色最多使用的图片数
        
    Returns:
        list: 图片路径列表
    """
    images = []
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return []
    
    # 遍历所有角色文件夹
    role_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"📂 找到 {len(role_dirs)} 个角色文件夹")
    
    for role_dir in role_dirs:
        # 获取该角色的所有图片
        image_files = list(role_dir.glob("*.jpg")) + list(role_dir.glob("*.png")) + list(role_dir.glob("*.webp"))
        
        if not image_files:
            continue
        
        # 随机选择指定数量的图片
        selected = random.sample(image_files, min(max_images_per_role, len(image_files)))
        images.extend(selected)
        print(f"  ✓ {role_dir.name}: {len(selected)} 张图片")
    
    print(f"\n✅ 共收集 {len(images)} 张图片")
    return images

def create_video_from_images(images, output_path, fps=1, duration_per_image=2):
    """
    将图片合成为视频
    
    Args:
        images: 图片路径列表
        output_path: 输出视频路径
        fps: 帧率
        duration_per_image: 每张图片显示的秒数
    """
    if not images:
        print("❌ 没有图片可以合成视频")
        return False
    
    print(f"\n🎬 开始合成视频...")
    print(f"   图片数量: {len(images)}")
    print(f"   帧率: {fps} FPS")
    print(f"   每张持续时间: {duration_per_image} 秒")
    
    # 读取第一张图片获取尺寸
    first_img = cv2.imread(str(images[0]))
    if first_img is None:
        print(f"❌ 无法读取图片: {images[0]}")
        return False
    
    height, width = first_img.shape[:2]
    print(f"   视频尺寸: {width}x{height}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("❌ 无法创建视频文件")
        return False
    
    # 计算每张图片需要的帧数
    frames_per_image = int(fps * duration_per_image)
    total_frames = len(images) * frames_per_image
    
    print(f"   总帧数: {total_frames}")
    print(f"   预计时长: {total_frames / fps:.1f} 秒")
    
    # 逐帧写入视频
    success_count = 0
    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  跳过无效图片: {img_path}")
            continue
        
        # 调整图片尺寸以匹配视频尺寸
        img_resized = cv2.resize(img, (width, height))
        
        # 写入多帧（让图片停留一段时间）
        for _ in range(frames_per_image):
            video_writer.write(img_resized)
        
        success_count += 1
        
        # 显示进度
        if i % 10 == 0 or i == len(images):
            print(f"   进度: {i}/{len(images)} ({i*100//len(images)}%)")
    
    video_writer.release()
    
    print(f"\n✅ 视频合成完成!")
    print(f"   成功处理: {success_count}/{len(images)} 张图片")
    print(f"   输出文件: {output_path}")
    
    # 验证视频文件
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"   文件大小: {file_size:.2f} MB")
        return True
    else:
        print("❌ 视频文件创建失败")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🎥 测试视频生成工具")
    print("=" * 60)
    
    # 配置
    dataset_dir = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset"
    output_dir = "/Users/caozhaoqi/PycharmProjects/anime_role_detect/test_videos"
    output_video = os.path.join(output_dir, "role_recognition_test.mp4")
    
    # 参数设置
    max_images_per_role = 5  # 每个角色最多使用5张图片
    fps = 1  # 帧率（每秒1帧，因为每张图片会重复多帧）
    duration_per_image = 2  # 每张图片显示2秒
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集图片
    print("\n📸 步骤1: 收集图片")
    print("-" * 60)
    images = collect_images(dataset_dir, max_images_per_role)
    
    if not images:
        print("\n❌ 没有找到图片，退出")
        return
    
    # 打乱图片顺序
    random.shuffle(images)
    
    # 合成视频
    print("\n🎬 步骤2: 合成视频")
    print("-" * 60)
    success = create_video_from_images(images, output_video, fps, duration_per_image)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 测试视频生成成功！")
        print("=" * 60)
        print(f"\n📹 视频位置: {output_video}")
        print(f"\n💡 使用方法:")
        print(f"   curl -X POST http://localhost:8002/video/recognize \\")
        print(f"     -F \"file=@{output_video}\" \\")
        print(f"     -F \"frame_interval=2.0\" \\")
        print(f"     -F \"confidence_threshold=0.3\" \\")
        print(f"     -F \"top_k=3\"")
    else:
        print("\n❌ 视频生成失败")

if __name__ == "__main__":
    main()
