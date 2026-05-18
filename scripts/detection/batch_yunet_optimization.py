#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量 YuNet 检测优化脚本

优化策略：
1. 使用 YuNet 替代 Haar 级联
2. 降低置信度阈值至 0.3-0.5
3. 自动切割多角色图片
4. 提高数据集利用率
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.detection.run_yunet_detection_optimized import YuNetFaceDetector


def batch_process_dataset(input_dir, output_base_dir, thresholds=[0.3, 0.5]):
    """
    批量处理数据集
    
    Args:
        input_dir: 输入目录
        output_base_dir: 输出基础目录
        thresholds: 置信度阈值列表
    """
    input_path = Path(input_dir)
    output_path = Path(output_base_dir)
    
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    single_face_dir = output_path / 'single_face'
    multi_face_dir = output_path / 'multi_face'
    multi_face_cropped_dir = output_path / 'multi_face_cropped'
    no_face_dir = output_path / 'no_face'
    
    for dir_path in [single_face_dir, multi_face_dir, multi_face_cropped_dir, no_face_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
    
    print(f"📂 找到 {len(image_files)} 张图像")
    
    # 统计结果
    stats = {
        'total': 0,
        'no_face': 0,
        'single_face': 0,
        'multi_face': 0,
        'cropped': 0
    }
    
    # 使用最优阈值
    detector = YuNetFaceDetector(score_threshold=0.3)
    
    for i, image_file in enumerate(image_files):
        if i % 100 == 0:
            print(f"\n{'='*60}")
            print(f"处理进度: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%)")
            print(f"当前统计: 无角色={stats['no_face']}, 单人={stats['single_face']}, 多人={stats['multi_face']}, 切割={stats['cropped']}")
            print('='*60)
        
        stats['total'] += 1
        filename = image_file.name
        
        try:
            # 检测人脸
            faces = detector.detect_faces(str(image_file))
            
            if len(faces) == 0:
                stats['no_face'] += 1
                # 复制到 no_face 目录
                os.system(f"cp '{image_file}' '{no_face_dir / filename}'")
                
            elif len(faces) == 1:
                stats['single_face'] += 1
                # 复制到 single_face 目录
                os.system(f"cp '{image_file}' '{single_face_dir / filename}'")
                
            else:
                stats['multi_face'] += 1
                
                # 复制原图到 multi_face 目录
                os.system(f"cp '{image_file}' '{multi_face_dir / filename}'")
                
                # 自动切割多角色图片
                cropped_files = auto_crop_faces(str(image_file), str(multi_face_cropped_dir), padding=0.3)
                stats['cropped'] += len(cropped_files)
                
        except Exception as e:
            print(f"❌ 处理失败 {filename}: {e}")
            stats['no_face'] += 1
    
    # 输出统计
    print(f"\n{'='*60}")
    print("📊 批量处理统计")
    print('='*60)
    print(f"总图像数: {stats['total']}")
    print(f"无角色: {stats['no_face']} ({stats['no_face']/stats['total']*100:.1f}%)")
    print(f"单人角色: {stats['single_face']} ({stats['single_face']/stats['total']*100:.1f}%)")
    print(f"多人角色: {stats['multi_face']} ({stats['multi_face']/stats['total']*100:.1f}%)")
    print(f"切割出的单人样本: {stats['cropped']}")
    print(f"有效产出率: {(stats['single_face'] + stats['cropped']) / stats['total'] * 100:.1f}%")
    print(f"数据利用率提升: {(stats['single_face'] + stats['cropped']) / stats['single_face'] * 100:.1f}% (相比仅单人角色)")
    
    # 保存统计报告
    report_path = output_path / 'detection_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"YuNet 检测优化报告\n")
        f.write(f"="*40 + "\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"输出目录: {output_base_dir}\n")
        f.write(f"检测阈值: 0.3\n")
        f.write(f"="*40 + "\n")
        f.write(f"总图像数: {stats['total']}\n")
        f.write(f"无角色: {stats['no_face']} ({stats['no_face']/stats['total']*100:.1f}%)\n")
        f.write(f"单人角色: {stats['single_face']} ({stats['single_face']/stats['total']*100:.1f}%)\n")
        f.write(f"多人角色: {stats['multi_face']} ({stats['multi_face']/stats['total']*100:.1f}%)\n")
        f.write(f"切割出的单人样本: {stats['cropped']}\n")
        f.write(f"有效产出率: {(stats['single_face'] + stats['cropped']) / stats['total'] * 100:.1f}%\n")
    
    print(f"\n✅ 报告已保存: {report_path}")


def auto_crop_faces(image_path, output_dir, padding=0.3):
    """
    自动切割多角色图片
    """
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    h, w = image.shape[:2]
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # 使用阈值 0.3 进行检测
    detector = YuNetFaceDetector(score_threshold=0.3)
    faces = detector.detect_faces(image_path)
    
    cropped_files = []
    
    for i, face in enumerate(faces):
        x, y, w_face, h_face = face['x'], face['y'], face['w'], face['h']
        
        # 扩充边界框
        pad_w = int(w_face * padding)
        pad_h = int(h_face * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + w_face + pad_w)
        y2 = min(h, y + h_face + pad_h // 2)
        
        # 切割图像
        cropped = image[y1:y2, x1:x2]
        
        # 保存切割后的图像
        output_filename = f"{name_without_ext}_crop_{i+1}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, cropped)
        cropped_files.append(output_path)
    
    return cropped_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量 YuNet 检测优化工具")
    parser.add_argument("--input", type=str, required=True, help="输入数据集目录")
    parser.add_argument("--output", type=str, default="./yunet_optimized_output", help="输出目录")
    
    args = parser.parse_args()
    
    print("🚀 开始批量 YuNet 检测优化")
    print(f"📥 输入: {args.input}")
    print(f"📤 输出: {args.output}")
    
    batch_process_dataset(args.input, args.output)