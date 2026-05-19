#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测优化脚本 - 调整阈值重新检测，召回更多样本
"""

import os
import cv2
import json
import shutil
from collections import defaultdict


def optimize_detection(input_dir, output_dir, score_threshold=0.3, nms_threshold=0.3):
    """
    使用更低的阈值重新检测图片
    
    Args:
        input_dir: 输入图片目录
        output_dir: 输出目录
        score_threshold: 置信度阈值（降低可召回更多）
        nms_threshold: NMS阈值
    """
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'single_face'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'multi_face'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'no_face'), exist_ok=True)
    
    # 初始化 YuNet 检测器
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'face_detection_yunet_2023mar.onnx')
    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config='',
        input_size=(320, 320),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=500
    )
    
    # 统计变量
    total_images = 0
    single_count = 0
    multi_count = 0
    no_face_count = 0
    
    # 遍历所有角色目录
    for role in os.listdir(input_dir):
        role_dir = os.path.join(input_dir, role)
        if not os.path.isdir(role_dir):
            continue
        
        for filename in os.listdir(role_dir):
            if not filename.endswith('.jpg'):
                continue
            
            image_path = os.path.join(role_dir, filename)
            
            try:
                # 读取图片
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                total_images += 1
                
                # 调整检测器输入尺寸
                h, w = img.shape[:2]
                detector.setInputSize((w, h))
                
                # 检测人脸
                _, faces = detector.detect(img)
                
                if faces is None:
                    # 未检测到人脸
                    shutil.copy(image_path, os.path.join(output_dir, 'no_face', filename))
                    no_face_count += 1
                elif len(faces) == 1:
                    # 单人脸
                    shutil.copy(image_path, os.path.join(output_dir, 'single_face', filename))
                    single_count += 1
                else:
                    # 多人脸
                    shutil.copy(image_path, os.path.join(output_dir, 'multi_face', filename))
                    multi_count += 1
                
                if total_images % 100 == 0:
                    print(f"🔄 已处理 {total_images} 张图片...")
                    
            except Exception as e:
                print(f"❌ 处理 {filename} 失败: {e}")
    
    # 生成报告
    report = {
        'total_images': total_images,
        'single_face': single_count,
        'multi_face': multi_count,
        'no_face': no_face_count,
        'score_threshold': score_threshold,
        'nms_threshold': nms_threshold,
        'single_ratio': single_count / total_images,
        'multi_ratio': multi_count / total_images,
        'no_face_ratio': no_face_count / total_images
    }
    
    with open(os.path.join(output_dir, 'detection_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("📊 检测完成")
    print("="*60)
    print(f"总图片数: {total_images}")
    print(f"单人脸: {single_count} ({single_count/total_images*100:.2f}%)")
    print(f"多人脸: {multi_count} ({multi_count/total_images*100:.2f}%)")
    print(f"无人脸: {no_face_count} ({no_face_count/total_images*100:.2f}%)")
    print(f"\n✅ 报告已保存: {os.path.join(output_dir, 'detection_report.json')}")
    
    return report


def crop_multi_face_images(multi_face_dir, output_dir, padding_ratio=0.2):
    """
    切割多人脸图片
    
    Args:
        multi_face_dir: 多人脸图片目录
        output_dir: 输出目录
        padding_ratio: 扩充比例
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'face_detection_yunet_2023mar.onnx')
    detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config='',
        input_size=(320, 320),
        score_threshold=0.3,
        nms_threshold=0.3,
        top_k=500
    )
    
    crop_count = 0
    
    for filename in os.listdir(multi_face_dir):
        if not filename.endswith('.jpg'):
            continue
        
        image_path = os.path.join(multi_face_dir, filename)
        img = cv2.imread(image_path)
        
        if img is None:
            continue
        
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        
        _, faces = detector.detect(img)
        
        if faces is not None and len(faces) > 1:
            for i, face in enumerate(faces):
                x, y, width, height = face[:4]
                
                # 扩充边界
                padding_x = int(width * padding_ratio)
                padding_y = int(height * padding_ratio)
                
                x1 = max(0, int(x) - padding_x)
                y1 = max(0, int(y) - padding_y)
                x2 = min(w, int(x) + int(width) + padding_x)
                y2 = min(h, int(y) + int(height) + padding_y)
                
                # 切割人脸区域
                face_crop = img[y1:y2, x1:x2]
                
                # 保存切割后的图片
                base_name = os.path.splitext(filename)[0]
                crop_filename = f"{base_name}_crop_{i}.jpg"
                cv2.imwrite(os.path.join(output_dir, crop_filename), face_crop)
                crop_count += 1
    
    print(f"\n✅ 切割完成，共生成 {crop_count} 张单人图片")
    return crop_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="检测优化工具")
    parser.add_argument("--input_dir", type=str, required=True, help="输入数据目录")
    parser.add_argument("--output_dir", type=str, default="./optimized_detection", help="输出目录")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--nms_threshold", type=float, default=0.3, help="NMS阈值")
    parser.add_argument("--crop_multi", action="store_true", help="是否切割多人脸图片")
    
    args = parser.parse_args()
    
    print("🚀 开始检测优化")
    print(f"阈值设置: score_threshold={args.score_threshold}, nms_threshold={args.nms_threshold}")
    
    # 执行检测
    report = optimize_detection(
        args.input_dir,
        args.output_dir,
        args.score_threshold,
        args.nms_threshold
    )
    
    # 如果需要，切割多人脸图片
    if args.crop_multi:
        multi_face_dir = os.path.join(args.output_dir, 'multi_face')
        cropped_dir = os.path.join(args.output_dir, 'multi_face_cropped')
        crop_multi_face_images(multi_face_dir, cropped_dir)


if __name__ == "__main__":
    main()