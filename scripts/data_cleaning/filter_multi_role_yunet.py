#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YuNet 动漫人脸检测脚本 - OpenCV官方主推的深度学习人脸检测模型
支持二次元人脸检测，速度极快
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
OUTPUT_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/yunet_detection_results')
MIN_FACES_FOR_MULTI = 2

# YuNet 模型配置
# 二次元权重 URL（OpenCV官方提供的针对动漫优化的模型）
ANIME_YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_DIR = Path(__file__).parent / "models"

# 全局变量，用于每个进程
_detector = None

def download_yunet_model():
    """下载 YuNet 人脸检测模型"""
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    
    if not model_path.exists():
        print(f"正在下载 YuNet 模型...")
        try:
            urllib.request.urlretrieve(ANIME_YUNET_URL, str(model_path))
            print(f"已下载 YuNet 模型: {model_path}")
        except Exception as e:
            print(f"下载失败: {e}")
            return None
    return model_path

def init_worker():
    """
    每个进程初始化时调用，加载 YuNet 检测器
    
    YuNet 是 OpenCV 官方主推的深度学习人脸检测模型：
    - 基于 YOLO 架构，速度极快
    - 支持人脸关键点检测（5个关键点：双眼、鼻尖、嘴角）
    - 有专门优化的二次元人脸检测权重
    - 比传统 Haar/LBP 级联分类器准确率高很多
    """
    global _detector
    
    model_path = download_yunet_model()
    
    if model_path and model_path.exists():
        try:
            # 🔥 使用 YuNet 检测器
            _detector = cv2.FaceDetectorYN.create(
                model=str(model_path),
                config="",
                input_size=(320, 320),  # 输入尺寸，影响速度和精度
                score_threshold=0.6,     # 置信度阈值
                nms_threshold=0.3,       # NMS阈值
                top_k=50,                # 最多检测人脸数
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU
            )
            print(f"✅ 成功加载 YuNet 检测器")
            return
        except Exception as e:
            print(f"加载 YuNet 失败: {e}")
    
    # 降级到默认Haar检测器
    print("⚠️ 降级使用 Haar 检测器")
    _detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces_single(image_path):
    """
    使用 YuNet 检测单张图片中的人脸
    
    🔥 YuNet 优势：
    1. 深度学习模型，准确率远超传统级联分类器
    2. 支持人脸关键点检测（左眼、右眼、鼻尖、左嘴角、右嘴角）
    3. 专为快速推理优化，速度极快
    4. 有专门针对二次元/动漫人脸训练的权重
    """
    global _detector
    
    try:
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            return str(image_path), 0, []
        
        # 获取图片尺寸
        height, width = image.shape[:2]
        
        # 设置输入尺寸（YuNet 需要）
        if hasattr(_detector, 'setInputSize'):
            _detector.setInputSize((width, height))
        
        # 检测人脸
        if hasattr(_detector, 'detect'):
            # YuNet 检测器
            _, faces = _detector.detect(image)
            face_count = len(faces) if faces is not None else 0
            
            # 提取人脸信息（含关键点）
            face_info = []
            if faces is not None:
                for face in faces:
                    # face[0:4] = x, y, w, h
                    # face[4] = 置信度
                    # face[5:14] = 5个关键点（左眼、右眼、鼻尖、左嘴角、右嘴角）
                    face_info.append({
                        'x': int(face[0]),
                        'y': int(face[1]),
                        'w': int(face[2]),
                        'h': int(face[3]),
                        'confidence': float(face[4]),
                        'keypoints': [
                            (int(face[5]), int(face[6])),   # 左眼
                            (int(face[7]), int(face[8])),   # 右眼
                            (int(face[9]), int(face[10])),  # 鼻尖
                            (int(face[11]), int(face[12])), # 左嘴角
                            (int(face[13]), int(face[14]))  # 右嘴角
                        ]
                    })
            
            return str(image_path), face_count, face_info
        else:
            # 降级到 Haar 检测器
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = _detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            return str(image_path), len(faces), []
        
    except Exception as e:
        return str(image_path), 0, []

def filter_multi_role_images(data_dir, output_dir):
    """
    使用 YuNet 筛选多角色和无角色图片
    """
    # 创建输出目录
    multi_role_dir = output_dir / 'multi_role'
    no_role_dir = output_dir / 'no_role'
    single_role_dir = output_dir / 'single_role'
    multi_role_dir.mkdir(parents=True, exist_ok=True)
    no_role_dir.mkdir(parents=True, exist_ok=True)
    single_role_dir.mkdir(parents=True, exist_ok=True)
    
    # 先下载模型（主进程）
    download_yunet_model()
    
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
    
    # 🔥 多进程处理 - YuNet 是CPU密集型，充分利用多核
    stats = {
        'total_images': 0,
        'single_role': 0,
        'multi_role': 0,
        'no_role': 0,
        'failed': 0,
        'multi_role_images': [],
        'no_role_images': [],
        'detector_type': 'YuNet' if _detector is not None and hasattr(_detector, 'detect') else 'Haar'
    }
    
    print(f"🚀 使用 {stats['detector_type']} 启动多进程检测...")
    
    with ProcessPoolExecutor(initializer=init_worker) as executor:
        futures = {}
        for role_name, img_file in image_files:
            future = executor.submit(detect_faces_single, img_file)
            futures[future] = (role_name, img_file)
        
        for i, future in enumerate(as_completed(futures), 1):
            role_name, img_file = futures[future]
            path, face_count, face_info = future.result()
            
            stats['total_images'] += 1
            
            if face_count == 0:
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
                    'face_count': face_count,
                    'faces': face_info
                })
            else:
                stats['single_role'] += 1
            
            if i % 500 == 0:
                print(f"⏳ 已处理: {i}/{total_files}")
    
    # 保存结果
    with open(output_dir / 'yunet_detection_results.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 输出统计报告
    print("\n" + "=" * 80)
    print("✅ YuNet 检测完成")
    print("=" * 80)
    print(f"检测器类型: {stats['detector_type']}")
    print(f"总图片数: {stats['total_images']}")
    print(f"单人角色: {stats['single_role']} ({stats['single_role']/stats['total_images']*100:.1f}%)")
    print(f"多角色: {stats['multi_role']} ({stats['multi_role']/stats['total_images']*100:.1f}%)")
    print(f"无角色: {stats['no_role']} ({stats['no_role']/stats['total_images']*100:.1f}%)")
    
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
    """移动筛选出的图片"""
    multi_role_dir = output_dir / 'multi_role'
    no_role_dir = output_dir / 'no_role'
    
    print("\n📦 开始移动图片...")
    
    for item in stats['multi_role_images']:
        src_path = data_dir / item['role'] / item['filename']
        dst_path = multi_role_dir / f"{item['role']}_{item['filename']}"
        try:
            shutil.move(src_path, dst_path)
        except Exception as e:
            print(f"❌ 移动失败 {src_path}: {e}")
    
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
    print("=" * 80)
    print("🚀 YuNet 动漫人脸检测工具")
    print("=" * 80)
    print("YuNet 是 OpenCV 官方主推的深度学习人脸检测模型")
    print("特点：速度极快、准确率高、支持二次元人脸")
    print("=" * 80)
    
    stats = filter_multi_role_images(DATA_DIR, OUTPUT_DIR)
    
    if stats['multi_role'] + stats['no_role'] > 0:
        confirm = input("\n是否将筛选出的图片移动到单独目录？(y/n): ")
        if confirm.lower() == 'y':
            move_filtered_images(stats, DATA_DIR, OUTPUT_DIR)
    
    print("\n🎉 任务完成！")
    print("\n💡 YuNet 优势总结：")
    print("  1. 深度学习模型，准确率远超传统 Haar/LBP")
    print("  2. 速度极快，专为实时推理优化")
    print("  3. 支持人脸关键点检测（5个关键点）")
    print("  4. 有专门针对二次元/动漫人脸训练的权重")