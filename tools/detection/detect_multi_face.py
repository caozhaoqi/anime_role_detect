#!/usr/bin/env python3
"""多角色图片检测脚本 - 使用OpenCV进行人脸检测"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import shutil
import json

DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "combined_dataset"
MULTI_FACE_DIR = Path(__file__).parent.parent.parent / "data" / "multi_face_detected"

def get_face_detector():
    """获取人脸检测器 - 优先使用DNN，fallback到Haar"""
    dnn_model = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    dnn_weights = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not cascade.empty():
            return ("Haar", cascade)
    except:
        pass

    return None

def detect_faces_haar(cascade, image):
    """使用Haar级联检测人脸"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    return [tuple(f) for f in faces]

def detect_faces(image_path, detector_info):
    """检测图片中的人脸数量"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return None, 0, []

        detector_type, detector = detector_info
        faces = detect_faces_haar(detector, image)

        return str(image_path), len(faces), faces
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return str(image_path), 0, []

def process_dataset(dataset_path, output_path, min_faces=2, max_workers=8, sample_limit=None):
    """处理整个数据集，检测多角色图片"""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(dataset_path.rglob(f'*{ext}'))

    if sample_limit:
        image_files = image_files[:sample_limit]

    print(f"找到 {len(image_files)} 张图片")
    print(f"将检测包含 {min_faces} 张或多张人脸的图片...")

    detector_info = get_face_detector()
    if detector_info is None:
        print("错误: 无法加载人脸检测器")
        return

    print(f"使用 {detector_info[0]} 方式进行检测")

    multi_face_images = []
    single_face_images = []
    no_face_images = []
    error_images = []

    total = len(image_files)
    for i, img in enumerate(image_files):
        path, face_count, _ = detect_faces(img, detector_info)
        if path is None:
            error_images.append(path)
        elif face_count >= min_faces:
            multi_face_images.append((path, face_count))
        elif face_count == 1:
            single_face_images.append((path, face_count))
        else:
            no_face_images.append((path, face_count))

        if (i + 1) % 100 == 0:
            print(f"已处理: {i + 1}/{total} | 多脸: {len(multi_face_images)} | 单脸: {len(single_face_images)} | 无脸: {len(no_face_images)}")

    print(f"\n处理完成!")
    print(f"=" * 50)
    print(f"总图片数: {len(image_files)}")
    print(f"多角色图片 (>=2人脸): {len(multi_face_images)}")
    print(f"单角色图片 (=1人脸): {len(single_face_images)}")
    print(f"无检测到人脸: {len(no_face_images)}")
    print(f"处理错误: {len(error_images)}")

    multi_face_dir = output_path / "multi_face"
    multi_face_dir.mkdir(exist_ok=True)

    if multi_face_images:
        print(f"\n正在复制多角色图片到 {multi_face_dir}...")
        for i, (img_path, face_count) in enumerate(sorted(multi_face_images, key=lambda x: x[1], reverse=True)):
            src = Path(img_path)
            dst = multi_face_dir / f"face_{face_count}_{src.name}"
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"复制失败 {src}: {e}")

        with open(output_path / "multi_face_images.txt", "w", encoding="utf-8") as f:
            for img_path, face_count in sorted(multi_face_images, key=lambda x: x[1], reverse=True):
                f.write(f"{face_count}\t{img_path}\n")

        print(f"已保存 {len(multi_face_images)} 张多角色图片到 {multi_face_dir}")

    summary = {
        "total_images": len(image_files),
        "multi_face_count": len(multi_face_images),
        "single_face_count": len(single_face_images),
        "no_face_count": len(no_face_images),
        "error_count": len(error_images),
        "min_faces_threshold": min_faces
    }

    with open(output_path / "detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n结果摘要已保存到 {output_path / 'detection_summary.json'}")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多角色图片检测工具")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="数据集路径")
    parser.add_argument("--output", type=str, default=str(MULTI_FACE_DIR), help="输出路径")
    parser.add_argument("--min-faces", type=int, default=2, help="最小人脸数量 (默认: 2)")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数 (默认: 8)")
    parser.add_argument("--sample", type=int, default=None, help="仅处理前N张图片 (测试用)")

    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"错误: 数据集路径不存在: {args.dataset}")
        sys.exit(1)

    process_dataset(args.dataset, args.output, args.min_faces, args.workers, args.sample)