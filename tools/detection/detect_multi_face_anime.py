#!/usr/bin/env python3
"""多角色图片检测脚本 - 支持动漫人脸和黑白图片检测"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import shutil
import json
import urllib.request

DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "combined_dataset"
MULTI_FACE_DIR = Path(__file__).parent.parent.parent / "data" / "multi_face_detected_anime"

ANIME_CASCADE_URL = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
CASCADE_DIR = Path(__file__).parent / "cascades"


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


def get_anime_face_detector():
    """获取动漫人脸检测器"""
    cascade_path = download_anime_cascade()

    if cascade_path and cascade_path.exists():
        try:
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if not cascade.empty():
                return ("Anime-LBP", cascade)
        except Exception as e:
            print(f"加载动漫检测器失败: {e}")

    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if not cascade.empty():
            return ("Haar", cascade)
    except:
        pass

    return None


def is_grayscale(image):
    """检测图片是否为灰度/黑白图"""
    if len(image.shape) == 2:
        return True

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)

    if len(image.shape) == 3:
        color_variance = np.var(r) + np.var(g) + np.var(b)
        gray_mean = np.mean([np.mean(r), np.mean(g), np.mean(b)])

        if color_variance < 500 and gray_mean > 30 and gray_mean < 225:
            return True

        diff_rg = np.mean(np.abs(r.astype(float) - g.astype(float)))
        diff_rb = np.mean(np.abs(r.astype(float) - b.astype(float)))
        diff_gb = np.mean(np.abs(g.astype(float) - b.astype(float)))

        if diff_rg < 10 and diff_rb < 10 and diff_gb < 10:
            return True

    return False


def detect_faces(cascade, image, is_gray):
    """检测人脸 - 对灰度图使用不同参数"""
    if is_gray:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if is_gray:
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE
        )
    else:
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
        )

    return [tuple(f) for f in faces]


def process_image(image_path, detector):
    """处理单张图片"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return str(image_path), 0, [], False

        is_gray = is_grayscale(image)

        if is_gray:
            gray_image = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3:
                image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        faces = detect_faces(detector, image, is_gray)
        return str(image_path), len(faces), faces, is_gray
    except Exception as e:
        print(f"处理失败 {image_path}: {e}")
        return str(image_path), 0, [], False


def process_dataset(dataset_path, output_path, min_faces=2, sample_limit=None):
    """处理整个数据集"""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = []
    for ext in image_extensions:
        image_files.extend(dataset_path.rglob(f"*{ext}"))

    if sample_limit:
        image_files = image_files[:sample_limit]

    print(f"找到 {len(image_files)} 张图片")
    print(f"将检测包含 {min_faces} 张或多张人脸的图片...")

    detector_info = get_anime_face_detector()
    if detector_info is None:
        print("错误: 无法加载人脸检测器")
        return

    detector_type, detector = detector_info
    print(f"使用 {detector_type} 动漫人脸检测器")
    print(f"灰度图片将使用优化的检测参数")

    multi_face_images = []
    single_face_images = []
    no_face_images = []
    grayscale_images = []

    total = len(image_files)
    for i, img_path in enumerate(image_files):
        path, face_count, _, is_gray = process_image(img_path, detector)

        if is_gray:
            grayscale_images.append((path, face_count))

        if face_count >= min_faces:
            multi_face_images.append((path, face_count, is_gray))
        elif face_count == 1:
            single_face_images.append((path, face_count, is_gray))
        else:
            no_face_images.append((path, face_count, is_gray))

        if (i + 1) % 500 == 0:
            gray_count = len(grayscale_images)
            print(
                f"已处理: {i + 1}/{total} | 多脸: {len(multi_face_images)} | 单脸: {len(single_face_images)} | 无脸: {len(no_face_images)} | 灰度: {gray_count}"
            )

    print(f"\n处理完成!")
    print(f"=" * 70)
    print(f"总图片数: {len(image_files)}")
    print(
        f"多角色图片 (>=2人脸): {len(multi_face_images)} ({len(multi_face_images)/len(image_files)*100:.1f}%)"
    )
    print(
        f"单角色图片 (=1人脸): {len(single_face_images)} ({len(single_face_images)/len(image_files)*100:.1f}%)"
    )
    print(f"无检测到人脸: {len(no_face_images)} ({len(no_face_images)/len(image_files)*100:.1f}%)")
    print(f"-" * 70)
    print(
        f"灰度/黑白图片: {len(grayscale_images)} ({len(grayscale_images)/len(image_files)*100:.1f}%)"
    )
    print(f"  - 多角色: {sum(1 for _, fc, ig in multi_face_images if ig)}")
    print(f"  - 单角色: {sum(1 for _, fc, ig in single_face_images if ig)}")
    print(f"  - 无人脸: {sum(1 for _, fc, ig in no_face_images if ig)}")

    multi_face_dir = output_path / "multi_face"
    grayscale_dir = output_path / "grayscale_multi_face"

    multi_face_dir.mkdir(exist_ok=True)
    grayscale_dir.mkdir(exist_ok=True)

    if multi_face_images:
        print(f"\n正在复制多角色图片...")
        gray_multi = 0
        for img_path, face_count, is_gray in sorted(
            multi_face_images, key=lambda x: x[1], reverse=True
        ):
            src = Path(img_path)
            dst = (grayscale_dir if is_gray else multi_face_dir) / f"face_{face_count}_{src.name}"
            try:
                shutil.copy2(src, dst)
                if is_gray:
                    gray_multi += 1
            except Exception as e:
                print(f"复制失败 {src}: {e}")

        print(f"已保存 {len(multi_face_images)} 张多角色图片")
        print(f"  - 彩色: {len(multi_face_images) - gray_multi}")
        print(f"  - 灰度: {gray_multi}")

        with open(output_path / "multi_face_images.txt", "w", encoding="utf-8") as f:
            for img_path, face_count, is_gray in sorted(
                multi_face_images, key=lambda x: x[1], reverse=True
            ):
                f.write(f"{face_count}\t{'[灰度]' if is_gray else '[彩色]'}\t{img_path}\n")

        with open(output_path / "grayscale_images.txt", "w", encoding="utf-8") as f:
            for img_path, face_count in grayscale_images:
                f.write(f"{face_count}\t{img_path}\n")

    summary = {
        "detector_type": detector_type,
        "total_images": len(image_files),
        "multi_face_count": len(multi_face_images),
        "single_face_count": len(single_face_images),
        "no_face_count": len(no_face_images),
        "grayscale_count": len(grayscale_images),
        "grayscale_multi_face": sum(1 for _, fc, ig in multi_face_images if ig),
        "grayscale_single_face": sum(1 for _, fc, ig in single_face_images if ig),
        "grayscale_no_face": sum(1 for _, fc, ig in no_face_images if ig),
        "min_faces_threshold": min_faces,
    }

    with open(output_path / "detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n结果摘要已保存到 {output_path / 'detection_summary.json'}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="动漫人脸多角色检测工具 - 支持黑白图片")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="数据集路径")
    parser.add_argument("--output", type=str, default=str(MULTI_FACE_DIR), help="输出路径")
    parser.add_argument("--min-faces", type=int, default=2, help="最小人脸数量")
    parser.add_argument("--sample", type=int, default=None, help="仅处理前N张图片")

    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"错误: 数据集路径不存在: {args.dataset}")
        sys.exit(1)

    process_dataset(args.dataset, args.output, args.min_faces, args.sample)
