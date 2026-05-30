#!/usr/bin/env python3
"""NSFW检测工具 - 使用ONNX运行时"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import json

DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "combined_dataset"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "nsfw_detection_onnx"

MODEL_URL = "https://github.com/onnx/models/raw/main/vision/body_analysis/nsfw/model/nsfw-resnet50-v2-9.onnx"
MODEL_PATH = Path(__file__).parent / "models" / "nsfw-resnet50-v2-9.onnx"


def download_model():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    if MODEL_PATH.exists():
        return True
    try:
        import urllib.request

        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        return True
    except:
        return False


def load_model():
    try:
        download_model()
        import onnxruntime as ort

        session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        return session
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None


def analyze_nsfw(image_path, session):
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return str(image_path), 0.0, "无法加载"

        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        outputs = session.run(None, {"input": image})

        nsfw_score = outputs[0][0][1] * 100

        if nsfw_score > 50:
            label = "NSFW"
        elif nsfw_score > 25:
            label = "Suggestive"
        else:
            label = "Safe"

        return str(image_path), nsfw_score, label

    except Exception as e:
        print(f"检测失败 {image_path}: {e}")
        return str(image_path), 0.0, "错误"


def process_dataset(dataset_path, output_path, sample_limit=None):
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

    net = load_model()
    if net is None:
        print("错误: 无法加载模型")
        return None

    nsfw_count = 0
    suggestive_count = 0
    safe_count = 0
    results = []

    total = len(image_files)
    for i, img_path in enumerate(image_files):
        path, score, label = analyze_nsfw(img_path, net)

        results.append({"path": path, "score": score, "label": label})

        if label == "NSFW":
            nsfw_count += 1
        elif label == "Suggestive":
            suggestive_count += 1
        else:
            safe_count += 1

        if (i + 1) % 50 == 0:
            print(
                f"已处理: {i + 1}/{total} | NSFW: {nsfw_count} | Suggestive: {suggestive_count} | Safe: {safe_count}"
            )

    print(f"\n处理完成!")
    print(f"总图片数: {len(image_files)}")
    print(f"NSFW: {nsfw_count} ({nsfw_count/len(image_files)*100:.1f}%)")
    print(f"Suggestive: {suggestive_count} ({suggestive_count/len(image_files)*100:.1f}%)")
    print(f"Safe: {safe_count} ({safe_count/len(image_files)*100:.1f}%)")

    with open(output_path / "nsfw_detection_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n检测结果已保存到 {output_path / 'nsfw_detection_results.json'}")

    return {
        "total_images": len(image_files),
        "nsfw_count": nsfw_count,
        "suggestive_count": suggestive_count,
        "safe_count": safe_count,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSFW检测工具 - 使用ONNX")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="数据集路径")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="输出路径")
    parser.add_argument("--sample", type=int, default=None, help="仅处理前N张图片")

    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"错误: 数据集路径不存在: {args.dataset}")
        sys.exit(1)

    process_dataset(args.dataset, args.output, args.sample)
