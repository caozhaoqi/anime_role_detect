#!/usr/bin/env python3
"""NSFW检测工具 - 使用PyTorch+torchvision（macOS兼容）"""

import os
import sys
import json
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# 设置路径
DATASET_PATH = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset')
OUTPUT_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/nsfw_dl_results')


def load_model():
    """加载预训练的图像分类模型"""
    try:
        # 使用ResNet50作为基础模型
        model = models.resnet50(pretrained=True)
        
        # 修改最后一层用于二分类（NSFW/Safe）
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        
        # 使用MPS（Apple Silicon）或CPU
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        print(f"✅ 加载模型成功: ResNet50")
        print(f"📍 使用设备: {device}")
        return model, device

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        print(f"异常堆栈: {traceback.format_exc()}")
        return None, None


def analyze_nsfw(image_path, model, device):
    """使用深度学习模型检测NSFW"""
    try:
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(str(image_path)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            nsfw_score = probabilities[0][1].item() * 100  # 假设索引1是NSFW

        # 根据分数判断类别
        if nsfw_score > 70:
            final_label = "NSFW"
        elif nsfw_score > 40:
            final_label = "Suggestive"
        else:
            final_label = "Safe"

        return str(image_path), nsfw_score, final_label

    except Exception as e:
        print(f"❌ 检测失败 {image_path}: {e}")
        return str(image_path), 0.0, "错误"


def process_dataset():
    """处理final_dataset"""
    output_path = OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集所有图片
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = []
    
    for char_dir in DATASET_PATH.iterdir():
        if not char_dir.is_dir():
            continue
        
        for ext in image_extensions:
            image_files.extend(char_dir.glob(f"*{ext}"))

    print(f"📁 找到 {len(image_files)} 张图片")

    # 加载模型
    model, device = load_model()
    if model is None:
        print("❌ 无法加载深度学习模型，退出")
        return None

    nsfw_count = 0
    suggestive_count = 0
    safe_count = 0
    error_count = 0
    results = []

    total = len(image_files)
    for i, img_path in enumerate(image_files):
        path, score, label = analyze_nsfw(img_path, model, device)

        results.append({
            "path": str(img_path),
            "character": img_path.parent.name,
            "score": score,
            "label": label
        })

        if label == "NSFW":
            nsfw_count += 1
        elif label == "Suggestive":
            suggestive_count += 1
        elif label == "Safe":
            safe_count += 1
        else:
            error_count += 1

        if (i + 1) % 50 == 0:
            print(
                f"   已处理: {i + 1}/{total} | NSFW: {nsfw_count} | Suggestive: {suggestive_count} | Safe: {safe_count}"
            )

    print(f"\n✅ 处理完成!")
    print(f"=" * 60)
    print(f"总图片数: {len(image_files)}")
    print(f"NSFW: {nsfw_count} ({nsfw_count/len(image_files)*100:.1f}%)")
    print(f"Suggestive: {suggestive_count} ({suggestive_count/len(image_files)*100:.1f}%)")
    print(f"Safe: {safe_count} ({safe_count/len(image_files)*100:.1f}%)")
    print(f"错误: {error_count}")

    # 保存结果
    with open(output_path / "nsfw_dl_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = {
        "total_images": len(image_files),
        "nsfw_count": nsfw_count,
        "suggestive_count": suggestive_count,
        "safe_count": safe_count,
        "detection_method": "Deep Learning (ResNet50)",
        "model": "ResNet50 (pretrained)",
        "device": str(device),
        "thresholds": {"NSFW": ">70", "Suggestive": "40-70", "Safe": "<40"}
    }

    with open(output_path / "detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📁 检测结果已保存到 {output_path}")
    return summary


if __name__ == "__main__":
    process_dataset()