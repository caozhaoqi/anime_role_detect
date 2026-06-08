#!/usr/bin/env python3
"""NSFW检测工具 - 使用PyTorch深度学习模型（支持macOS）"""

import os
import sys
import json
from pathlib import Path
from PIL import Image

# 设置路径
DATASET_PATH = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset')
OUTPUT_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/nsfw_dl_results')


def load_model():
    """加载预训练的NSFW检测模型"""
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch

        model_name = "Falconsai/nsfw_image_detection"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)

        # 使用CPU模式（macOS兼容）
        model = model.to("cpu")
        model.eval()

        print(f"✅ 加载模型成功: {model_name}")
        return processor, model

    except ImportError as e:
        print(f"❌ transformers库未安装: {e}")
        print("请安装依赖: pip install transformers torch pillow")
        return None, None
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        print(f"异常堆栈: {traceback.format_exc()}")
        return None, None


def analyze_nsfw(image_path, processor, model):
    """使用深度学习模型检测NSFW"""
    try:
        image = Image.open(str(image_path)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        import torch
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        label = model.config.id2label[predicted_class_idx]
        nsfw_score = probabilities[0][model.config.label2id.get("nsfw", 0)].item() * 100

        # 根据分数判断类别
        if nsfw_score > 70:
            final_label = "NSFW"
        elif nsfw_score > 40:
            final_label = "Suggestive"
        else:
            final_label = "Safe"

        return str(image_path), nsfw_score, final_label, label

    except Exception as e:
        print(f"❌ 检测失败 {image_path}: {e}")
        return str(image_path), 0.0, "错误", "未知"


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
    processor, model = load_model()
    if processor is None or model is None:
        print("❌ 无法加载深度学习模型，退出")
        return None

    nsfw_count = 0
    suggestive_count = 0
    safe_count = 0
    error_count = 0
    results = []

    total = len(image_files)
    for i, img_path in enumerate(image_files):
        path, score, label, raw_label = analyze_nsfw(img_path, processor, model)

        results.append({
            "path": str(img_path),
            "character": img_path.parent.name,
            "score": score,
            "label": label,
            "raw_label": raw_label
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
        "detection_method": "Deep Learning (Transformers)",
        "model": "Falconsai/nsfw_image_detection",
        "thresholds": {"NSFW": ">70", "Suggestive": "40-70", "Safe": "<40"}
    }

    with open(output_path / "detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 按角色统计
    char_stats = {}
    for result in results:
        char = result["character"]
        if char not in char_stats:
            char_stats[char] = {"total": 0, "nsfw": 0, "suggestive": 0, "safe": 0}
        char_stats[char]["total"] += 1
        if result["label"] == "NSFW":
            char_stats[char]["nsfw"] += 1
        elif result["label"] == "Suggestive":
            char_stats[char]["suggestive"] += 1
        else:
            char_stats[char]["safe"] += 1

    print(f"\n📊 各角色NSFW分布（深度学习模型）:")
    print(f"{'角色名称':<30} {'总数':>6} {'NSFW':>6} {'Suggestive':>12} {'Safe':>6}")
    print(f"-" * 70)
    for char, stats in sorted(char_stats.items(), key=lambda x: x[1]["nsfw"], reverse=True):
        print(f"{char:<30} {stats['total']:>6} {stats['nsfw']:>6} {stats['suggestive']:>12} {stats['safe']:>6}")

    print(f"\n📁 检测结果已保存到 {output_path}")

    return summary


if __name__ == "__main__":
    process_dataset()