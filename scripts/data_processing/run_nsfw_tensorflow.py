#!/usr/bin/env python3
"""NSFW检测工具 - 使用TensorFlow加载MobileNet V2模型"""

import os
import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np

# 设置路径
DATASET_PATH = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset')
OUTPUT_DIR = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/nsfw_tf_results')
MODEL_PATH = Path('/Users/caozhaoqi/PycharmProjects/anime_role_detect/archived/nsfw_model_img/src/main/resources/mobilenet_v2_140_224')

# 标签定义
LABELS = ["drawings", "hentai", "neutral", "porn", "sexy"]


def load_model():
    """加载TensorFlow模型"""
    try:
        import tensorflow as tf
        
        # 检查模型路径
        if not MODEL_PATH.exists():
            print(f"❌ 模型路径不存在: {MODEL_PATH}")
            return None, None
        
        # 加载SavedModel
        model = tf.saved_model.load(str(MODEL_PATH))
        print(f"✅ 加载模型成功: {MODEL_PATH}")
        return model, LABELS

    except ImportError:
        print("❌ TensorFlow未安装，请安装: pip install tensorflow")
        return None, None
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        print(f"异常堆栈: {traceback.format_exc()}")
        return None, None


def preprocess_image(image_path):
    """图像预处理"""
    try:
        # 加载图像
        image = Image.open(str(image_path)).convert("RGB")
        
        # 调整大小到224x224
        image = image.resize((224, 224))
        
        # 转换为numpy数组
        image_array = np.array(image, dtype=np.float32)
        
        # 归一化到[0, 1]
        image_array = image_array / 255.0
        
        # 添加批次维度
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array

    except Exception as e:
        print(f"❌ 图像预处理失败 {image_path}: {e}")
        return None


def analyze_nsfw(image_path, model, labels):
    """使用TensorFlow模型检测NSFW"""
    try:
        # 预处理图像
        image_array = preprocess_image(image_path)
        if image_array is None:
            return str(image_path), 0.0, "错误", {}

        # 获取推理函数
        infer = model.signatures["serving_default"]
        
        # 进行推理
        result = infer(input=tf.constant(image_array))
        
        # 获取预测结果
        predictions = result["prediction"].numpy()[0]
        
        # 构建结果字典
        scores = {}
        for i, label in enumerate(labels):
            scores[label] = float(predictions[i])
        
        # 找到最高分数的类别
        max_score = max(scores.values())
        predicted_label = max(scores, key=scores.get)
        
        # 计算NSFW得分
        nsfw_categories = ["porn", "sexy", "hentai"]
        nsfw_score = sum(scores.get(cat, 0) for cat in nsfw_categories)
        
        # 根据分数判断类别
        if nsfw_score > 0.7:
            final_label = "NSFW"
        elif nsfw_score > 0.4:
            final_label = "Suggestive"
        else:
            final_label = "Safe"

        return str(image_path), nsfw_score, final_label, scores

    except Exception as e:
        print(f"❌ 检测失败 {image_path}: {e}")
        return str(image_path), 0.0, "错误", {}


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
    model, labels = load_model()
    if model is None:
        print("❌ 无法加载模型，退出")
        return None

    nsfw_count = 0
    suggestive_count = 0
    safe_count = 0
    error_count = 0
    results = []

    total = len(image_files)
    for i, img_path in enumerate(image_files):
        path, score, label, scores = analyze_nsfw(img_path, model, labels)

        results.append({
            "path": str(img_path),
            "character": img_path.parent.name,
            "score": score,
            "label": label,
            "details": scores
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
    with open(output_path / "nsfw_tf_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary = {
        "total_images": len(image_files),
        "nsfw_count": nsfw_count,
        "suggestive_count": suggestive_count,
        "safe_count": safe_count,
        "detection_method": "TensorFlow (MobileNet V2)",
        "model": "MobileNet V2 140 224",
        "model_path": str(MODEL_PATH),
        "thresholds": {"NSFW": ">0.7", "Suggestive": "0.4-0.7", "Safe": "<0.4"}
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

    print(f"\n📊 各角色NSFW分布（TensorFlow模型）:")
    print(f"{'角色名称':<30} {'总数':>6} {'NSFW':>6} {'Suggestive':>12} {'Safe':>6}")
    print(f"-" * 70)
    for char, stats in sorted(char_stats.items(), key=lambda x: x[1]["nsfw"], reverse=True):
        print(f"{char:<30} {stats['total']:>6} {stats['nsfw']:>6} {stats['suggestive']:>12} {stats['safe']:>6}")

    print(f"\n📁 检测结果已保存到 {output_path}")
    return summary


if __name__ == "__main__":
    process_dataset()