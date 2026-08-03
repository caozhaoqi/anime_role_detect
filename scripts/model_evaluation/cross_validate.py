#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交叉验证脚本：用最终采集数据评估现有模型

1. 加载 EfficientNet-B3 模型 (51 类)
2. 在 final_dataset 上逐类推理
3. 输出: 准确率/精确率/召回率/F1/混淆矩阵

Usage:
    python3 scripts/model_evaluation/cross_validate.py
    python3 scripts/model_evaluation/cross_validate.py --output cv_report.json
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b3_v3" / "model_best.pth"
CLASS_IDX_PATH = PROJECT_ROOT / "models" / "efficientnet_b3_v3" / "class_to_idx.json"
TRAINING_RESULTS_PATH = PROJECT_ROOT / "models" / "efficientnet_b3_v3" / "training_results.json"
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"
TRAIN_DIR = PROJECT_ROOT / "data" / "training_dataset"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# 模型类名 → 最终集目录名映射（处理短名/全名差异）
NAME_MAP = {
    "Aru": "rikuhachima_aru",
    "Kayoko": "onikata_kayoko",
    "ako": "amau_ako",
    "Hina": "sorasaki_hina",
    "Herta": "herta",
    "Kachina": "kachina",
    "Kirara": "kirara",
    "Klee": "klee",
    "Yaoyao": "yaoyao",
    "Firefly": "firefly",
    "Furina": "furina",
    "Clara": "clara",
    "Diona": "diona",
}


def load_model(device):
    """加载训练好的模型"""
    with open(CLASS_IDX_PATH) as f:
        class_to_idx = json.load(f)
    with open(TRAINING_RESULTS_PATH) as f:
        config = json.load(f)

    num_classes = config["num_classes"]
    image_size = config.get("image_size", 256)

    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 768),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(768),
        nn.Dropout(p=0.15),
        nn.Linear(768, num_classes),
    )
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return model, transform, class_to_idx, config


def find_data_dir(class_name, class_to_idx):
    """找到类名对应的数据目录"""
    # 先尝试 NAME_MAP 映射
    mapped = NAME_MAP.get(class_name, class_name)

    # 在 final_dataset 中查找（不区分大小写）
    final_map = {}
    if FINAL_DIR.exists():
        for d in FINAL_DIR.iterdir():
            if d.is_dir():
                final_map[d.name.lower()] = d.name

    for candidate in [mapped, mapped.lower()]:
        if candidate.lower() in final_map:
            return FINAL_DIR / final_map[candidate.lower()]

    # 在 training_dataset 中查找
    if TRAIN_DIR.exists():
        train_map = {}
        for d in TRAIN_DIR.iterdir():
            if d.is_dir():
                train_map[d.name.lower()] = d.name
        for candidate in [class_name, class_name.lower()]:
            if candidate.lower() in train_map:
                return TRAIN_DIR / train_map[candidate.lower()]

    return None


def predict_image(model, transform, image_path, device):
    """预测单张图片"""
    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top5_probs, top5_idx = torch.topk(probabilities, 5)
        return {
            "pred_class": top5_idx[0][0].item(),
            "confidence": top5_probs[0][0].item(),
            "top5": [(top5_idx[0][i].item(), top5_probs[0][i].item()) for i in range(5)],
        }
    except Exception as e:
        return None


def compute_metrics(confusion, class_list, idx_to_class):
    """计算精确率/召回率/F1"""
    n = len(class_list)
    metrics = {}
    total_tp = total_fp = total_fn = 0

    for i, cls in enumerate(class_list):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[cls] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_precision = np.mean([m["precision"] for m in metrics.values()])
    macro_recall = np.mean([m["recall"] for m in metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in metrics.values()])

    return {
        "per_class": metrics,
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="模型交叉验证")
    parser.add_argument("--output", type=str, default="outputs/cross_validation_report.json")
    parser.add_argument("--limit", type=int, default=0, help="每类最多评估N张")
    args = parser.parse_args()

    # 设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # 加载模型
    print("Loading model...")
    model, transform, class_to_idx, config = load_model(device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = config["class_names"]
    print(f"Model: {config['model_name']}, {len(class_names)} classes, trained acc={config['best_accuracy']:.2%}")

    # 初始化混淆矩阵
    n_classes = len(class_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    results = {
        "model": config["model_name"],
        "model_path": str(MODEL_PATH),
        "num_classes": n_classes,
        "training_accuracy": round(config["best_accuracy"], 4),
        "timestamp": datetime.now().isoformat(),
        "total_images": 0,
        "correct": 0,
        "per_class": {},
        "errors": [],
    }

    print(f"\n{'='*60}")
    print(f"Cross-Validation on final_dataset")
    print(f"{'='*60}")

    for cls_name in class_names:
        true_idx = class_to_idx[cls_name]
        data_dir = find_data_dir(cls_name, class_to_idx)

        if not data_dir:
            results["per_class"][cls_name] = {"images": 0, "correct": 0, "accuracy": 0, "note": "no data"}
            continue

        images = sorted([f for f in data_dir.iterdir()
                         if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
        if args.limit > 0:
            images = images[:args.limit]

        if not images:
            results["per_class"][cls_name] = {"images": 0, "correct": 0, "accuracy": 0, "note": "empty dir"}
            continue

        correct = 0
        top5_correct = 0
        for img_path in images:
            pred = predict_image(model, transform, img_path, device)
            if pred is None:
                continue

            results["total_images"] += 1
            pred_idx = pred["pred_class"]
            confusion[true_idx, pred_idx] += 1

            if pred_idx == true_idx:
                correct += 1
                results["correct"] += 1

            if any(idx == true_idx for idx, _ in pred["top5"]):
                top5_correct += 1

        acc = correct / len(images) if images else 0
        top5_acc = top5_correct / len(images) if images else 0
        results["per_class"][cls_name] = {
            "images": len(images),
            "correct": correct,
            "accuracy": round(acc, 4),
            "top5_accuracy": round(top5_acc, 4),
            "data_source": str(data_dir),
        }

        status = "✅" if acc > 0.5 else ("⚠️" if acc > 0.2 else "❌")
        print(f"  {status} {cls_name:25s}: {correct}/{len(images)} ({acc:.1%}) top5={top5_acc:.1%}")

    # 计算指标
    overall_acc = results["correct"] / results["total_images"] if results["total_images"] > 0 else 0
    metrics = compute_metrics(confusion, class_names, idx_to_class)

    results["overall_accuracy"] = round(overall_acc, 4)
    results["confusion_matrix"] = confusion.tolist()
    results["class_names"] = class_names
    results["metrics"] = metrics

    # 输出摘要
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {results['total_images']}")
    print(f"Top-1 Accuracy: {overall_acc:.2%}")
    print(f"Macro Precision: {metrics['macro_precision']:.2%}")
    print(f"Macro Recall: {metrics['macro_recall']:.2%}")
    print(f"Macro F1: {metrics['macro_f1']:.2%}")

    # 最好和最差的类别
    class_accs = [(k, v["accuracy"], v["images"]) for k, v in results["per_class"].items()
                  if v.get("images", 0) > 0]
    class_accs.sort(key=lambda x: -x[1])

    print(f"\nBest 5:")
    for name, acc, cnt in class_accs[:5]:
        print(f"  ✅ {name}: {acc:.1%} ({cnt} imgs)")

    print(f"\nWorst 5:")
    for name, acc, cnt in class_accs[-5:]:
        print(f"  ❌ {name}: {acc:.1%} ({cnt} imgs)")

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {args.output}")

    return results


if __name__ == "__main__":
    main()