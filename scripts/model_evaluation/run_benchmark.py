#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anime Role Detect - 模型基准测试脚本

测试范围:
  1. EfficientNet-B3 分类模型 (models/efficientnet_b3/model_best.pth)
     - Top-1 / Top-5 准确率
     - 每类准确率与宏平均 F1
     - 推理延迟 (单张 / batch) 与吞吐 (FPS)
     - 模型大小、参数量、内存占用
     - 混淆最严重的类别对
  2. YOLOv8n 检测模型 (yolov8n.pt)
     - 模型大小、参数量
     - 推理延迟与吞吐
     - 在采样图像上的检测统计

输出:
  - scripts/model_evaluation/benchmark_results.json
  - 屏幕打印关键指标
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final_dataset"
MODEL_DIR = PROJECT_ROOT / "models" / "efficientnet_b3"
YOLO_PATH = PROJECT_ROOT / "yolov8n.pt"

# 测试样本上限（每类）—— 控制总耗时
MAX_PER_CLASS = 25
# 速度测试用的样本数
SPEED_TEST_SAMPLES = 200
BATCH_SIZE_SPEED = 32

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_file_size(path: Path) -> float:
    """返回 MB"""
    return path.stat().st_size / (1024 * 1024)


def build_test_samples(class_to_idx: dict, max_per_class: int):
    """从 final_dataset 中按 class_to_idx 采样测试数据"""
    samples = []  # list of (image_path, true_label_idx)
    missing_classes = []
    for class_name, idx in class_to_idx.items():
        class_dir = DATA_DIR / class_name
        if not class_dir.is_dir():
            missing_classes.append(class_name)
            continue
        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        if not files:
            missing_classes.append(class_name)
            continue
        random.shuffle(files)
        chosen = files[:max_per_class]
        for f in chosen:
            samples.append((str(class_dir / f), idx))
    random.shuffle(samples)
    return samples, missing_classes


# ---------------------------------------------------------------------------
# EfficientNet-B3 基准测试
# ---------------------------------------------------------------------------

def create_efficientnet_b3(num_classes: int) -> nn.Module:
    """与 model_loader.py / train_efficientnet_b3.py 一致的架构"""
    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 768),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(768),
        nn.Dropout(p=0.15),
        nn.Linear(768, num_classes),
    )
    return model


def benchmark_efficientnet():
    print("=" * 70)
    print("EfficientNet-B3 分类模型基准测试")
    print("=" * 70)

    model_path = MODEL_DIR / "model_best.pth"
    results_path = MODEL_DIR / "training_results.json"
    class_to_idx_path = MODEL_DIR / "class_to_idx.json"

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    with open(results_path, "r", encoding="utf-8") as f:
        training_cfg = json.load(f)

    num_classes = len(class_to_idx)
    image_size = training_cfg.get("image_size", 224)
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]

    device = get_device()
    print(f"设备: {device}")
    print(f"类别数: {num_classes}")
    print(f"输入分辨率: {image_size}x{image_size}")
    print(f"模型文件: {model_path.name} ({get_model_file_size(model_path):.2f} MB)")

    # 构建并加载模型
    model = create_efficientnet_b3(num_classes)
    state_dict = torch.load(model_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    print(f"参数量: {total_params:,} (可训练: {trainable_params:,})")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 采样测试集
    samples, missing = build_test_samples(class_to_idx, MAX_PER_CLASS)
    print(f"测试样本数: {len(samples)} (每类上限 {MAX_PER_CLASS})")
    if missing:
        print(f"⚠️ 缺少数据目录的类别 ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # ----- 准确率测试 -----
    print("\n[1/3] 准确率评估...")
    t0 = time.time()
    top1_correct = 0
    top5_correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion_pairs = Counter()  # (true, pred) -> count
    all_preds = []
    all_labels = []

    # 按 batch 评估
    batch_size = 32
    batch_imgs = []
    batch_labels = []
    skipped = 0

    def flush_batch():
        nonlocal top1_correct, top5_correct, total, skipped
        if not batch_imgs:
            return
        try:
            x = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                top5 = torch.topk(probs, k=min(5, num_classes), dim=1)
            for i, true_idx in enumerate(batch_labels):
                pred_top1 = top5.indices[i, 0].item()
                pred_top5 = top5.indices[i].tolist()
                per_class_total[class_names[true_idx]] += 1
                if pred_top1 == true_idx:
                    top1_correct += 1
                    per_class_correct[class_names[true_idx]] += 1
                if true_idx in pred_top5:
                    top5_correct += 1
                if pred_top1 != true_idx:
                    confusion_pairs[(class_names[true_idx], class_names[pred_top1])] += 1
                all_preds.append(pred_top1)
                all_labels.append(true_idx)
                total += 1
        except Exception as e:
            skipped += len(batch_imgs)
            print(f"  batch 评估失败 ({len(batch_imgs)} 张): {e}")
        batch_imgs.clear()
        batch_labels.clear()

    for img_path, label_idx in samples:
        try:
            img = Image.open(img_path).convert("RGB")
            t = transform(img)
            batch_imgs.append(t)
            batch_labels.append(label_idx)
            if len(batch_imgs) >= batch_size:
                flush_batch()
        except Exception as e:
            skipped += 1
            continue
    flush_batch()

    eval_time = time.time() - t0
    top1_acc = top1_correct / total if total else 0.0
    top5_acc = top5_correct / total if total else 0.0
    print(f"  评估耗时: {eval_time:.2f}s")
    print(f"  有效样本: {total}, 跳过: {skipped}")
    print(f"  Top-1 准确率: {top1_acc * 100:.2f}%")
    print(f"  Top-5 准确率: {top5_acc * 100:.2f}%")

    # 每类准确率 + 宏平均 F1
    per_class_acc = {}
    macro_precision = 0.0
    macro_recall = 0.0
    f1_per_class = []
    pred_per_class = defaultdict(int)
    for p in all_preds:
        pred_per_class[class_names[p]] += 1

    for cls in class_names:
        gt = per_class_total.get(cls, 0)
        tp = per_class_correct.get(cls, 0)
        pred = pred_per_class.get(cls, 0)
        acc = tp / gt if gt else 0.0
        per_class_acc[cls] = {"correct": tp, "total": gt, "accuracy": acc}
        precision = tp / pred if pred else 0.0
        recall = tp / gt if gt else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)
        macro_precision += precision
        macro_recall += recall
    macro_precision /= max(len(class_names), 1)
    macro_recall /= max(len(class_names), 1)
    macro_f1 = sum(f1_per_class) / max(len(f1_per_class), 1)
    print(f"  宏平均 Precision / Recall / F1: {macro_precision:.4f} / {macro_recall:.4f} / {macro_f1:.4f}")

    # 最易混淆的 top-10 类别对
    top_confused = confusion_pairs.most_common(10)
    print("  最易混淆 Top-10 (真实 → 预测: 次数):")
    for (t, p), c in top_confused:
        print(f"    {t} → {p}: {c}")

    # ----- 速度测试：单张 -----
    print("\n[2/3] 推理速度测试...")
    speed_samples = [s[0] for s in samples[:SPEED_TEST_SAMPLES]]
    if len(speed_samples) < 10:
        # 数据不够，从 final_dataset 凑齐
        for cls_dir in DATA_DIR.iterdir():
            if len(speed_samples) >= SPEED_TEST_SAMPLES:
                break
            if not cls_dir.is_dir():
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    speed_samples.append(str(cls_dir / f))
                    if len(speed_samples) >= SPEED_TEST_SAMPLES:
                        break

    speed_tensors = []
    for p in speed_samples:
        try:
            img = Image.open(p).convert("RGB")
            speed_tensors.append(transform(img))
        except Exception:
            continue
    print(f"  速度测试样本: {len(speed_tensors)}")

    # 预热
    with torch.no_grad():
        for t in speed_tensors[:5]:
            _ = model(t.unsqueeze(0).to(device))
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    # 单张推理
    t0 = time.time()
    with torch.no_grad():
        for t in speed_tensors:
            _ = model(t.unsqueeze(0).to(device))
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    single_time = time.time() - t0
    single_fps = len(speed_tensors) / single_time if single_time > 0 else 0
    single_latency_ms = (single_time / max(len(speed_tensors), 1)) * 1000
    print(f"  单张推理: 平均延迟 {single_latency_ms:.2f} ms, 吞吐 {single_fps:.2f} FPS")

    # batch 推理
    batches = [torch.stack(speed_tensors[i:i + BATCH_SIZE_SPEED])
               for i in range(0, len(speed_tensors), BATCH_SIZE_SPEED)]
    # 预热
    with torch.no_grad():
        _ = model(batches[0].to(device))
    if device.type == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for b in batches:
            _ = model(b.to(device))
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    batch_time = time.time() - t0
    batch_fps = len(speed_tensors) / batch_time if batch_time > 0 else 0
    batch_latency_ms = (batch_time / max(len(batches), 1)) * 1000
    print(f"  Batch({BATCH_SIZE_SPEED}) 推理: 平均延迟 {batch_latency_ms:.2f} ms/batch, 吞吐 {batch_fps:.2f} FPS")

    # ----- 内存占用 -----
    print("\n[3/3] 内存占用...")
    import psutil
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    print(f"  进程 RSS: {rss_mb:.2f} MB")

    # 模型大小（state_dict 内存）
    param_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    buf_size_mb = sum(b.nelement() * b.element_size() for b in model.buffers()) / (1024 * 1024)
    print(f"  模型参数内存: {param_size_mb:.2f} MB, buffers: {buf_size_mb:.2f} MB")

    # 每类准确率排序
    sorted_per_class = sorted(per_class_acc.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    best_classes = [{"class": c, **v} for c, v in sorted_per_class[:5]]
    worst_classes = [{"class": c, **v} for c, v in sorted_per_class[-5:]]

    return {
        "model_name": "efficientnet_b3",
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "num_classes": num_classes,
        "image_size": image_size,
        "model_size_mb": round(get_model_file_size(model_path), 2),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "param_memory_mb": round(param_size_mb, 2),
        "buffer_memory_mb": round(buf_size_mb, 2),
        "process_rss_mb": round(rss_mb, 2),
        "test_samples": total,
        "skipped_samples": skipped,
        "eval_time_seconds": round(eval_time, 2),
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "single_latency_ms": round(single_latency_ms, 2),
        "single_fps": round(single_fps, 2),
        "batch_size": BATCH_SIZE_SPEED,
        "batch_latency_ms": round(batch_latency_ms, 2),
        "batch_fps": round(batch_fps, 2),
        "best_5_classes": best_classes,
        "worst_5_classes": worst_classes,
        "top_confused_pairs": [{"true": t, "pred": p, "count": c} for (t, p), c in top_confused],
        "per_class_accuracy": per_class_acc,
    }


# ---------------------------------------------------------------------------
# YOLOv8n 基准测试
# ---------------------------------------------------------------------------

def benchmark_yolov8n():
    print("\n" + "=" * 70)
    print("YOLOv8n 检测模型基准测试")
    print("=" * 70)

    if not YOLO_PATH.exists():
        print(f"⚠️ YOLOv8n 权重不存在: {YOLO_PATH}")
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️ 未安装 ultralytics，跳过 YOLOv8n 测试")
        return {
            "model_name": "yolov8n",
            "model_path": str(YOLO_PATH.relative_to(PROJECT_ROOT)),
            "model_size_mb": round(get_model_file_size(YOLO_PATH), 2),
            "error": "ultralytics not installed",
        }

    device = get_device()
    yolo_device = "mps" if device.type == "mps" else ("cuda:0" if device.type == "cuda" else "cpu")
    print(f"设备: {yolo_device}")
    print(f"模型文件: {YOLO_PATH.name} ({get_model_file_size(YOLO_PATH):.2f} MB)")

    t0 = time.time()
    model = YOLO(str(YOLO_PATH))
    load_time = time.time() - t0
    print(f"加载耗时: {load_time:.2f}s")

    # 收集速度测试样本
    speed_imgs = []
    for cls_dir in DATA_DIR.iterdir():
        if not cls_dir.is_dir():
            continue
        for f in os.listdir(cls_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                speed_imgs.append(str(cls_dir / f))
                if len(speed_imgs) >= SPEED_TEST_SAMPLES:
                    break
        if len(speed_imgs) >= SPEED_TEST_SAMPLES:
            break
    print(f"速度测试样本: {len(speed_imgs)}")

    # 预热
    for p in speed_imgs[:3]:
        _ = model.predict(p, device=yolo_device, verbose=False)

    # 推理
    t0 = time.time()
    total_detections = 0
    conf_sum = 0.0
    for p in speed_imgs:
        res = model.predict(p, device=yolo_device, verbose=False)
        for r in res:
            if r.boxes is not None:
                total_detections += len(r.boxes)
                if len(r.boxes) > 0:
                    conf_sum += float(r.boxes.conf.sum().item())
    infer_time = time.time() - t0
    fps = len(speed_imgs) / infer_time if infer_time > 0 else 0
    avg_latency = (infer_time / max(len(speed_imgs), 1)) * 1000
    avg_dets_per_img = total_detections / max(len(speed_imgs), 1)
    avg_conf = conf_sum / max(total_detections, 1) if total_detections else 0
    print(f"推理耗时: {infer_time:.2f}s")
    print(f"平均延迟: {avg_latency:.2f} ms, 吞吐: {fps:.2f} FPS")
    print(f"总检测数: {total_detections}, 平均每张 {avg_dets_per_img:.2f} 个, 平均置信度 {avg_conf:.3f}")

    # 参数量
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"参数量: {total_params:,}")

    return {
        "model_name": "yolov8n",
        "model_path": str(YOLO_PATH.relative_to(PROJECT_ROOT)),
        "device": yolo_device,
        "model_size_mb": round(get_model_file_size(YOLO_PATH), 2),
        "total_parameters": total_params,
        "load_time_seconds": round(load_time, 2),
        "test_samples": len(speed_imgs),
        "infer_time_seconds": round(infer_time, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 2),
        "total_detections": total_detections,
        "avg_detections_per_image": round(avg_dets_per_img, 2),
        "avg_confidence": round(avg_conf, 3),
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    results = {
        "benchmark_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project": "anime_role_detect",
    }

    try:
        results["efficientnet_b3"] = benchmark_efficientnet()
    except Exception as e:
        print(f"❌ EfficientNet-B3 基准测试失败: {e}")
        traceback.print_exc()
        results["efficientnet_b3"] = {"error": str(e)}

    try:
        results["yolov8n"] = benchmark_yolov8n()
    except Exception as e:
        print(f"❌ YOLOv8n 基准测试失败: {e}")
        traceback.print_exc()
        results["yolov8n"] = {"error": str(e)}

    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存: {out_path}")

    # 屏幕摘要
    print("\n" + "=" * 70)
    print("摘要")
    print("=" * 70)
    if isinstance(results.get("efficientnet_b3"), dict) and "top1_accuracy" in results["efficientnet_b3"]:
        e = results["efficientnet_b3"]
        print(f"EfficientNet-B3:")
        print(f"  Top-1: {e['top1_accuracy'] * 100:.2f}%  Top-5: {e['top5_accuracy'] * 100:.2f}%  Macro-F1: {e['macro_f1']:.4f}")
        print(f"  单张延迟 {e['single_latency_ms']:.2f} ms ({e['single_fps']:.1f} FPS) / Batch({e['batch_size']}) {e['batch_fps']:.1f} FPS")
        print(f"  参数 {e['total_parameters']:,} / 模型 {e['model_size_mb']} MB / RSS {e['process_rss_mb']} MB")
    if isinstance(results.get("yolov8n"), dict) and "fps" in results["yolov8n"]:
        y = results["yolov8n"]
        print(f"YOLOv8n:")
        print(f"  延迟 {y['avg_latency_ms']:.2f} ms / {y['fps']:.1f} FPS / 模型 {y['model_size_mb']} MB / 参数 {y['total_parameters']:,}")


if __name__ == "__main__":
    main()
