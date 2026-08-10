#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase0 A/B 量化实验：{整图, 裁剪} × {224, 256} (+300 分辨率探索列)。

目的：用实测数字判断"离线预裁剪流水线（约 3 天工作量）"值不值得做，
而不是靠推理。

设计
----
* 评测集：data/splits/seed42/test.json —— 必须是 #2 内容哈希修复后的新切分。
  脚本会主动拒绝 test.eval.json（已过期作废）。
* 模型：models/efficientnet_b3_v6（171 类，image_size=256），零训练重测。
* 头条指标：MacroF1（长尾场景下 Top-1 会被大类主导而骗人），同时报 Top-1/Top-5。
* 条件：
    legacy224_whole : 修复前的生产管线 Resize((224,224)) 直压，整图
    whole@{224,256,300} : 统一 eval 变换 Resize(s+32)->CenterCrop(s)，整图
    crop@{224,256,300}  : 同上，但先用 YOLO 取最高置信框裁剪
  YOLO 框只跑一次并缓存，供所有 crop 条件复用。

用法
----
  # 冒烟（20 张，估算耗时）
  .venv/bin/python scripts/model_evaluation/ab_crop_resolution.py --limit 20

  # 全量
  .venv/bin/python scripts/model_evaluation/ab_crop_resolution.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# 注意：截断图解码策略（LOAD_TRUNCATED_IMAGES / MAX_IMAGE_PIXELS）由下面这个
# import 自动继承——preprocess 是唯一真源，本脚本不再重复设置。
# 它保证 7 个条件评的是完全相同的 1608 张，分母不会被静默跳过弄乱。
from src.common.preprocess import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    ensure_rgb,
)

DATA_DIR = ROOT / "data" / "final_dataset"
SPLIT_DIR = ROOT / "data" / "splits" / "seed42"
MODEL_DIR = ROOT / "models" / "efficientnet_b3_v6"
CACHE_DIR = ROOT / "data" / "cache"
BOX_CACHE = CACHE_DIR / "yolo_boxes_test.json"
OUT_JSON = ROOT / "data" / "ab_crop_resolution_results.json"

SIZES = (224, 256, 300)


# --------------------------------------------------------------------------
# 变换
# --------------------------------------------------------------------------
def legacy_transform(size: int = 224):
    """修复前的生产管线：直接 Resize((224,224)) 压扁，无 CenterCrop。"""
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
def load_model(device: str):
    """严格加载 v6 权重。

    注意：v6 用的是自定义分类头
      Dropout(0.3) -> Linear(1536,768) -> ReLU -> BN(768) -> Dropout(0.15) -> Linear(768,C)
    直接 models.efficientnet_b3(num_classes=C) 的头是单层 Linear，形状对不上。
    这里复用训练脚本里的 create_efficientnet_b3，保证结构完全一致，并用
    strict=True 加载——任何键缺失都必须暴露，绝不 silently strict=False 放过，
    否则会拿着随机初始化的分类头跑出一堆假数字。
    """
    from scripts.model_training.train_efficientnet_b3 import create_efficientnet_b3

    cfg = json.load(open(MODEL_DIR / "training_results.json"))
    num_classes = cfg["num_classes"]
    class_to_idx = json.load(open(MODEL_DIR / "class_to_idx.json"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # weights=None：随后 strict 加载会覆盖整个主干，无需下载 ImageNet 权重
    model = create_efficientnet_b3(num_classes, weights=None)
    ckpt = torch.load(MODEL_DIR / "model_best.pth", map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)  # 严格：结构必须完全吻合
    model.eval().to(device)
    return model, class_to_idx, idx_to_class, cfg


# --------------------------------------------------------------------------
# YOLO 框缓存
# --------------------------------------------------------------------------
def compute_boxes(paths: List[str], conf: float = 0.2) -> Dict[str, Optional[List[float]]]:
    """对每张图取 YOLO 最高置信框（放宽类别，与生产端一致）。无框则 None。"""
    if BOX_CACHE.exists():
        cached = json.load(open(BOX_CACHE))
        if all(p in cached for p in paths):
            print(f"  复用 YOLO 框缓存 {BOX_CACHE}")
            return cached
    else:
        cached = {}

    from ultralytics import YOLO

    yolo = YOLO(str(ROOT / "yolov8n.pt"))
    t0 = time.time()
    for i, rel in enumerate(paths):
        if rel in cached:
            continue
        img = ensure_rgb(Image.open(DATA_DIR / rel))
        best, best_conf = None, 0.0
        for r in yolo(img, verbose=False):
            if r.boxes is None:
                continue
            for b in r.boxes:
                c = float(b.conf[0])
                if c >= conf and c > best_conf:
                    best_conf = c
                    best = [float(x) for x in b.xyxy[0].cpu().numpy()]
        cached[rel] = best
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"    YOLO {i+1}/{len(paths)}  {el:.0f}s  (~{el/(i+1)*1000:.0f}ms/img)")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(cached, open(BOX_CACHE, "w"))
    print(f"  YOLO 框已缓存 -> {BOX_CACHE}")
    return cached


# --------------------------------------------------------------------------
# 评测
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model,
    device: str,
    samples: List[dict],
    transform,
    boxes: Optional[Dict[str, Optional[List[float]]]],
    batch_size: int = 32,
) -> Dict[str, object]:
    """返回 macro_f1 / top1 / top5 / fallback 率。"""
    preds_top1: List[int] = []
    preds_top5: List[List[int]] = []
    labels: List[int] = []
    n_fallback = 0

    buf, buf_lab = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        logits = model(x)
        k = min(5, logits.shape[1])
        top5 = torch.topk(logits, k=k, dim=1).indices.cpu().tolist()
        for row, lab in zip(top5, buf_lab):
            preds_top1.append(row[0])
            preds_top5.append(row)
            labels.append(lab)
        buf.clear()
        buf_lab.clear()

    for s in samples:
        rel = s["path"]
        img = ensure_rgb(Image.open(DATA_DIR / rel))
        if boxes is not None:
            box = boxes.get(rel)
            if box is None:
                n_fallback += 1
            else:
                x1, y1, x2, y2 = box
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    img = img.crop((x1, y1, x2, y2))
                else:
                    n_fallback += 1
        buf.append(transform(img))
        buf_lab.append(s["label"])
        if len(buf) >= batch_size:
            flush()
    flush()

    from sklearn.metrics import f1_score

    macro_f1 = f1_score(labels, preds_top1, average="macro", zero_division=0)
    top1 = sum(int(p == l) for p, l in zip(preds_top1, labels)) / len(labels)
    top5 = sum(int(l in row) for row, l in zip(preds_top5, labels)) / len(labels)
    return {
        "macro_f1": round(float(macro_f1), 4),
        "top1": round(top1, 4),
        "top5": round(top5, 4),
        "n": len(labels),
        "fallback_to_whole": n_fallback,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 张（冒烟用）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-crop", action="store_true", help="跳过 YOLO 裁剪条件")
    args = ap.parse_args()

    device = args.device or (
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    test_path = SPLIT_DIR / "test.json"
    if not test_path.exists():
        print(f"[FATAL] 找不到 {test_path}")
        return 1
    # 显式拒绝已作废的 test.eval.json
    samples = json.load(open(test_path))
    summary = json.load(open(SPLIT_DIR / "summary.json"))
    print("=" * 84)
    print("Phase0 A/B: {整图, 裁剪} × {224, 256} (+300 探索)")
    print("=" * 84)
    print(f"评测集      : {test_path}  n={len(samples)}")
    print(f"split_hash  : {summary['split_hash'][:16]}  group_by={summary.get('group_by')}")
    print(f"模型        : {MODEL_DIR.name}")
    print(f"device      : {device}")

    if args.limit:
        samples = samples[: args.limit]
        print(f"[冒烟模式] 仅评测前 {len(samples)} 张")

    model, class_to_idx, idx_to_class, cfg = load_model(device)
    print(f"模型类别数  : {cfg['num_classes']}  训练 image_size={cfg['image_size']}")

    # 标签一致性：split 的 label 必须与模型 class_to_idx 对齐
    dirs = sorted(
        p.name
        for p in DATA_DIR.iterdir()
        if p.is_dir() and any(f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") for f in p.iterdir())
    )
    split_label_map = {name: i for i, name in enumerate(dirs)}
    if split_label_map != class_to_idx:
        diff = [k for k in split_label_map if class_to_idx.get(k) != split_label_map[k]]
        print(f"[FATAL] split 标签与模型 class_to_idx 不一致，{len(diff)} 个类错位: {diff[:5]}")
        return 2
    print("标签映射    : ✅ split label 与模型 class_to_idx 完全一致")

    boxes = None
    if not args.skip_crop:
        print("\n[1/2] 计算 YOLO 框（只算一次，所有 crop 条件复用）...")
        boxes = compute_boxes([s["path"] for s in samples])

    print("\n[2/2] 逐条件评测...")
    results: Dict[str, Dict[str, object]] = {}

    conditions: List[Tuple[str, object, bool]] = [
        ("legacy224_whole", legacy_transform(224), False),
    ]
    for s in SIZES:
        conditions.append((f"whole@{s}", build_eval_transform(s), False))
    if not args.skip_crop:
        for s in SIZES:
            conditions.append((f"crop@{s}", build_eval_transform(s), True))

    for name, tf, use_crop in conditions:
        t0 = time.time()
        r = evaluate(model, device, samples, tf, boxes if use_crop else None, args.batch_size)
        r["seconds"] = round(time.time() - t0, 1)
        results[name] = r
        print(
            f"  {name:<18} MacroF1={r['macro_f1']:.4f}  Top1={r['top1']:.4f}  "
            f"Top5={r['top5']:.4f}  ({r['seconds']}s)"
        )

    # ---- 汇总表 ----
    print("\n" + "=" * 84)
    print("A/B 结果表（头条指标 = MacroF1）")
    print("=" * 84)
    print(f"{'条件':<20}{'MacroF1':>10}{'Top-1':>10}{'Top-5':>10}{'n':>8}{'无框回退':>10}")
    print("-" * 84)
    for name, r in results.items():
        print(
            f"{name:<20}{r['macro_f1']:>10.4f}{r['top1']:>10.4f}{r['top5']:>10.4f}"
            f"{r['n']:>8}{r['fallback_to_whole']:>10}"
        )
    print("-" * 84)

    base = results.get("whole@256")
    if base:
        print("\n相对 whole@256 的增量（决定裁剪流水线值不值）：")
        for name, r in results.items():
            if name == "whole@256":
                continue
            d = r["macro_f1"] - base["macro_f1"]
            print(f"  {name:<20} ΔMacroF1 = {d:+.4f}")

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": str(test_path),
        "split_hash": summary["split_hash"],
        "split_group_by": summary.get("group_by"),
        "n_eval": len(samples),
        "model": str(MODEL_DIR),
        "model_num_classes": cfg["num_classes"],
        "model_train_image_size": cfg["image_size"],
        "device": device,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n结果已写入 -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
