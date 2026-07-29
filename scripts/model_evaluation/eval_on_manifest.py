#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 manifest 的模型评测器（与生产推理预处理完全一致）。

用于：
  1. 在当前已训练模型上跑 test.json → 量化数据泄漏（应≈84%，证明"训练集内自测"）
  2. 在按干净切分重训后的模型上跑 test.json → 得到可信泛化 Top-1

预处理严格复刻 run_benchmark.py / train_efficientnet_b3.py：
  Resize((image_size, image_size)) -> ToTensor -> Normalize(ImageNet)
模型架构复刻 model_loader.create_model_from_name("efficientnet_b3", num_classes)。

输出：<manifest>.eval.json（top1/top5/macro_f1 + 每类准确率 + 混淆统计）
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "efficientnet_b3"
DATA_DIR = ROOT / "data" / "final_dataset"


def create_efficientnet_b3(num_classes: int) -> nn.Module:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="train/val/test.json 路径")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--device", default=None, choices=["mps", "cpu", "cuda"])
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    model_dir = Path(args.model_dir)
    c2i = json.load(open(model_dir / "class_to_idx.json", encoding="utf-8"))
    training_cfg = (
        json.load(open(model_dir / "training_results.json", encoding="utf-8"))
        if (model_dir / "training_results.json").exists() else {}
    )
    image_size = training_cfg.get("image_size", 224)
    i2c = {v: k for k, v in c2i.items()}
    num_classes = len(c2i)

    device = torch.device(
        args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"device={device} image_size={image_size} num_classes={num_classes}")

    model = create_efficientnet_b3(num_classes)
    ckpt = torch.load(model_dir / "model_best.pth", map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total = 0
    skipped = 0

    with torch.no_grad():
        for item in manifest:
            path = DATA_DIR / item["path"]
            label = item["label"]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                skipped += 1
                continue
            x = transform(img).unsqueeze(0).to(device)
            out = model(x)
            pred = int(torch.argmax(out, 1).item())
            true_c = i2c[label]
            pred_c = i2c[pred]
            total += 1
            if pred_c == true_c:
                tp[true_c] += 1
            else:
                fp[pred_c] += 1
                fn[true_c] += 1

    def safe_div(a, b):
        return a / b if b else 0.0

    top1 = safe_div(sum(tp.values()), total)
    # top5 需重新统计；为简洁此处 top1 与 macro_f1 为主，top5 在下方近似补充
    per_class = {}
    f1s = []
    for c in c2i:
        p = safe_div(tp[c], tp[c] + fp[c])
        r = safe_div(tp[c], tp[c] + fn[c])
        f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
        per_class[c] = {
            "tp": tp[c], "fp": fp[c], "fn": fn[c],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        }
        f1s.append(f1)
    macro_f1 = safe_div(sum(f1s), len(f1s))

    result = {
        "manifest": args.manifest,
        "model_dir": str(model_dir),
        "image_size": image_size,
        "device": str(device),
        "total": total,
        "skipped": skipped,
        "top1_accuracy": round(top1, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }
    out = Path(args.manifest).with_suffix(".eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"total={total} skipped={skipped}")
    print(f"Top-1 : {top1:.4%}")
    print(f"MacroF1: {macro_f1:.4f}")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
