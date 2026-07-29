#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 final_dataset 的无交叠 train/val/test 切分（修复数据泄漏的根因）。

现状问题（来自 train_efficientnet_b3.py）：
  - 它用 train_ratio=0.85 把整个 final_dataset 切成 train/val，
    没有独立 test 集；val 集既用于早停选模型又用于报指标 → 制度性泄漏。
  - 因此任何在 final_dataset 上的评测都是"训练集内自测"，84% 不代表泛化。

本脚本产出：
  - data/splits/seed42/train.json  (path, label) 列表
  - data/splits/seed42/val.json
  - data/splits/seed42/test.json   ← 永远不参与训练/选模型，仅作权威评测
  - data/splits/seed42/summary.json（每类数量 + 切分统计）

切分方式：按 51 个模型类分层（stratified），固定随机种子 42，
比例 train 0.70 / val 0.15 / test 0.15。

后续重训应：只用 train.json 训练、val.json 早停、test.json 仅在最后评测。
"""
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "final_dataset"
MODEL_DIR = ROOT / "models" / "efficientnet_b3"
OUT_DIR = ROOT / "data" / "splits" / "seed42"

SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
EXTS = (".jpg", ".png", ".jpeg", ".webp")


def main():
    c2i = json.load(open(MODEL_DIR / "class_to_idx.json"))
    i2c = {v: k for k, v in c2i.items()}

    splits = {k: [] for k in RATIOS}
    per_class = {}
    for cls in sorted(c2i, key=lambda c: c2i[c]):
        p = DATA_DIR / cls
        if not p.is_dir():
            continue
        imgs = sorted(f.name for f in p.iterdir() if f.suffix.lower() in EXTS)
        if not imgs:
            continue
        # 每层独立洗牌，保证不同类之间切分互不干扰但可复现
        random.seed(SEED)
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(round(n * RATIOS["train"]))
        n_val = int(round(n * RATIOS["val"]))
        parts = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train + n_val],
            "test": imgs[n_train + n_val:],
        }
        for split, files in parts.items():
            for f in files:
                splits[split].append({"path": f"{cls}/{f}", "label": c2i[cls]})
        per_class[cls] = {
            "total": n,
            "train": len(parts["train"]),
            "val": len(parts["val"]),
            "test": len(parts["test"]),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        with open(OUT_DIR / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)

    summary = {
        "seed": SEED,
        "ratios": RATIOS,
        "num_classes": len(c2i),
        "total_images": sum(len(v) for v in splits.values()),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "per_class": per_class,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote splits -> {OUT_DIR}")
    print("split_counts:", summary["split_counts"])
    print("classes:", summary["num_classes"])


if __name__ == "__main__":
    main()
