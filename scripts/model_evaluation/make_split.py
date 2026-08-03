#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 final_dataset 的无交叠 train/val/test 切分（修复数据泄漏的根因）。

!!! MUST group by character to avoid leakage !!!
================================================
同一角色目录下的所有图必须整体进 train / val / test 之一，不能拆散。
上一版按"类内逐图洗牌再切分"仍会把同一角色的图分别放进 train 和 test
（制度性泄漏）。本版改用 src/core/data/split_utils.grouped_split，以
"角色"为 group 做 GroupShuffleSplit，确保任何角色的图像只出现在一个 split。

产出（与 train_clean_split.py 兼容）：
  - data/splits/seed42/train.json  (path, label) 列表
  - data/splits/seed42/val.json
  - data/splits/seed42/test.json   ← 永远不参与训练/选模型，仅作权威评测
  - data/splits/seed42/summary.json（每类数量 + 切分统计）

切分方式：按角色分组（GroupShuffleSplit），固定随机种子 42，
比例 train 0.70 / val 0.15 / test 0.15。

后续重训应：只用 train.json 训练、val.json 早停、test.json 仅在最后评测。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.core.data.split_utils import make_character_grouped_split  # noqa: E402

DATA_DIR = ROOT / "data" / "final_dataset"
MODEL_DIR = ROOT / "models" / "efficientnet_b3"
OUT_DIR = ROOT / "data" / "splits" / "seed42"

SEED = 42
RATIOS = (0.70, 0.15, 0.15)


def main():
    c2i = json.load(open(MODEL_DIR / "class_to_idx.json", encoding="utf-8"))

    result = make_character_grouped_split(
        dataset_dir=DATA_DIR,
        out_dir=OUT_DIR,
        ratios=RATIOS,
        seed=SEED,
        label_map=c2i,
    )
    summary = result["summary"]

    # 防回归断言：确认没有任何角色跨 split 出现
    per_class = summary["per_class"]
    leaked = [
        c
        for c, v in per_class.items()
        if (v.get("train", 0) > 0) + (v.get("val", 0) > 0) + (v.get("test", 0) > 0) > 1
    ]
    if leaked:
        raise SystemExit(f"[FATAL] 切分存在泄漏，角色跨 split: {leaked}")

    print(f"Wrote splits -> {OUT_DIR}")
    print("split_counts:", summary["split_counts"])
    print("classes:", summary["num_classes"])
    print(f"[OK] 已按角色分组切分，无角色跨 train/val/test（seed={SEED}）")


if __name__ == "__main__":
    main()
