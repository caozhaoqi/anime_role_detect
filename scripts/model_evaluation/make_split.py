#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 final_dataset 的无交叠 train/val/test 切分（修复数据泄漏的根因）。

切分按 **post-id（图片来源 / 近重复簇）** 分组，杜绝同源泄漏；同一角色的
不同 post 允许跨 train/val/test（闭集分类器的正常训练目标）。再叠加
min_train_guarantee：任何只有 1-2 个 post 组的弱类若被整组踢到 val/test，
会被整体提升回 train，保证零样本清零。详见 src/core/data/split_utils.py。

产出（与 train_clean_split.py 兼容）：
  - data/splits/seed42/train.json  (path, label) 列表
  - data/splits/seed42/val.json
  - data/splits/seed42/test.json   ← 永远不参与训练/选模型，仅作权威评测
  - data/splits/seed42/summary.json（schema_version=2：每类数量 + eval_status + split_hash）

切分方式：post-id 分组 GroupShuffleSplit，固定随机种子 42，
比例 train 0.70 / val 0.15 / test 0.15，min_train_guarantee=True。

后续重训应：只用 train.json 训练、val.json 早停、test.json 仅在最后评测。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.core.data.split_utils import make_character_grouped_split, extract_post_id  # noqa: E402

DATA_DIR = ROOT / "data" / "final_dataset"
MODEL_DIR = ROOT / "models" / "efficientnet_b3"
OUT_DIR = ROOT / "data" / "splits" / "seed42"

SEED = 42
RATIOS = (0.70, 0.15, 0.15)


def main():
    # 不再依赖旧模型的 class_to_idx.json（51 类），直接从 final_dataset 子目录
    # 推导类目（与 split_utils 的 label_map fallback 一致），得到今天的 172 类。
    result = make_character_grouped_split(
        dataset_dir=DATA_DIR,
        out_dir=OUT_DIR,
        ratios=RATIOS,
        seed=SEED,
    )
    summary = result["summary"]

    # 防回归断言：确认没有任何 post-id（近重复簇）跨 split 出现
    def _pids(split):
        return {extract_post_id(s["path"].split("/", 1)[1]) for s in split}

    tp, vp, ep = _pids(result["train"]), _pids(result["val"]), _pids(result["test"])
    leak = (tp & vp) | (tp & ep) | (vp & ep)
    if leak:
        raise SystemExit(f"[FATAL] 切分存在 post-id 泄漏，跨 split: {sorted(leak)[:10]}")

    # 零样本清零断言（min_train_guarantee 必须保证此为空）
    if summary["zero_shot_characters"]:
        raise SystemExit(f"[FATAL] 仍存在零样本类: {summary['zero_shot_characters']}")

    print(f"Wrote splits -> {OUT_DIR}")
    print("split_counts:", summary["split_counts"])
    print("schema_version:", summary["schema_version"])
    print(
        "classes(with_images):",
        summary["num_classes_with_images"],
        "/ declared:",
        summary["num_classes"],
    )
    print("train_character_coverage:", summary["train_character_coverage"])
    print("zero_shot_characters:", summary["zero_shot_characters"])
    print("split_hash:", summary["split_hash"][:16])
    print(f"[OK] post-id 零跨集 + 零样本清零（seed={SEED}）")


if __name__ == "__main__":
    main()
