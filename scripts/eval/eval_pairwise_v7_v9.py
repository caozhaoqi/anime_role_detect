#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v7 vs v9 配对显著性分析（书中 E9 方法落地，2026-08-20）。

背景
----
v9 与 v7 的 held-out 测试集是「同一组图片」（post_id 分组切分未变，仅 T3-1
标签清理：删 4 个 TRAIN_ONLY 类 + 保序重编号 0..166，见 model-baseline-v9 报告），
因此 1608 个测试样本天然可配对 —— 适合逐样本四象限 + McNemar + 配对 bootstrap，
回答「v9 相对 v7 的 +1.0~1.3 点改进是否显著超过噪声」。

方法
----
- 模型结构：与 train_efficientnet_b3.create_efficientnet_b3 逐字节一致（复用 eval_v9_test.py）
- 预处理：canonical Resize(288)->CenterCrop(256)（对外报数口径）
- 逐样本记录：(path, true_label, pred_v7_role, pred_v9_role)
- v7 输出 171 类索引 -> 角色名（用 v7 空间备份切分 train.json 的 label->path 目录名映射）
  -> 映射到当前 167 类空间（不在契约内 => 记 top-1 错）
- 统计：四象限计数 / ΔTop1 / McNemar 精确检验（binomtest）/ 配对 bootstrap 95% CI

用法
----
  .venv/bin/python scripts/eval/eval_pairwise_v7_v9.py                    # 全量 CPU
  .venv/bin/python scripts/eval/eval_pairwise_v7_v9.py --device mps      # MPS 加速
  .venv/bin/python scripts/eval/eval_pairwise_v7_v9.py --max-samples 64  # 冒烟
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.preprocess import build_eval_transform, ensure_rgb  # noqa: E402

# ---------------------------------------------------------------------------
# 与 eval_v9_test.py 一致的模型结构 / 数据加载
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> nn.Module:
    m = models.efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(m.classifier[1].in_features, 768),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(768),
        nn.Dropout(p=0.15),
        nn.Linear(768, num_classes),
    )
    return m


MAX_SIDE = 1280
TF = build_eval_transform(256)  # canonical: Resize(288)->CenterCrop(256)


class ManifestDS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        try:
            with Image.open(p) as im:
                im = im.copy()
                if max(im.size) > MAX_SIDE:
                    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                img = ensure_rgb(im)
            return TF(img), int(y), 1
        except Exception:
            return torch.zeros(3, 256, 256), int(y), 0


@torch.no_grad()
def predict_top1(model, items, device, batch_size=64):
    """逐样本 top-1 预测索引（保持顺序）。返回 (pred_idx, valid_mask)。"""
    dl = DataLoader(ManifestDS(items), batch_size=batch_size, num_workers=0, shuffle=False)
    preds, valid = [], []
    for x, _y, v in dl:
        out = model(x.to(device)).float().cpu()
        preds.append(out.argmax(dim=1).numpy())
        valid.append(v.numpy())
    return np.concatenate(preds), np.concatenate(valid).astype(bool)


def load_items(test_path: Path, data_dir: Path, max_samples=None):
    rows = json.load(open(test_path, encoding="utf-8"))
    items, skipped = [], 0
    for r in rows:
        p = data_dir / r["path"]
        if not p.exists():
            skipped += 1
            continue
        items.append((str(p), int(r["label"])))
        if max_samples and len(items) >= max_samples:
            break
    return items, skipped


def build_label_role_map(split_path: Path):
    """从切分 json 构建 label -> role（path 首段目录名）。"""
    rows = json.load(open(split_path, encoding="utf-8"))
    mapping = {}
    for r in rows:
        role = r["path"].split("/")[0]
        mapping.setdefault(int(r["label"]), role)
    return mapping


# 12 个「数据充足却认不准」弱类（v7 报告第 5 节清单：k>=30 且 test sup>=8 且 F1<0.45）
WEAK_CLASSES = [
    "asagi_mutsuki", "lisa", "kaname_madoka", "paimon", "lynx",
    "sorasaki_hina", "silver_wolf", "tsushima_yoshiko", "kagura_onmyouji",
    "ram_re:zero", "hiiragi_kagami", "yuuki_asuna",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="auto=cpu（Mac 上 eval 一贯 CPU 稳）；可显式 mps")
    ap.add_argument("--v7-model-dir", default=str(ROOT / "models" / "efficientnet_b3_v7"))
    ap.add_argument("--v9-model-dir", default=str(ROOT / "models" / "efficientnet_b3_v9"))
    ap.add_argument("--split-dir", default=str(ROOT / "data" / "splits" / "seed42"))
    ap.add_argument("--v7-split-dir", default=str(ROOT / "data" / "splits" / "seed42_backup_pre_167clean_20260810_144211"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "final_dataset"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "pairwise_v7_v9_report.json"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=None, help="冒烟用")
    ap.add_argument("--bootstrap-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    split_dir, v7_split_dir, data_dir = Path(args.split_dir), Path(args.v7_split_dir), Path(args.data_dir)
    device = args.device or "cpu"

    # ---- 1) 映射 ----
    v7_label_role = build_label_role_map(v7_split_dir / "train.json")       # 171 空间: v7 idx -> role
    cur_label_role = build_label_role_map(split_dir / "test.json")          # 167 空间: cur label -> role
    cur_role_label = {v: k for k, v in cur_label_role.items()}
    print(f"[map] v7 空间 {len(v7_label_role)} 类 -> 当前空间 {len(cur_label_role)} 类")

    # ---- 2) 加载数据与模型 ----
    items, skipped = load_items(split_dir / "test.json", data_dir, args.max_samples)
    print(f"[data] test 样本 {len(items)} (跳过 {skipped})")

    v7_model = build_model(171).eval()
    v7_model.load_state_dict(torch.load(Path(args.v7_model_dir) / "model_best.pth", map_location="cpu")["model_state_dict"])
    v9_model = build_model(167).eval()
    v9_model.load_state_dict(torch.load(Path(args.v9_model_dir) / "model_best.pth", map_location="cpu")["model_state_dict"])
    v7_model.to(device)
    v9_model.to(device)
    print(f"[model] v7(171) + v9(167) 已加载 -> {device}")

    # ---- 3) 逐样本推理 ----
    pred7_idx, valid = predict_top1(v7_model, items, device, args.batch_size)
    pred9_idx, _ = predict_top1(v9_model, items, device, args.batch_size)
    n = len(items)

    # ---- 4) 逐样本配对判定 ----
    rows = []
    for i in range(n):
        path, true_label = items[i]
        if not valid[i]:
            rows.append({"path": path, "true_label": int(true_label), "valid": False})
            continue
        role_v7 = v7_label_role.get(int(pred7_idx[i]), None)
        label_v7 = cur_role_label.get(role_v7, -1)          # v7 预测映射到当前空间，不在契约=>-1
        ok7 = bool(label_v7 == true_label)
        ok9 = bool(int(pred9_idx[i]) == true_label)
        rows.append({
            "path": path, "true_label": int(true_label),
            "pred_v7_idx": int(pred7_idx[i]), "pred_v7_role": role_v7,
            "pred_v9_idx": int(pred9_idx[i]), "pred_v9_role": cur_label_role.get(int(pred9_idx[i])),
            "ok_v7": ok7, "ok_v9": ok9, "valid": True,
        })
    ok_rows = [r for r in rows if r["valid"]]
    y = np.array([r["true_label"] for r in ok_rows])
    ok7 = np.array([r["ok_v7"] for r in ok_rows])
    ok9 = np.array([r["ok_v9"] for r in ok_rows])

    top1_v7, top1_v9 = ok7.mean(), ok9.mean()
    delta = top1_v9 - top1_v7
    a = int((ok7 & ok9).sum())    # 双对
    b = int((ok7 & ~ok9).sum())   # v7 对 v9 错
    c = int((~ok7 & ok9).sum())   # v7 错 v9 对
    d = int((~ok7 & ~ok9).sum())  # 双错
    print(f"\n=== 四象限 (n={len(ok_rows)}) ===")
    print(f"双对 a={a} | v7对v9错 b={b} | v7错v9对 c={c} | 双错 d={d}")

    # McNemar 精确检验：只关注 b vs c（不一致样本）
    mcnemar_p = None
    if b + c > 0:
        mcnemar_p = stats.binomtest(min(b, c), n=b + c, p=0.5).pvalue * 2  # 双侧
        mcnemar_p = min(mcnemar_p, 1.0)
    print(f"ΔTop1 = {delta*100:+.2f} 点 (v9 {top1_v9*100:.2f}% vs v7 {top1_v7*100:.2f}%)")
    print(f"McNemar 精确检验: b={b}, c={c}, p={mcnemar_p:.4f} {'<-- 显著' if mcnemar_p is not None and mcnemar_p < 0.05 else ''}")

    # 配对 bootstrap（保留配对关系重采样）
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(ok7))
    deltas = np.empty(args.bootstrap_n)
    for j in range(args.bootstrap_n):
        s = rng.choice(idx, size=len(idx), replace=True)
        deltas[j] = ok9[s].mean() - ok7[s].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"配对 bootstrap 95% CI of ΔTop1: [{lo*100:+.2f}, {hi*100:+.2f}] 点 "
          f"{'(不含 0 => 显著)' if lo > 0 or hi < 0 else '(含 0 => 不显著)'}")

    # ---- 5) 弱类错例归因 ----
    print("\n=== 12 弱类错例归因（v9 视角）===")
    weak_notes = {}
    for role in WEAK_CLASSES:
        lab = cur_role_label.get(role)
        if lab is None:
            continue
        cls_rows = [r for r in ok_rows if r["true_label"] == lab]
        if not cls_rows:
            continue
        errs = [r for r in cls_rows if not r["ok_v9"]]
        imp = [r for r in cls_rows if not r["ok_v7"] and r["ok_v9"]]  # v9 修复的
        confusions = {}
        for r in errs:
            confusions[r["pred_v9_role"]] = confusions.get(r["pred_v9_role"], 0) + 1
        top_conf = sorted(confusions.items(), key=lambda kv: -kv[1])[:3]
        weak_notes[role] = {
            "n": len(cls_rows), "n_err": len(errs),
            "err_rate": round(len(errs) / len(cls_rows), 3),
            "v9_fixed_from_v7": len(imp),
            "top_confusions": top_conf,
            "sample_errors": [{"img": r["path"].split("/")[-1], "pred": r["pred_v9_role"]} for r in errs[:3]],
        }
        print(f"  {role:18s} n={len(cls_rows):3d} 错={len(errs):3d} ({len(errs)/len(cls_rows)*100:4.1f}%) "
              f"v9修复={len(imp):2d} 最混淆→{[f'{k}({v})' for k, v in top_conf]}")

    # ---- 6) 落盘 ----
    report = {
        "meta": {"n": len(ok_rows), "skipped": skipped, "device": device,
                 "test_split": "seed42/test.json (167类)", "v7_space": "seed42_backup_pre_167clean (171类)",
                 "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds")},
        "top1": {"v7": round(top1_v7, 4), "v9": round(top1_v9, 4),
                 "delta_points": round(delta * 100, 3)},
        "four_quadrant": {"both_ok": a, "v7_ok_v9_bad": b, "v7_bad_v9_ok": c, "both_bad": d},
        "mcnemar": {"p_value": round(mcnemar_p, 6) if mcnemar_p is not None else None,
                    "n_discordant": b + c},
        "bootstrap": {"n": args.bootstrap_n, "ci95_low": round(lo * 100, 3),
                      "ci95_high": round(hi * 100, 3), "significant": bool(lo > 0 or hi < 0)},
        "weak_class_attribution": weak_notes,
        "per_sample": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n[out] 报告已保存: {out}")


if __name__ == "__main__":
    main()
