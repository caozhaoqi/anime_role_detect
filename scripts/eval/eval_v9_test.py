#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v9 held-out Test 评测器（167 类全量契约口径，复用 v7/v8 诚实基线纪律）。

设计要点
--------
- 模型结构：与 train_efficientnet_b3.create_efficientnet_b3 **逐字节一致**
  (Dropout0.3 -> Linear768 -> ReLU -> BN768 -> Dropout0.15 -> Linear N)，
  因此 strict=True 加载 v9 权重不会静默错配。
- 预处理：src.common.preprocess.build_eval_transform(256) —— canonical
  Resize(288)->CenterCrop(256)，与 v9 训练期 val 变换一致（非 legacy Resize(256)）。
- 权重：models/efficientnet_b3_v9/model_best.pth，strict=True 加载 model_state_dict。
- 口径：v9 在 seed42（167 类，min-samples=0）上训练，test.json 的 label 即为
  seed42 连续编号 0..166，与 v9 输出索引逐位对齐，**不做任何子集过滤/剔除**。
- 指标：Top-1 / Top-5 / MacroF1(all167) / MacroF1(present) / weighted-F1 / balanced-acc。
- 同时跑 canonical 与 legacy_resize256 两路，佐证预处理一致性（v7 审计发现
  legacy 改长宽比会使 val 差 1.77~2.02 点，故对外只报 canonical）。

--compare-v7（可选）：额外在 v7 冻结快照（171 类 pre-167-clean 空间）上对同样的
167 个共享角色跑 test，计算 v9 - v7 的同口径差值。⚠️ 注意：v9 的 test 分区
split_hash=ecbe69e0 与 v7 的 71b7101b 不一致（二者 test 不是同一份文件），
故该差值仅作近似参考，正式对外报数只用 v9 自身 held-out test（canonical）。

用法
----
  .venv/bin/python scripts/eval/eval_v9_test.py --device mps
  .venv/bin/python scripts/eval/eval_v9_test.py --device cpu --max-samples 64   # 冒烟
  .venv/bin/python scripts/eval/eval_v9_test.py --device mps --compare-v7
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.preprocess import build_eval_transform, ensure_rgb  # noqa: E402

# v7 冻结快照（171 类 pre-167-clean 空间），与 v7 审计脚本一致
V7_SPLIT_DIR = ROOT / "data" / "splits" / "seed42_backup_pre_167clean_20260810_144211"
V7_MODEL_DIR = ROOT / "models" / "efficientnet_b3_v7"
V7_NAME2IDX = ROOT / "outputs" / "v7_audit" / "name2idx.json"

LEGACY_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes: int) -> nn.Module:
    """与 train_efficientnet_b3.create_efficientnet_b3(num_classes, weights=None) 完全一致。"""
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


# 超大图护栏：测试集中存在 >49MP 的图（如 akemi_homura/7000610.jpg），
# 直接 ensure_rgb+变换会撑爆内存。模型只需 256×256，故超长边先缩到 MAX_SIDE。
MAX_SIDE = 1280


class ManifestDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        try:
            with Image.open(p) as im:
                im = im.copy()  # 强制解码进内存，避免惰性文件句柄在 transform/convert 时 seek 失败
                if max(im.size) > MAX_SIDE:
                    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                img = ensure_rgb(im)
            return self.tf(img), int(y), 1
        except Exception:
            return torch.zeros(3, 256, 256), int(y), 0


def evaluate(model, items, tf, device, n_classes, keep_cols=None, num_workers=0, batch_size=32):
    """keep_cols: 若提供（v7 子集评测用），模型输出只取这些列再 argmax。"""
    dl = DataLoader(ManifestDS(items, tf), batch_size=batch_size, num_workers=num_workers, shuffle=False)
    ys, p1, p5, ok = [], [], [], []
    with torch.no_grad():
        for x, y, valid in dl:
            out = model(x.to(device)).float().cpu()
            if keep_cols is not None:
                out = out[:, keep_cols]
            t5 = out.topk(5, dim=1).indices
            p1.append(t5[:, 0].numpy())
            p5.append(t5.numpy())
            ys.append(y.numpy())
            ok.append(valid.numpy())
    if not ys:
        return {"n_evaluated": 0, "n_skipped": 0, "empty": True}
    y = np.concatenate(ys)
    pred = np.concatenate(p1)
    top5 = np.concatenate(p5)
    okm = np.concatenate(ok).astype(bool)
    y, pred, top5 = y[okm], pred[okm], top5[okm]

    labels = list(range(n_classes))
    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    present = sup > 0
    return {
        "n_evaluated": int(okm.sum()),
        "n_skipped": int((~okm).sum()),
        "top1": float((pred == y).mean()),
        "top5": float(np.mean([yy in tt for yy, tt in zip(y, top5)])),
        "macro_f1_all": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1_present": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "n_classes_present": int(present.sum()),
        "per_class": {
            i: {"support": int(sup[i]), "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4), "f1": round(float(f1[i]), 4)}
            for i in labels
        },
    }


def load_test_all(test_path: Path, data_dir: Path, max_samples=None):
    """test.json 的 label 是 seed42 连续编号，与 v9 输出索引逐位对齐，全量保留。"""
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
    return items, skipped, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--v9-model-dir", default=str(ROOT / "models" / "efficientnet_b3_v9"))
    ap.add_argument("--split-dir", default=str(ROOT / "data" / "splits" / "seed42"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "final_dataset"))
    ap.add_argument("--compare-v7", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "v9_audit" / "v9_eval_report.json"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader 子进程数（沙箱内存受限时设为 0）")
    ap.add_argument("--max-samples", type=int, default=None, help="冒烟用：最多评测 N 个样本")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    v9_model_dir = Path(args.v9_model_dir)
    split_dir = Path(args.split_dir)
    data_dir = Path(args.data_dir)

    # v9 类数（来自模型自身 class_to_idx，数字键 '0'..'166'）
    v9_map = {str(k): int(v) for k, v in json.load(open(v9_model_dir / "class_to_idx.json")).items()}
    n_classes = len(v9_map)
    assert n_classes == 167, f"v9 应为 167 类，实际 {n_classes}"
    print(f"[info] v9 类数={n_classes}，device={device}")

    # ---- v9 test 评测（全量 167 类）----
    test_path = split_dir / "test.json"
    items9, skip9, n_test_total = load_test_all(test_path, data_dir, args.max_samples)
    print(f"[info] test.json 共 {n_test_total} 条；可用 {len(items9)} 条；缺图跳过 {skip9} 条")

    model9 = build_model(167)
    ck9 = torch.load(v9_model_dir / "model_best.pth", map_location="cpu", weights_only=False)
    ck9_epoch = ck9.get("epoch")
    ck9_best_acc = ck9.get("best_acc")
    ck9_best_macro_f1 = ck9.get("best_macro_f1")
    missing = model9.load_state_dict(ck9["model_state_dict"], strict=True)
    print(f"[ok] v9 strict load, epoch={ck9_epoch} "
          f"best_acc={ck9_best_acc:.6f} best_macro_f1={ck9_best_macro_f1} {missing}")
    # 释放 optimizer_state_dict 等冗余状态，避免沙箱内存上限被撑爆
    del ck9
    import gc
    gc.collect()
    model9.to(device).eval()

    canon = build_eval_transform(256)
    out = {"device": str(device), "v9_classes": n_classes,
           "n_test_total": n_test_total, "n_skipped_missing_image": skip9,
           "v9_ckpt_epoch": ck9_epoch,
           "v9_ckpt_best_acc": ck9_best_acc,
           "v9_ckpt_best_macro_f1": ck9_best_macro_f1,
           "v9_split_hash": json.load(open(v9_model_dir / "training_results.json")).get("split_hash"),
           "runs": {}}

    for tag, tf in [("canonical", canon), ("legacy_resize256", LEGACY_TF)]:
        r = evaluate(model9, items9, tf, device, 167, num_workers=args.num_workers, batch_size=args.batch_size)
        out["runs"][f"v9_test_{tag}"] = r
        print(f"  v9 test/{tag}: top1={r['top1']:.4%} top5={r['top5']:.4%} "
              f"macroF1(all167)={r['macro_f1_all']:.4f} "
              f"macroF1(present)={r['macro_f1_present']:.4f} "
              f"wF1={r['weighted_f1']:.4f} balAcc={r['balanced_acc']:.4f} "
              f"present={r['n_classes_present']}/167 skipped={r['n_skipped']}", flush=True)

    # v9 两路算完即刻落盘（即便后续 v7 对比被中断，v9 诚实 TEST 报告也已保全）
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote (v9 base) ->", args.out, flush=True)

    # ---- 可选：v7 同口径对比（167 共享角色，v7 冻结快照）----
    if args.compare_v7:
        if not V7_SPLIT_DIR.exists():
            raise SystemExit(f"[FATAL] v7 冻结快照不存在: {V7_SPLIT_DIR}")
        name2idx_v7 = json.load(open(V7_NAME2IDX, encoding="utf-8"))            # 角色名 -> v7_label(171)
        seed42_c2i = json.load(open(split_dir / "class_to_idx.json", encoding="utf-8"))  # 角色名 -> seed42_label
        seed42_i2n = {int(v): k for k, v in seed42_c2i.items()}                 # seed42_label -> 角色名
        # 167 个 v9 类（seed42_label 顺序）对应角色名 -> v7 列
        v9_labels = sorted(int(k) for k in v9_map.keys())
        shared_names = [seed42_i2n[L] for L in v9_labels]
        missing_names = [n for n in shared_names if n not in name2idx_v7]
        assert not missing_names, f"以下 v9 角色在 v7 name2idx 缺映射: {missing_names[:5]}"
        v7_cols = [name2idx_v7[n] for n in shared_names]                        # v7 输出中对应列（167 顺序）
        v7_col_to_sub = {c: i for i, c in enumerate(v7_cols)}                   # v7_label -> 167子集idx

        v7_rows = json.load(open(V7_SPLIT_DIR / "test.json", encoding="utf-8"))
        items7, oos7 = [], 0
        for r in v7_rows:
            L = r["label"]
            if L in v7_col_to_sub:
                p = data_dir / r["path"]
                if p.exists():
                    items7.append((str(p), v7_col_to_sub[L]))
                else:
                    oos7 += 1
            else:
                oos7 += 1
            if args.max_samples and len(items7) >= args.max_samples:
                break
        print(f"[info] v7 同口径：v7 test 共 {len(v7_rows)} 条；落入 167 共享角色 {len(items7)} 条；"

              f"其余/缺图剔除 {oos7} 条")

        model7 = build_model(171)
        ck7 = torch.load(V7_MODEL_DIR / "model_best.pth", map_location="cpu", weights_only=False)
        ck7_epoch = ck7.get("epoch")
        ck7_best_acc = ck7.get("best_acc")
        m7 = model7.load_state_dict(ck7["model_state_dict"], strict=True)
        print(f"[ok] v7 strict load, epoch={ck7_epoch} best_acc={ck7_best_acc:.6f} {m7}")
        del ck7
        gc.collect()
        model7.to(device).eval()

        for tag, tf in [("canonical", canon), ("legacy_resize256", LEGACY_TF)]:
            r = evaluate(model7, items7, tf, device, 167, keep_cols=v7_cols, num_workers=args.num_workers, batch_size=args.batch_size)
            out["runs"][f"v7_167class_{tag}"] = r
            print(f"  v7 167class/{tag}: top1={r['top1']:.4%} top5={r['top5']:.4%} "
                  f"macroF1(all167)={r['macro_f1_all']:.4f} "
                  f"macroF1(present)={r['macro_f1_present']:.4f} "
                  f"wF1={r['weighted_f1']:.4f} balAcc={r['balanced_acc']:.4f} "
                  f"present={r['n_classes_present']}/167 skipped={r['n_skipped']}", flush=True)

        # 差值（v9 - v7，同 167 类近似口径；注意分区不同）
        for tag in ("canonical", "legacy_resize256"):
            a = out["runs"][f"v9_test_{tag}"]
            b = out["runs"][f"v7_167class_{tag}"]
            out.setdefault("delta_v9_minus_v7", {})[tag] = {
                "top1": round(a["top1"] - b["top1"], 4),
                "macro_f1_all": round(a["macro_f1_all"] - b["macro_f1_all"], 4),
                "macro_f1_present": round(a["macro_f1_present"] - b["macro_f1_present"], 4),
                "balanced_acc": round(a["balanced_acc"] - b["balanced_acc"], 4),
            }
            print(f"  Δ(v9-v7)/{tag}: top1={out['delta_v9_minus_v7'][tag]['top1']:+.4%} "
                  f"macroF1={out['delta_v9_minus_v7'][tag]['macro_f1_all']:+.4f} "
                  f"balAcc={out['delta_v9_minus_v7'][tag]['balanced_acc']:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote ->", args.out)


if __name__ == "__main__":
    main()
