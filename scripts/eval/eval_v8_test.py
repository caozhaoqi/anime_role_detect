#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v8 held-out Test 评测器（复用 v7 诚实基线口径）。

设计要点
--------
- 模型结构：与 train_efficientnet_b3.create_efficientnet_b3 **逐字节一致**
  (Dropout0.3 -> Linear768 -> ReLU -> BN768 -> Dropout0.15 -> Linear N)，
  因此 strict=True 加载 v8 权重不会静默错配。
- 预处理：src.common.preprocess.build_eval_transform(256) —— canonical
  Resize(288)->CenterCrop(256)，与 v7 训练期 val 变换一致（非 legacy Resize(256)）。
- 权重：models/efficientnet_b3_v8/model_best.pth，strict=True 加载 model_state_dict。
- 指标：Top-1 / Top-5 / MacroF1(all48) / MacroF1(present) / weighted-F1 / balanced-acc。

口径桥接（最关键，错则成糊涂账）
--------------------------------
v8 的 class_to_idx.json 的 **key 是 seed42 的连续 label(0..166) 中的 48 个**，
value 是 0..47 新索引（已与 seed42/class_to_idx.json 核对：Furina=5, Klee=9, Plana=10…）。
test.json 的 label 也是 seed42 连续编号。故：
  test 样本 label L 若 str(L) 在 v8_map → new_label = v8_map[str(L)]，纳入评测；
  否则该样本属被砍的长尾类，v8 从未见过 → 剔除（n_out_of_scope）。
分母严格锁定 48 类，绝不混入 v8 不可预测的类。

--compare-v7：额外加载 v7(171 类 pre-167-clean 权重) 在 48 个共享角色上的 test 指标，
做同口径对比，输出两者差值（=v8 真实模型增量）。v7 test 集用其冻结快照，仅取
48 共享角色的样本；两模型在 48 类上分母相同，可比（近似，见报告局限）。
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


class ManifestDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        try:
            with Image.open(p) as im:
                img = ensure_rgb(im.copy())
            return self.tf(img), int(y), 1
        except Exception:
            return torch.zeros(3, 256, 256), int(y), 0


def evaluate(model, items, tf, device, n_classes, keep_cols=None):
    """keep_cols: 若提供（v7 子集评测用），模型输出只取这些列再 argmax。"""
    dl = DataLoader(ManifestDS(items, tf), batch_size=32, num_workers=4, shuffle=False)
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


def load_and_filter_v8(test_path: Path, v8_map: dict, max_samples=None):
    """test.json 的 label 是 seed42 连续编号；只保留 label 落在 v8 48 类内的样本。"""
    rows = json.load(open(test_path, encoding="utf-8"))
    kept, oos = [], 0
    for r in rows:
        L = r["label"]
        if str(L) in v8_map:
            kept.append((str(ROOT / "data" / "final_dataset" / r["path"]), v8_map[str(L)]))
        else:
            oos += 1
        if max_samples and len(kept) >= max_samples:
            break
    return kept, oos, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--v8-model-dir", default=str(ROOT / "models" / "efficientnet_b3_v8"))
    ap.add_argument("--v8-class-map", default=None)
    ap.add_argument("--split-dir", default=str(ROOT / "data" / "splits" / "seed42"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "final_dataset"))
    ap.add_argument("--compare-v7", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "v8_audit" / "v8_eval_report.json"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-samples", type=int, default=None, help="冒烟用：每个模型最多评测 N 个样本")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    v8_model_dir = Path(args.v8_model_dir)
    v8_map_path = Path(args.v8_class_map) if args.v8_class_map else v8_model_dir / "class_to_idx.json"
    split_dir = Path(args.split_dir)
    data_dir = Path(args.data_dir)

    v8_map = {str(k): int(v) for k, v in json.load(open(v8_map_path)).items()}
    n_classes = len(v8_map)
    assert n_classes == 48, f"v8 应为 48 类，实际 {n_classes}（与 training_results 不一致）"
    print(f"[info] v8 类数={n_classes}，device={device}")

    # ---- v8 test 评测 ----
    test_path = split_dir / "test.json"
    items8, oos8, n_test_total = load_and_filter_v8(test_path, v8_map, args.max_samples)
    print(f"[info] test.json 共 {n_test_total} 条；落入 v8 48 类 {len(items8)} 条；"
          f"被砍长尾类(剔除) {oos8} 条")

    model8 = build_model(48)
    ck8 = torch.load(v8_model_dir / "model_best.pth", map_location="cpu", weights_only=False)
    missing = model8.load_state_dict(ck8["model_state_dict"], strict=True)
    print(f"[ok] v8 strict load, epoch={ck8.get('epoch')} "
          f"best_acc={ck8.get('best_acc'):.6f} best_macro_f1={ck8.get('best_macro_f1')} {missing}")
    model8.to(device).eval()

    canon = build_eval_transform(256)
    out = {"device": str(device), "v8_classes": n_classes,
           "n_test_total": n_test_total, "n_out_of_scope": oos8,
           "v8_ckpt_epoch": ck8.get("epoch"),
           "v8_ckpt_best_acc": ck8.get("best_acc"),
           "v8_ckpt_best_macro_f1": ck8.get("best_macro_f1"),
           "runs": {}}

    for tag, tf in [("canonical", canon), ("legacy_resize256", LEGACY_TF)]:
        r = evaluate(model8, items8, tf, device, 48)
        out["runs"][f"v8_test_{tag}"] = r
        print(f"  v8 test/{tag}: top1={r['top1']:.4%} top5={r['top5']:.4%} "
              f"macroF1(all48)={r['macro_f1_all']:.4f} "
              f"macroF1(present)={r['macro_f1_present']:.4f} "
              f"wF1={r['weighted_f1']:.4f} balAcc={r['balanced_acc']:.4f} "
              f"present={r['n_classes_present']}/48 skipped={r['n_skipped']}", flush=True)

    # ---- 可选：v7 同口径对比（48 共享角色）----
    if args.compare_v7:
        if not V7_SPLIT_DIR.exists():
            raise SystemExit(f"[FATAL] v7 冻结快照不存在: {V7_SPLIT_DIR}")
        name2idx_v7 = json.load(open(V7_NAME2IDX, encoding="utf-8"))       # 角色名 -> v7_label(171)
        seed42_c2i = json.load(open(split_dir / "class_to_idx.json", encoding="utf-8"))  # 角色名 -> seed42_label
        seed42_i2n = {int(v): k for k, v in seed42_c2i.items()}            # seed42_label -> 角色名
        # 48 共享角色名（按 v8 新索引顺序）
        shared_names = [seed42_i2n[int(k)] for k in sorted(v8_map, key=lambda k: v8_map[k])]
        missing_names = [n for n in shared_names if n not in name2idx_v7]
        assert not missing_names, f"以下 v8 角色在 v7 name2idx 中缺映射: {missing_names[:5]}"
        # v7 输出中这 48 个角色对应的列（顺序对齐 v8 new_idx）
        v7_cols = [name2idx_v7[n] for n in shared_names]
        v7_col_to_sub = {c: i for i, c in enumerate(v7_cols)}             # v7_label -> 48子集idx

        v7_rows = json.load(open(V7_SPLIT_DIR / "test.json", encoding="utf-8"))
        items7, oos7 = [], 0
        for r in v7_rows:
            L = r["label"]
            if L in v7_col_to_sub:
                items7.append((str(data_dir / r["path"]), v7_col_to_sub[L]))
            else:
                oos7 += 1
            if args.max_samples and len(items7) >= args.max_samples:
                break
        print(f"[info] v7 同口径：v7 test 共 {len(v7_rows)} 条；"
              f"共享 48 角色 {len(items7)} 条；其余剔除 {oos7} 条")

        model7 = build_model(171)
        ck7 = torch.load(V7_MODEL_DIR / "model_best.pth", map_location="cpu", weights_only=False)
        m7 = model7.load_state_dict(ck7["model_state_dict"], strict=True)
        print(f"[ok] v7 strict load, epoch={ck7.get('epoch')} best_acc={ck7.get('best_acc'):.6f} {m7}")
        model7.to(device).eval()

        for tag, tf in [("canonical", canon), ("legacy_resize256", LEGACY_TF)]:
            r = evaluate(model7, items7, tf, device, 48, keep_cols=v7_cols)
            out["runs"][f"v7_48class_{tag}"] = r
            print(f"  v7 48class/{tag}: top1={r['top1']:.4%} top5={r['top5']:.4%} "
                  f"macroF1(all48)={r['macro_f1_all']:.4f} "
                  f"macroF1(present)={r['macro_f1_present']:.4f} "
                  f"wF1={r['weighted_f1']:.4f} balAcc={r['balanced_acc']:.4f} "
                  f"present={r['n_classes_present']}/48 skipped={r['n_skipped']}", flush=True)

        # 差值（v8 - v7，同 48 类口径）
        for tag in ("canonical", "legacy_resize256"):
            a = out["runs"][f"v8_test_{tag}"]
            b = out["runs"][f"v7_48class_{tag}"]
            out.setdefault("delta_v8_minus_v7", {})[tag] = {
                "top1": round(a["top1"] - b["top1"], 4),
                "macro_f1_all": round(a["macro_f1_all"] - b["macro_f1_all"], 4),
                "macro_f1_present": round(a["macro_f1_present"] - b["macro_f1_present"], 4),
                "balanced_acc": round(a["balanced_acc"] - b["balanced_acc"], 4),
            }
            print(f"  Δ(v8-v7)/{tag}: top1={out['delta_v8_minus_v7'][tag]['top1']:+.4%} "
                  f"macroF1={out['delta_v8_minus_v7'][tag]['macro_f1_all']:+.4f} "
                  f"balAcc={out['delta_v8_minus_v7'][tag]['balanced_acc']:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote ->", args.out)


if __name__ == "__main__":
    main()
