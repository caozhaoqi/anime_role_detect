#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP kNN 错标检测器 —— 判别力标定 / 对照实验
=============================================

背景
----
`audit_label_quality.py` 的 A1 路 (CLIP kNN 近邻投票) 在 12 个弱类上把 93%
的图都判成"错标嫌疑"(foreign_ratio > 0.6)。一个把几乎所有样本都报警的检测器
是**没有判别力**的，不能直接拿去下结论。

本脚本做对照实验回答一个问题:
    高 foreign_ratio 是「弱类特有的标签噪声」，还是「CLIP ViT-B/32 在本数据集
    上的普遍行为」？

方法
----
同一套参考底库 (171 类, 每类等量上限, 避免大类主导近邻投票)，分别计算:
  * WEAK   组: 12 个 test F1 < 0.45 的弱类
  * STRONG 组: 若干 test F1 >= 0.75 的强类
两组 foreign_ratio 分布若无显著差异 => A1 阈值 0.6 无判别力，须重新标定。

用法
----
  ./.venv/bin/python scripts/data_processing/audit_knn_calibration.py
  ./.venv/bin/python scripts/data_processing/audit_knn_calibration.py \
      --ref-per-class 25 --per-class 40 --out outputs/label_audit/calibration.json

本脚本只读数据，不修改任何图片。
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

WEAK_DEFAULT = [
    "sorasaki_hina", "ram_(re:zero)", "hiiragi_kagami", "kaname_madoka",
    "lynx", "silver_wolf", "lisa", "yuuki_asuna", "asagi_mutsuki",
    "kagura_(onmyouji)", "tsushima_yoshiko", "paimon",
]
STRONG_DEFAULT = [
    "sigewinne", "ilulu_(maid_dragon)", "collei", "hu_tao",
    "hachikuji_mayoi", "corin_wickes", "sayu", "ningguang",
]


def load_audit_module():
    path = os.path.join(HERE, "audit_label_quality.py")
    spec = importlib.util.spec_from_file_location("audit_label_quality", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def describe(arr: np.ndarray) -> dict:
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "frac_gt_0.6": round(float((arr > 0.6).mean()), 4),
        "frac_eq_1.0": round(float((arr >= 0.999).mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/final_dataset")
    ap.add_argument("--ref-per-class", type=int, default=25)
    ap.add_argument("--per-class", type=int, default=40, help="每类评估图片上限")
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weak", default=",".join(WEAK_DEFAULT))
    ap.add_argument("--strong", default=",".join(STRONG_DEFAULT))
    ap.add_argument("--out", default="outputs/label_audit/knn_calibration.json")
    args = ap.parse_args()

    os.chdir(PROJECT_ROOT)
    alq = load_audit_module()
    alq.setup_logging(False)

    weak = [c.strip() for c in args.weak.split(",") if c.strip()]
    strong = [c.strip() for c in args.strong.split(",") if c.strip()]

    device = alq.pick_device(args.device)
    print(f"[calib] device={device} ref_per_class={args.ref_per_class} "
          f"per_class={args.per_class} K={args.knn_k}", flush=True)

    knn = alq.ClipKnn(device=device, batch_size=args.batch_size, workers=args.workers)
    if not knn.load():
        print("[calib] CLIP 不可用，退出", flush=True)
        return 1

    ref_paths, ref_labels = alq.build_reference_set(
        args.data_dir, weak, "all", args.ref_per_class
    )
    print(f"[calib] 参考底库 {len(ref_paths)} 张 / {len(set(ref_labels))} 类", flush=True)
    knn.build_reference(ref_paths, ref_labels)

    def eval_group(classes, tag):
        per = {}
        for cls in classes:
            cdir = os.path.join(args.data_dir, cls)
            if not os.path.isdir(cdir):
                print(f"[calib] 跳过不存在的类 {cls}", flush=True)
                continue
            paths = alq.list_images(cdir)[: args.per_class]
            if not paths:
                continue
            vecs = knn.encode(paths)
            ratios = []
            for i, p in enumerate(paths):
                q = knn.query(vecs[i], p, cls, args.knn_k)
                if q.get("status") == "ok":
                    ratios.append(q["neighbor_foreign_ratio"])
            if ratios:
                per[cls] = np.array(ratios, dtype=np.float32)
                print(f"[calib] {tag:<6} {cls:<24} n={len(ratios):>3} "
                      f"median={np.median(per[cls]):.2f} "
                      f">0.6={(per[cls] > 0.6).mean():.0%}", flush=True)
        return per

    strong_per = eval_group(strong, "STRONG")
    weak_per = eval_group(weak, "WEAK")

    sa = np.concatenate(list(strong_per.values())) if strong_per else np.array([])
    wa = np.concatenate(list(weak_per.values())) if weak_per else np.array([])

    print("\n" + "=" * 72)
    print("对照实验结论: CLIP kNN foreign_ratio 分布")
    print("=" * 72)
    if sa.size and wa.size:
        print(f"STRONG (F1>=0.75) pooled: {describe(sa)}")
        print(f"WEAK   (F1<0.45)  pooled: {describe(wa)}")
        delta = wa.mean() - sa.mean()
        print(f"\nmean(WEAK) - mean(STRONG) = {delta:+.4f}")
        # Mann-Whitney U (非参数, 不假设正态)
        try:
            from scipy.stats import mannwhitneyu

            u, p = mannwhitneyu(wa, sa, alternative="two-sided")
            print(f"Mann-Whitney U p = {p:.3e}")
        except Exception:
            p = None
        verdict = ("无判别力 (两组几乎相同) -> 阈值 0.6 必须重新标定"
                   if abs(delta) < 0.10 else "有一定判别力")
        print(f"判定: {verdict}")
    else:
        print("样本不足")

    out = {
        "config": {
            "ref_per_class": args.ref_per_class,
            "per_class": args.per_class,
            "knn_k": args.knn_k,
            "n_ref": len(ref_paths),
            "n_ref_classes": len(set(ref_labels)),
            "device": device,
        },
        "strong": {c: describe(a) for c, a in strong_per.items()},
        "weak": {c: describe(a) for c, a in weak_per.items()},
        "pooled": {
            "strong": describe(sa) if sa.size else None,
            "weak": describe(wa) if wa.size else None,
            "mean_delta_weak_minus_strong": (
                round(float(wa.mean() - sa.mean()), 4) if (sa.size and wa.size) else None
            ),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n已写出: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
