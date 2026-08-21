#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2: CLIP 相似度评估——生成图 vs 参考图集的语义相似度（双指标之一）。

背景（T2I 改进方案 S2，2026-08-20）
-----------------------------------
现有 verify_t2i_role.py 只有「是否被 v9 识别为目标角色」的二元判定。
本脚本补上连续分数：CLIP embedding 余弦相似度（生成图 vs data/final_dataset/<role> 参考集），
输出 mean/max/percentile，可与 v9 一致率组成「双指标」质量评估。

与 verify_t2i_role.py 串联（双指标报告）：
  .venv/bin/python scripts/t2i/eval_t2i_similarity.py --image <gen.png> --target-role Bailu --json sim.json
  .venv/bin/python scripts/t2i/verify_t2i_role.py --image <gen.png> --target-role Bailu --json verify.json
  # 两个 json 合并即为完整双指标

健壮性
-------
- CLIP 权重重（~600MB），懒加载；模型不可用时优雅报错（本机 HF 不通需先下载权重）
- 参考图集加载失败（角色无图）时退出并提示

用法
----
  .venv/bin/python scripts/t2i/eval_t2i_similarity.py --image outputs/t2i/Bailu/x.png --target-role Bailu
  .venv/bin/python scripts/t2i/eval_t2i_similarity.py --image-dir outputs/t2i/Bailu --target-role Bailu --json sim_report.json
  .venv/bin/python scripts/t2i/eval_t2i_similarity.py --dry-run                      # 只验证数据通路
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.services.t2i_service import config  # noqa: E402

EXT = (".jpg", ".jpeg", ".png", ".webp")


def cosine(a, b) -> float:
    """余弦相似度（自动 L2 归一化）。"""
    import numpy as np
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_reference_embeddings(embedder, role: str, data_root: Path, limit: int | None):
    """参考图集全部 embed，返回逐图向量列表。"""
    ref_dir = data_root / role
    if not ref_dir.exists():
        raise FileNotFoundError(f"参考图目录不存在: {ref_dir}")
    paths = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in EXT)
    if not paths:
        raise FileNotFoundError(f"{ref_dir} 下无图片")
    if limit:
        paths = paths[:limit]
    vecs = []
    for p in paths:
        v = embedder.embed_image(str(p))
        if v is not None and len(v) > 0:
            vecs.append(v)
    if not vecs:
        raise RuntimeError(f"参考图全部 embed 失败: {ref_dir}")
    return vecs, paths


def main():
    ap = argparse.ArgumentParser(description="CLIP 相似度评估（S2）")
    ap.add_argument("--image", default=None, help="单张生成图")
    ap.add_argument("--image-dir", default=None, help="批量：目录内所有图")
    ap.add_argument("--target-role", required=True, help="目标角色（参考图集 = data/final_dataset/<role>）")
    ap.add_argument("--data-dir", default=str(config.DATASET_ROOT))
    ap.add_argument("--model", default="ViT-B/32", help="CLIP 模型名（默认 ViT-B/32；可 ViT-L/14）")
    ap.add_argument("--ref-limit", type=int, default=32, help="参考图最多取 N 张（控制耗时）")
    ap.add_argument("--json", default=None, help="导出 JSON 报告路径")
    ap.add_argument("--dry-run", action="store_true", help="不加载模型，只验证数据通路")
    args = ap.parse_args()

    if not args.image and not args.image_dir:
        print("[ERROR] 必须提供 --image 或 --image-dir 之一")
        sys.exit(1)

    data_root = Path(args.data_dir)
    if not data_root.exists():
        print(f"[ERROR] 数据目录不存在: {data_root}")
        sys.exit(1)

    # 收集待评估图
    targets = []
    if args.image:
        targets = [Path(args.image)]
    else:
        d = Path(args.image_dir)
        targets = sorted(p for p in d.iterdir() if p.suffix.lower() in EXT) if d.exists() else []
    if not targets:
        print("[ERROR] 未找到待评估图")
        sys.exit(1)

    if args.dry_run:
        ref_dir = data_root / args.target_role
        n_ref = len([p for p in ref_dir.iterdir() if p.suffix.lower() in EXT]) if ref_dir.exists() else 0
        print(f"[dry-run] 参考图集 {args.target_role}: {n_ref} 张 | 待评估图 {len(targets)} 张 | 模型 {args.model}")
        print("[dry-run] 数据通路 OK（未加载模型）")
        return

    # 懒加载 CLIP
    try:
        from src.core.recognition.clip_embedder import CLIPEmbedder
        embedder = CLIPEmbedder(model_name=args.model)
        embedder.initialize()
        print(f"[clip] 模型 {args.model} 已加载")
    except Exception as e:
        print(f"[ERROR] CLIP 模型加载失败: {e}")
        print("        本机 HF 不通——需先经 ModelScope 下载 CLIP 权重（openai/clip-vit-base-patch32）")
        sys.exit(2)

    # 参考集 embedding
    try:
        ref_vecs, ref_paths = load_reference_embeddings(embedder, args.target_role, data_root, args.ref_limit)
    except Exception as e:
        print(f"[ERROR] 参考图 embed 失败: {e}")
        sys.exit(1)
    print(f"[ref] {args.target_role} 参考 {len(ref_vecs)} 张 embed 完成")

    # 逐目标图评分
    import numpy as np
    rows = []
    for p in targets:
        v = embedder.embed_image(str(p))
        if v is None or len(v) == 0:
            rows.append({"image": p.name, "error": "embed 失败", "score_mean": None})
            continue
        scores = [cosine(v, rv) for rv in ref_vecs]
        arr = np.asarray(scores)
        row = {
            "image": p.name,
            "score_mean": round(float(arr.mean()), 4),
            "score_max": round(float(arr.max()), 4),
            "score_p50": round(float(np.percentile(arr, 50)), 4),
            "score_p95": round(float(np.percentile(arr, 95)), 4),
        }
        rows.append(row)
        print(f"  {p.name[:40]:40} mean={row['score_mean']:.3f} max={row['score_max']:.3f}")

    # 汇总
    means = [r["score_mean"] for r in rows if r.get("score_mean") is not None]
    summary = {
        "target_role": args.target_role,
        "model": args.model,
        "n_images": len(rows),
        "n_scored": len(means),
        "mean_of_means": round(float(np.mean(means)), 4) if means else None,
        "p95_of_means": round(float(np.percentile(means, 95)), 4) if means else None,
    }
    print(f"\n[summary] {summary['n_scored']}/{summary['n_images']} 图评分，mean_of_means={summary['mean_of_means']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {"summary": summary, "per_image": rows,
                  "note": "双指标之一：与 verify_t2i_role.py 的 v9 一致率合并使用"}
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[json] 报告已导出: {out}")


if __name__ == "__main__":
    main()
