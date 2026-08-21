#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载真 WD ViT v3 权重（SmilingWolf/wd-vit-tagger-v3）。

本机 HF 不通（代理无 HF 路由），ModelScope 走同一代理可达 —— 与 SD1.5 底座
下载同思路（download_sd15_modelscope.py）。

默认从 ModelScope 镜像拉取，落盘到项目 HF 缓存布局：
  huggingface_cache/hub/models--SmilingWolf--wd-vit-tagger-v3/snapshots/<hash>/

用法
----
  .venv/bin/python scripts/t2i/download_wd_vit_modelscope.py                 # 默认 AI-ModelScope 镜像
  .venv/bin/python scripts/t2i/download_wd_vit_modelscope.py --repo SmilingWolf/wd-vit-tagger-v3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_REPO = "AI-ModelScope/wd-vit-tagger-v3"   # ModelScope 常见镜像前缀
TARGET_DIR = ROOT / "huggingface_cache" / "hub" / "models--SmilingWolf--wd-vit-tagger-v3"


def main():
    ap = argparse.ArgumentParser(description="下载真 WD ViT v3 权重（ModelScope）")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="ModelScope repo id")
    ap.add_argument("--local-dir", default=str(TARGET_DIR),
                    help="落盘目录（HF 缓存布局，需含 snapshots/<hash>/）")
    args = ap.parse_args()

    from modelscope.hub.snapshot_download import snapshot_download

    local_dir = os.path.abspath(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    print(f"[download] repo={args.repo} -> {local_dir}", flush=True)
    path = snapshot_download(args.repo, local_dir=local_dir, allow_patterns=["*.safetensors", "*.json", "*.csv"])
    print(f"[download] 完成: {path}")

    # 校验权重真实存在（>1MB）
    w = list(Path(path).rglob("model.safetensors"))
    if w and w[0].stat().st_size > 1_000_000:
        print(f"[verify] 权重 OK: {w[0]} ({w[0].stat().st_size / 1e6:.0f} MB)")
        print("[next] 运行: .venv/bin/python scripts/t2i/enrich_captions.py --role Bailu --limit 3")
    else:
        print("[WARN] 未找到真实权重文件（>1MB），请检查 repo id 是否正确")


if __name__ == "__main__":
    main()
