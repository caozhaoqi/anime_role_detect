#!/usr/bin/env python3
"""
从 ModelScope 下载 stable-diffusion-v1-5 (diffusers 格式) 到本地目录。

背景：本沙箱无法访问 HuggingFace（代理对 hf.co 返回 000 / 无直连出口），
但 modelscope.cn 走同一代理可达。故改用 ModelScope 作为 SD1.5 底座来源。

用法：
  ./t2i-mac/bin/python scripts/t2i/download_sd15_modelscope.py \
      --repo AI-ModelScope/stable-diffusion-v1-5 \
      --local-dir models_cache/stable-diffusion-v1-5
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="AI-ModelScope/stable-diffusion-v1-5")
    parser.add_argument(
        "--local-dir", default="models_cache/stable-diffusion-v1-5"
    )
    args = parser.parse_args()

    from modelscope.hub.snapshot_download import snapshot_download

    local_dir = os.path.abspath(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"[download] repo={args.repo} -> {local_dir}", flush=True)
    path = snapshot_download(
        args.repo,
        local_dir=local_dir,
        # 只下 diffusers 必需的权重与配置，跳过 .bin 双份与用不到的 safety_checker
        ignore_file_pattern=["*.bin", "safety_checker/*"],
    )
    print(f"[download] DONE -> {path}", flush=True)

    # 校验关键文件
    need = [
        "model_index.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "tokenizer/vocab.json",
        "scheduler/scheduler_config.json",
    ]
    print("[download] 校验关键文件:", flush=True)
    for f in need:
        p = os.path.join(local_dir, f)
        ok = os.path.exists(p)
        size = os.path.getsize(p) if ok else 0
        print(f"  [{'OK' if ok else 'MISSING'}] {f} ({size/1e6:.1f} MB)" if ok
              else f"  [MISSING] {f}", flush=True)


if __name__ == "__main__":
    main()
