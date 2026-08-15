#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ip_adapter_poc.py — IP-Adapter 免训练动漫角色生成 PoC
========================================================================
核心思路：输入几张角色参考图 → IP-Adapter 把角色一致性「注入」SD1.5
          → 直接生成该角色，跳过 LoRA 训练环节。

与 LoRA 路线对比：
    LoRA  : 每角色需训练一个小 LoRA(~6MB)，一致性 95%+，但慢（CPU 训练数小时）
    IP-Adapter: 免训练、秒级出图，一致性 70~85%，适合快速探索/试角色长相

本脚本两个 stage：
    download : 从 ModelScope 拉 IP-Adapter 权重（★ 这就是「能不能通」的测试点）
    generate : 加载 SD1.5 + IP-Adapter，吃参考图生成

========================================================================
⚠️ 沙箱/网络约束（重要）
------------------------------------------------------------------------
- HuggingFace 在本机沙箱被完全封锁（代理无 HF 路由、无直连），所以权重
  **只能从 ModelScope 拉**（代理 127.0.0.1:7890 可达，已验证可用）。
- 跑本脚本前请确认：① 代理软件已开且 127.0.0.1:7890 可达；
  ② 使用你训练 LoRA 时同一个 python（你的 .venv 或 t2i-mac venv，
     需装有 diffusers / transformers / modelscope / torch）。

========================================================================
用法（用你训练 LoRA 的 python 跑）
------------------------------------------------------------------------
# 1) 先测「能不能通」——拉权重并校验文件（默认就是这步）
.venv/bin/python scripts/t2i/ip_adapter_poc.py --stage download

# 2) 用参考图生成（默认吃 data/final_dataset/amber/ 的图）
.venv/bin/python scripts/t2i/ip_adapter_poc.py --stage generate --role amber \
    --out outputs/t2i_ip/amber

# 3) 生成并自动用 v9 校验一致性（可选）
.venv/bin/python scripts/t2i/ip_adapter_poc.py --stage generate --role amber \
    --verify

# 4) 一步到位（先下载再生成）
.venv/bin/python scripts/t2i/ip_adapter_poc.py --stage all --role amber
"""

import argparse
import os
import sys
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parents[2]
SD15_DIR = ROOT / "models_cache" / "stable-diffusion-v1-5"
LOCAL_IP_DIR = ROOT / "models_cache" / "ip-adapter"      # snapshot_download 落盘根
IP_MODELS_DIR = LOCAL_IP_DIR / "models"                  # 含 ip-adapter-plus_sd15.bin / image_encoder
DATASET_ROOT = ROOT / "data" / "final_dataset"

# ModelScope 上的 IP-Adapter 官方镜像（HF 被封，只能走这个）
IP_REPO = "AI-ModelScope/IP-Adapter"
WEIGHT_NAME = "ip-adapter-plus_sd15.bin"   # plus 版对动漫风格更稳；sdxl 另说

DEFAULT_DEVICE = "auto"   # auto = 有 MPS 用 MPS，否则 CPU


def resolve_device(req: str) -> str:
    if req and req != "auto":
        return req
    import torch
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ======================================================================
# Stage 1: 下载（「能不能通」测试点）
# ======================================================================
def stage_download(args):
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("✗ 未安装 modelscope，请先: pip install modelscope")
        sys.exit(1)

    LOCAL_IP_DIR.mkdir(parents=True, exist_ok=True)

    # 仅拉我们需要的两个东西，避免把整个仓库（含 SDXL 等）全下下来
    allow = [
        "models/ip-adapter-plus_sd15.bin",
        "models/image_encoder/*",
    ]

    print(f"[dl] 从 ModelScope 拉取 {IP_REPO}")
    print(f"[dl] 仅取 {WEIGHT_NAME} + image_encoder（需代理 127.0.0.1:7890 可达）")
    print(f"[dl] 落盘目录: {LOCAL_IP_DIR}")
    try:
        local = snapshot_download(
            IP_REPO,
            local_dir=str(LOCAL_IP_DIR),
            allow_patterns=allow,
        )
    except Exception as e:  # noqa: BLE001
        print(f"\n✗ 下载失败: {type(e).__name__}: {e}")
        print("  可能原因：① 代理未开 / 127.0.0.1:7890 不可达；② ModelScope 该仓库/文件路径有变动")
        print("  代理自查: curl -x http://127.0.0.1:7890 https://www.modelscope.cn -I")
        sys.exit(1)

    print(f"[dl] 完成 -> {local}")

    binp = IP_MODELS_DIR / WEIGHT_NAME
    enc = IP_MODELS_DIR / "image_encoder"
    ok = True
    if binp.exists():
        mb = binp.stat().st_size / 1e6
        print(f"  ✅ 权重: {binp}  ({mb:.1f} MB)")
    else:
        print(f"  ❌ 未找到权重 {binp}")
        ok = False
    if enc.exists() and any(enc.iterdir()):
        print(f"  ✅ 图像编码器: {enc}")
    else:
        print(f"  ❌ 未找到 image_encoder {enc}")
        ok = False

    if ok:
        print("\n🎉 下载成功，链路已通！下一步跑生成：")
        role = args.role or "amber"
        print(f"   .venv/bin/python scripts/t2i/ip_adapter_poc.py --stage generate --role {role}")
    else:
        print("\n⚠️ 文件不全，无法进入生成阶段。")
        sys.exit(1)


# ======================================================================
# Stage 2: 生成
# ======================================================================
def stage_generate(args):
    if not (IP_MODELS_DIR / WEIGHT_NAME).exists():
        print(f"✗ 未找到权重 {IP_MODELS_DIR / WEIGHT_NAME}")
        print("  请先: .venv/bin/python scripts/t2i/ip_adapter_poc.py --stage download")
        sys.exit(1)
    if not SD15_DIR.exists():
        print(f"✗ 未找到 SD1.5 底座 {SD15_DIR}")
        print("  请先跑 download_sd15_modelscope.py 拉取底座。")
        sys.exit(1)

    try:
        import torch
        from PIL import Image
        from diffusers import StableDiffusionPipeline
        from transformers import CLIPVisionModelWithProjection
    except (ImportError, AttributeError) as e:
        print(f"\n✗ 导入失败: {type(e).__name__}: {e}")
        print("   多半是 torch 版本过低 —— diffusers 0.30.1 需要 torch>=2.1（缺 torch.xpu 属性）。")
        print("   请用你训练 LoRA 时那个 venv（如 t2i-mac）来跑本脚本，不要用在主项目 .venv。")
        print("   自查版本: <venv>/bin/python -c \"import torch,diffusers;print(torch.__version__,diffusers.__version__)\"")
        sys.exit(1)

    device = resolve_device(args.device)
    dtype = torch.float32  # CPU/MPS 最稳；fp16 需要 CUDA
    print(f"[init] device={device}  dtype={dtype}")

    print(f"[load] SD1.5 底座: {SD15_DIR}")
    pipe = StableDiffusionPipeline.from_pretrained(
        str(SD15_DIR),
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    print(f"[load] IP-Adapter 权重: {IP_MODELS_DIR / WEIGHT_NAME}")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        str(IP_MODELS_DIR / "image_encoder"), torch_dtype=dtype
    )
    pipe.load_ip_adapter(
        str(IP_MODELS_DIR),
        subfolder="",
        weight_name=WEIGHT_NAME,
        image_encoder=image_encoder,
    )
    pipe.set_ip_adapter_scale(args.scale)
    print(f"[init] IP-Adapter scale={args.scale}")

    # ---- 参考图 ----
    ref_dir = Path(args.ref) if args.ref else (DATASET_ROOT / args.role)
    if not ref_dir.exists():
        print(f"✗ 参考图目录不存在: {ref_dir}")
        sys.exit(1)
    exts = (".jpg", ".jpeg", ".png", ".webp")
    paths = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        print(f"✗ {ref_dir} 下无图片")
        sys.exit(1)
    refs = [Image.open(p).convert("RGB") for p in paths]
    print(f"[ref] 载入 {len(refs)} 张参考图: {ref_dir}")

    # IP-Adapter Plus 多图聚合：参考图过多/风格混杂反而稀释身份，且一次性编码 40 张
    # 对 MPS 16GB 有压力。默认随机采样上限更稳、更快；--num-ref 0 表示用全部。
    if args.num_ref and len(refs) > args.num_ref:
        import random as _rnd
        rng = _rnd.Random(42)
        refs = rng.sample(refs, args.num_ref)
        print(f"[ref] 采样 {len(refs)} 张用于身份注入（--num-ref 控制，0=全部）")

    # IP-Adapter 的文本 prompt 不应出现角色名，否则 CLIP 会被角色名（也可能是普通词）带偏；
    # 角色身份应完全由参考图注入。用户需要性别/姿势时通过 --prompt 覆盖。
    prompt = args.prompt or "solo character, anime style, high quality, detailed background"
    neg = args.negative or "low quality, blurry, extra characters, multiple people, deformed, watermark"
    print(f"[prompt] {prompt}")
    print(f"[neg]    {neg}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num):
        print(f"[gen] 第 {i+1}/{args.num} 张：向 IP-Adapter 注入 {len(refs)} 张参考图 (ip_adapter_image=[list])")
        out = pipe(
            prompt=prompt,
            negative_prompt=neg,
            ip_adapter_image=[refs],   # 外层 list=每个 adapter 一组参考图; 内层 refs=40 张参考图
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=torch.manual_seed(args.seed + i),
        ).images[0]
        save_path = out_dir / f"{args.role}_ip_{i+1}.png"
        out.save(save_path)
        print(f"[gen] 已保存 {save_path}")

    print(f"\n🎉 生成完成，共 {args.num} 张 -> {out_dir}")

    # ---- 可选：用 v9 校验一致性 ----
    if args.verify:
        verify_cmd = [
            sys.executable, str(ROOT / "scripts" / "t2i" / "verify_t2i_role.py"),
            "--target-role", args.role,
            "--image", str(out_dir / f"{args.role}_ip_1.png"),
        ]
        print("\n[verify] 调用 v9 校验一致性 ...")
        try:
            os.system(" ".join(verify_cmd))
        except Exception as e:  # noqa: BLE001
            print(f"  verify 调用失败（可手动跑）: {e}")
    else:
        print("\n（未加 --verify）手动校验命令：")
        print(f"   .venv/bin/python scripts/t2i/verify_t2i_role.py "
              f"--image {out_dir}/{args.role}_ip_1.png --target-role {args.role}")


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="IP-Adapter 免训练角色生成 PoC")
    ap.add_argument("--stage", choices=["download", "generate", "all"], default="download",
                   help="download=拉权重(测连通性,默认) / generate=生成 / all=都跑")
    ap.add_argument("--role", default="amber", help="角色名，对应 data/final_dataset/<role>/")
    ap.add_argument("--ref", default=None, help="参考图目录(默认 data/final_dataset/<role>/)")
    ap.add_argument("--out", default=None, help="生成图输出目录")
    ap.add_argument("--device", default=DEFAULT_DEVICE, help="cpu / mps / auto(默认)")
    ap.add_argument("--scale", type=float, default=0.6, help="IP-Adapter 强度 0.4~0.8")
    ap.add_argument("--steps", type=int, default=30, help="推理步数")
    ap.add_argument("--cfg", type=float, default=7.5, help="classifier-free guidance")
    ap.add_argument("--num", type=int, default=1, help="生成张数")
    ap.add_argument("--num-ref", type=int, default=12, help="参考图采样上限(0=全部)，过多反稀释身份")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--prompt", default=None, help="自定义 prompt")
    ap.add_argument("--negative", default=None, help="自定义 negative prompt")
    ap.add_argument("--verify", action="store_true", help="生成后用 v9 校验一致性")
    args = ap.parse_args()

    if args.out is None:
        args.out = str(ROOT / "outputs" / "t2i_ip" / (args.role or "out"))

    if args.stage in ("download", "all"):
        stage_download(args)
    if args.stage in ("generate", "all"):
        stage_generate(args)


if __name__ == "__main__":
    main()
