"""Phase 1 (Mac) - SD1.5 + LoRA 最小可跑训练样例（MPS 友好）。

基于 diffusers 的标准 SD1.5 LoRA 微调：只对 UNet 注意力层注入 LoRA，冻结底座，
训练 (image, caption) 对（由 scripts/t2i/prepare_captions.py 产出 metadata.csv）。

与 SDXL 版的关键差异（适配 Mac MPS + 16GB）：
- 单文本编码器（CLIP ViT-L/14），token 上限 77；底座 runwayml/stable-diffusion-v1-5。
- 分辨率 512（SD1.5 原生），显存占用远低于 SDXL 的 1024。
- 设备自动检测 mps > cuda > cpu。
- MPS 上默认 float32 训练以保证数值稳定（MPS 的 float16 训练偶发 NaN / 算子缺失）；
  可用 --mixed-prec 切回 float16 提速（不保证稳定）。
- VAE 强制 float32 编码，规避 MPS 半精度 NaN。
- 保存用 unet.save_attn_procs（SD1.5 的 pytorch_lora_weights.bin 格式）。

环境：
    /usr/bin/python3 -m venv t2i-mac
    ./t2i-mac/bin/pip install -r scripts/t2i/requirements-t2i-mac.txt

数据通路自检（无需权重，本机即可跑）：
    ./t2i-mac/bin/python scripts/t2i/train_lora_sd15.py \
        --metadata outputs/t2i_phase0/metadata.csv --role amber --dry-run

训练（单角色 PoC，MPS）：
    ./t2i-mac/bin/python scripts/t2i/train_lora_sd15.py \
        --metadata outputs/t2i_phase0/metadata.csv \
        --role amber \
        --base-model stable-diffusion-v1-5/stable-diffusion-v1-5 \
        --output-dir outputs/t2i_lora/amber_v1 \
        --rank 16 --resolution 512 --train-batch-size 2 \
        --num-train-epochs 10 --learning-rate 1e-4 \
        --smoke-generate --test-prompt "amber_(genshin_impact), solo, anime style, high quality"

训练后生成并交由识别系统校验（闭环）：
    # 生成图 -> 走现有 classify_image(v9) 链路打身份分（CPU 亦可）
"""
import argparse
import csv
from pathlib import Path


def build_examples(metadata_csv, role=None, limit=None):
    """读取 metadata.csv -> [(image_path, caption)]。纯函数，可独立测试。"""
    examples = []
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # role 列可能缺失（旧版两列 CSV）：用 .get() 防御，仅当列存在时过滤
            if role and row.get("role") not in (None, "", role):
                continue
            examples.append((row["image_path"], row["caption"]))
            if limit and len(examples) >= limit:
                break
    return examples


def _detect_device(override: str = "auto"):
    import torch
    # 显式指定设备（opt-in）：用户可用 --device mps 在 Mac 上尝试 MPS 训练提速，
    # 或 --device cuda 在 GPU 机器上训练。默认 auto 仍走下面的保守策略。
    if override in ("mps", "cuda", "cpu"):
        dtype = torch.float16 if override == "cuda" else torch.float32
        return override, dtype
    # Mac MPS 上 SD 训练不稳：fp16 训练触发 MPS 混合精度算子崩溃
    # ('mps.multiply' requires same element type)；fp32 训练 unet 整体搬 MPS 易 OOM 杀进程。
    # 统一退回 CPU 训练——底座冻结、只训 LoRA，内存安全（底座在 CPU 不占 MPS 统一内存），
    # 慢但必然跑完，足以完成「训 LoRA -> 生成 -> 识别校验」闭环 PoC。
    if torch.backends.mps.is_available():
        return "cpu", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _load_training_stack():
    """惰性导入重依赖，便于 --dry-run 在无 torch/diffusers 环境自检数据通路。"""
    import torch
    import diffusers
    from peft import LoraConfig, get_peft_model
    return torch, diffusers, LoraConfig, get_peft_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--role", help="只训练该角色（对应 caption 中的身份 token 目录名）")
    ap.add_argument("--base-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--output-dir", default="outputs/t2i_lora/role_v1")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--train-batch-size", type=int, default=2)
    ap.add_argument("--num-train-epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, help="调试：仅取前 N 条样本")
    ap.add_argument("--mixed-prec", action="store_true", help="MPS 上用 float16（更快但可能不稳定）")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="训练设备：auto=Mac 强制 CPU（防 MPS OOM，慢但必跑完）；"
                         "可选 mps/cuda 提速（MPS 训练可能不稳，仅提速验证时使用）")
    ap.add_argument("--smoke-generate", action="store_true", help="训练后加载 LoRA 生成 1 张图验证")
    ap.add_argument("--test-prompt", default=None)
    ap.add_argument("--dry-run", action="store_true", help="只校验数据通路与依赖，不训练")
    ap.add_argument("--generate-only", action="store_true",
                    help="跳过训练，仅加载已保存 LoRA 生成 smoke 图（配合 --smoke-generate 语义）")
    ap.add_argument("--inference-steps", type=int, default=50,
                    help="生成步数（越大越稳；默认 50，PoC 用 25 偏少易出多脸/抽象）")
    args = ap.parse_args()

    examples = build_examples(args.metadata, role=args.role, limit=args.limit)
    print(f"[data] 样本数: {len(examples)}"
          + (f" (role={args.role})" if args.role else ""))
    if examples:
        print(f"[data] 样例: {examples[0][1]}")

    if args.dry_run:
        try:
            _load_training_stack()
            print("[dry-run] torch/diffusers/peft 可用，依赖就绪。")
        except Exception as e:
            print(f"[dry-run] 重依赖未安装（{type(e).__name__}），跳过 pipeline 加载。")
            print("          训练前请: ./t2i-mac/bin/pip install -r scripts/t2i/requirements-t2i-mac.txt")
        print("[dry-run] 数据通路 OK，退出。")
        return

    torch, diffusers, LoraConfig, get_peft_model = _load_training_stack()
    from diffusers import StableDiffusionPipeline
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image

    device, default_dtype = _detect_device(args.device)
    # MPS 16GB OOM 治理：unet 用 float32 在 MPS 训练（最稳），但 vae/text_encoder 留在 CPU
    # （offload），避免 4.5GB 权重全常驻 MPS 触发 jetdam 杀进程（实测整体 to(device) 必 OOM）。
    weight_dtype = default_dtype
    if args.mixed_prec and device != "cpu":
        weight_dtype = torch.float16
    print(f"[device] {device}  weight_dtype={weight_dtype} (vae/text_encoder 留 CPU offload)")

    load_dtype = weight_dtype
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=load_dtype,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    tokenizer = pipe.tokenizer
    noise_scheduler = pipe.scheduler
    vae, unet, text_encoder = pipe.vae, pipe.unet, pipe.text_encoder
    # VAE / text_encoder 强制 float32（在 CPU 运行，规避 dtype 不匹配与 MPS NaN），
    # 只有 unet 用 fp16 留在 MPS 训练；latents/ctx 在送入 unet 前再转 fp16。
    vae = vae.to(torch.float32)
    text_encoder = text_encoder.to(torch.float32)
    # 仅把 unet 搬到 MPS（训练主体）；vae/text_encoder 留 CPU 按需本地计算
    unet = unet.to(device)

    # 注入 LoRA 到 UNet 交叉注意力层
    unet_lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        bias="none",
    )
    unet = get_peft_model(unet, unet_lora)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"[model] 可训练 LoRA 参数: {sum(p.numel() for p in params):,}")

    tf = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def collate(batch):
        imgs, caps = [], []
        for path, cap in batch:
            imgs.append(tf(Image.open(path).convert("RGB")))
            caps.append(cap)
        return torch.stack(imgs), caps

    loader = DataLoader(examples, batch_size=args.train_batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0)
    optim = torch.optim.AdamW(params, lr=args.learning_rate)
    vae_dtype = torch.float32  # VAE 全程 float32，规避 MPS NaN

    import resource

    def _mem_mb():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # macOS: bytes->MB
        mps = torch.mps.current_allocated_memory() / 1024 / 1024 if device == "mps" else 0.0
        return rss, mps

    for epoch in range(args.num_train_epochs):
        for i, (imgs, caps) in enumerate(loader):
            # imgs 在 CPU；vae/text_encoder 也留在 CPU 计算，仅把 latents/ctx 搬到 MPS
            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215
                if latents.dtype != weight_dtype:
                    latents = latents.to(weight_dtype)
                latents = latents.to(device)
                tokens = tokenizer(caps, padding="max_length", max_length=77,
                                   truncation=True, return_tensors="pt").input_ids
                ctx = text_encoder(tokens)[0].to(weight_dtype).to(device)
            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (latents.shape[0],), device=device).long()
            noisy = noise_scheduler.add_noise(latents, noise, t)
            pred = unet(noisy, t, encoder_hidden_states=ctx).sample
            loss = torch.nn.functional.mse_loss(pred, noise)
            optim.zero_grad(); loss.backward(); optim.step()
            if device == "mps":
                torch.mps.empty_cache()
            if i % 5 == 0:
                rss, mps = _mem_mb()
                print(f"[train] epoch {epoch} step {i} loss {loss.item():.4f} | RSS {rss:.0f}MB MPS {mps:.0f}MB")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # --generate-only: 跳过训练，直接加载已保存 LoRA 生成（重训是长任务，LoRA 已落盘时用这个）
    if args.generate_only:
        print(f"[generate] 加载已保存 LoRA: {out}")
        pipe.load_lora_weights(str(out))
        prompt = args.test_prompt or (examples[0][1] if examples else "solo, anime style, high quality")
        print(f"[generate] 生成: {prompt!r} (steps={args.inference_steps})")
        img = pipe(prompt, num_inference_steps=args.inference_steps, guidance_scale=7.5).images[0]
        smoke_path = out / "smoke_test.png"
        img.save(smoke_path)
        print(f"[generate] 已保存: {smoke_path}")
        return

    unet.save_attn_procs(out)  # 保存 pytorch_lora_weights.bin（SD1.5 格式）
    print(f"[done] LoRA 已保存: {out}")

    if args.smoke_generate:
        prompt = args.test_prompt or examples[0][1]
        print(f"[smoke] 把组件搬到 {device} 并生成: {prompt!r}")
        pipe = pipe.to(device)
        pipe.load_lora_weights(str(out))
        img = pipe(prompt, num_inference_steps=args.inference_steps, guidance_scale=7.5).images[0]
        smoke_path = out / "smoke_test.png"
        img.save(smoke_path)
        print(f"[smoke] 已保存: {smoke_path}")


if __name__ == "__main__":
    main()
