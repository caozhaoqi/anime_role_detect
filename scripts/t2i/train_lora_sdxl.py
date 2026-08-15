"""Phase 0/C - SDXL + LoRA 最小可跑训练样例。

基于 diffusers 的标准 SDXL LoRA 微调：只对 UNet（可选 + text_encoder_2）注入 LoRA，
冻结底座，训练图像→caption 对（由 scripts/t2i/prepare_captions.py 产出 metadata.csv）。

运行（GPU 机器）：
    pip install -r scripts/t2i/requirements-t2i.txt
    .venv/bin/python scripts/t2i/train_lora_sdxl.py \
        --metadata outputs/t2i_phase0/metadata.csv \
        --base-model stabilityai/stable-diffusion-xl-base-1.0 \
        --output-dir outputs/t2i_lora/amber_v1 \
        --role amber --rank 16 --resolution 1024 \
        --train-batch-size 4 --num-train-epochs 10 --learning-rate 1e-4

数据通路自检（无需 GPU/权重，本机即可跑）：
    .venv/bin/python scripts/t2i/train_lora_sdxl.py \
        --metadata outputs/t2i_phase0/metadata.csv --role amber --dry-run
"""
import argparse
import csv
import sys
from pathlib import Path


def build_examples(metadata_csv, role=None, limit=None):
    """读取 metadata.csv -> [(image_path, caption)]。纯函数，可独立测试。"""
    examples = []
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if role and row["role"] != role:
                continue
            examples.append((row["image_path"], row["caption"]))
            if limit and len(examples) >= limit:
                break
    return examples


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
    ap.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--output-dir", default="outputs/t2i_lora/role_v1")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--train-batch-size", type=int, default=4)
    ap.add_argument("--num-train-epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-text-encoder", action="store_true", help="同时对 text_encoder_2 注入 LoRA")
    ap.add_argument("--limit", type=int, help="调试：仅取前 N 条样本")
    ap.add_argument("--dry-run", action="store_true", help="只校验数据通路与依赖，不训练")
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
            print("          训练前请: pip install -r scripts/t2i/requirements-t2i.txt")
        print("[dry-run] 数据通路 OK，退出。")
        return

    torch, diffusers, LoraConfig, get_peft_model = _load_training_stack()
    from diffusers import StableDiffusionXLPipeline
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    ).to("cuda")
    tokenizer = pipe.tokenizer
    noise_scheduler = pipe.scheduler
    vae, unet, text_encoder, text_encoder_2 = pipe.vae, pipe.unet, pipe.text_encoder, pipe.text_encoder_2

    # 注入 LoRA
    unet_lora = LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    unet = get_peft_model(unet, unet_lora)
    if args.lora_text_encoder:
        te_lora = LoraConfig(r=args.rank, lora_alpha=args.rank, target_modules=["q_proj", "v_proj"])
        text_encoder_2 = get_peft_model(text_encoder_2, te_lora)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    params = [p for p in unet.parameters() if p.requires_grad]
    if args.lora_text_encoder:
        params += [p for p in text_encoder_2.parameters() if p.requires_grad]
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

    loader = DataLoader(examples, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate)
    optim = torch.optim.AdamW(params, lr=args.learning_rate)
    weight_dtype = torch.float16

    for epoch in range(args.num_train_epochs):
        for i, (imgs, caps) in enumerate(loader):
            imgs = imgs.to("cuda", dtype=weight_dtype)
            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample() * 0.18215
                tokens = tokenizer(caps, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to("cuda")
                enc = text_encoder(tokens)[0]
                enc2 = text_encoder_2(tokens)[0]
                ctx = torch.cat([enc, enc2], dim=-1)
            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device="cuda").long()
            noisy = noise_scheduler.add_noise(latents, noise, t)
            pred = unet(noisy, t, encoder_hidden_states=ctx).sample
            loss = torch.nn.functional.mse_loss(pred, noise)
            optim.zero_grad(); loss.backward(); optim.step()
            if i % 10 == 0:
                print(f"[train] epoch {epoch} step {i} loss {loss.item():.4f}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    unet.save_attn_procs(out)
    if args.lora_text_encoder:
        text_encoder_2.save_pretrained(out / "text_encoder_2")
    print(f"[done] LoRA 已保存: {out}")


if __name__ == "__main__":
    main()
