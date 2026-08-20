#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t2i_service 路径与运行配置。

所有路径相对项目根目录，可直接用环境变量覆盖。
"""
from __future__ import annotations

from pathlib import Path

# src/services/t2i_service/config.py -> parents[3] = 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = PROJECT_ROOT / "data" / "final_dataset"
MODELS_CACHE = PROJECT_ROOT / "models_cache"
SD15_DIR = MODELS_CACHE / "stable-diffusion-v1-5"
IP_DIR = MODELS_CACHE / "ip-adapter"
IP_MODELS_DIR = IP_DIR / "models"          # ip-adapter-plus_sd15.bin + image_encoder
T2I_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "t2i"
LORA_DIR = PROJECT_ROOT / "outputs" / "t2i_lora"
SCRIPTS_T2I = PROJECT_ROOT / "scripts" / "t2i"

# 训练子进程使用的 venv（必须含 diffusers/torch/peft，即 t2i-mac）
T2I_VENV_PYTHON = PROJECT_ROOT / "t2i-mac" / "bin" / "python"

# 服务监听
SERVICE_HOST = "0.0.0.0"
SERVICE_PORT = 8100

IP_WEIGHT_NAME = "ip-adapter-plus_sd15.bin"

# 推理默认参数
DEFAULT_SCALE = 0.6
# 默认步数：配合 DPM++(多步) 调度器（见 generator._ensure_base），同质量下 30→20 即提速 ~1.5×。
# 若换回默认 PNDM 调度器需调回 30 以维持画质；DPM++ 下 20 步已足够。
DEFAULT_STEPS = 20
DEFAULT_CFG = 7.5
DEFAULT_NUM_REF = 6
DEFAULT_NUM = 1

# 实验性提速：用 torch.compile 编译 UNet 计算图。
# 与 fp16 不同——它不牺牲 IP-Adapter 角色一致性（仅重排/融合算子），是安全的提速手段。
# 风险点：MPS 上 inductor 后端支持有限，可能 graph break 或回退 eager（无提速但不崩）。
# 首次推理会触发编译开销（首图更慢），后续复用缓存。可在 config 置 False 关闭。
COMPILE_UNET = True

# 实验性提速：多图批生成（num_images_per_prompt）的安全批大小上限。
# 把 num 张按此上限切成若干批，每批一次 pipeline 调用并行出多张，摊薄 UNet 开销提速。
# 风险：批内同时持有 batch 倍 latent，峰值统一内存随之上升。本机 SD1.5+IP-Adapter 基线已逼近
# 89.8% 内存，批生成曾直接把机器拖进 swap 卡死。故**默认=1（即逐张，无批生成）**，
# 仅在内存充裕的机器上才调大。若实跑 OOM/卡死，保持 1 即可。
T2I_MAX_BATCH = 1
