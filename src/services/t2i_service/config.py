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
DEFAULT_STEPS = 30
DEFAULT_CFG = 7.5
DEFAULT_NUM_REF = 12
DEFAULT_NUM = 1
