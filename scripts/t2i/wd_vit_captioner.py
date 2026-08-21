#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真 WD ViT v3 打标模块（S1 方案 A 落地，2026-08-20）。

背景
----
项目的 WDViTV3Tagger 类是"假打标器"：默认不加载模型、get_tags_api 返回硬编码
模拟标签、load_model 加载的其实是 CLIP 而非真 WD ViT v3（详见 wd_vit_v3_tagger_torch.py）。
本模块绕开该类，直接用 timm 加载 SmilingWolf/wd-vit-tagger-v3 真实权重：
  - config: vit_base_patch16_224, img_size=448, num_classes=10861,
            class_token=False, global_pool=avg, fc_norm=False, act_layer=gelu_tanh
  - 归一化 mean/std = 0.5/0.5/0.5（非 ImageNet 标准！用错会显著降质）
  - 后处理：sigmoid -> 按 category 分组 -> general 阈值 0.35 / character 0.85
  - caption 只取 general 标签（character 标签通常是角色名本身，且可能引入
    同图其他角色，对 LoRA 训练有害；rating 标签也不进 caption）

用法（作为模块被 enrich_captions.py 调用）：
    from wd_vit_captioner import WDV3Captioner
    cap = WDV3Captioner()          # 自动定位本地权重，探活失败抛 FileNotFoundError
    text = cap.caption("Bailu", "/path/img.jpg")
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.services.t2i_service import config  # noqa: E402


class _GELUTanh(nn.GELU):
    """匹配 SmilingWolf wd-vit-tagger-v3 的 gelu_tanh 激活。

    timm 的 act_layer 期望是一个"无参可实例化"的类（内部会 act_layer()），
    不能直接传 nn.GELU(approximate="tanh") 实例（那样会被二次调用而缺 input）。
    """

    def __init__(self, **kwargs):
        super().__init__(approximate="tanh")

# 权重缓存目录（HF 缓存布局：hub/models--<org>--<name>/snapshots/<hash>/）
_HF_HUB = ROOT / "huggingface_cache" / "hub"
WD_REPO_DIR = _HF_HUB / "models--SmilingWolf--wd-vit-tagger-v3"
MODEL_REPO_ID = "SmilingWolf/wd-vit-tagger-v3"

# 后处理阈值（SmilingWolf 官方推荐）
GENERAL_THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.85
MAX_GENERAL_TAGS = 12
IMG_SIZE = 448


def find_local_model_dir() -> Path | None:
    """定位本地权重目录；权重缺失（悬空软链/LFS 未拉）时返回 None。

    兼容两种落盘布局：
      - HF 缓存标准布局: .../snapshots/<hash>/model.safetensors（软链→blobs/）
      - 部分下载器 local_dir 直接平铺: .../model.safetensors
    二者都可能残留"悬空软链"（LFS 未拉/旧 blob 失效），必须用 exists()
    解析软链判断，并校验文件 >1MB 才是真实权重。
    """
    if not WD_REPO_DIR.exists():
        return None
    # 优先标准 snapshots 布局，其次平铺目录（兼容 download_wd_vit_modelscope.py 的 local_dir 落盘）
    candidates = sorted(WD_REPO_DIR.glob("snapshots/*"))
    candidates.append(WD_REPO_DIR)
    for d in candidates:
        model_file = d / "model.safetensors"
        tags_file = d / "selected_tags.csv"
        # exists() 会解析软链：悬空软链返回 False；真实文件返回 True
        if (model_file.exists() and tags_file.exists()
                and model_file.stat().st_size > 1_000_000):
            return d
    return None


def load_tag_table(csv_path: Path) -> list[dict]:
    """selected_tags.csv -> [{name, category}]（行序 = 模型输出索引序）。"""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"name": r["name"], "category": int(r["category"])})
    return rows


class WDV3Captioner:
    """真 WD ViT v3 打标器：timm 加载 + 448 预处理 + sigmoid 分组阈值。"""

    def __init__(self, model_dir: Path | None = None, device: str | None = None):
        self.model_dir = model_dir or find_local_model_dir()
        if self.model_dir is None:
            raise FileNotFoundError(
                f"未找到真实 WD ViT v3 权重（{WD_REPO_DIR}）。"
                f"请先运行: .venv/bin/python scripts/t2i/download_wd_vit_modelscope.py"
            )
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._load_model()
        self.tags = load_tag_table(self.model_dir / "selected_tags.csv")
        print(f"[wdvit] 真 WD ViT v3 已加载: {self.model_dir.name} (device={self.device}, "
              f"tags={len(self.tags)})")

    def _load_model(self):
        import timm
        m = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=10861,
            img_size=IMG_SIZE,
            class_token=False,
            global_pool="avg",
            fc_norm=False,
            act_layer=_GELUTanh,
        )
        # .safetensors 不是 pickle 格式，必须用 safetensors 加载（torch.load 会崩）
        from safetensors.torch import load_file
        state = load_file(str(self.model_dir / "model.safetensors"), device="cpu")
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing:
            print(f"[wdvit] 缺失 key {len(missing)} 个（前5: {missing[:5]}）")
        if unexpected:
            print(f"[wdvit] 多余 key {len(unexpected)} 个（前5: {unexpected[:5]}）")
        self.model = m.to(self.device).eval()

    @torch.no_grad()
    def predict_tags(self, image_path: str, general_thr: float = GENERAL_THRESHOLD) -> list[str]:
        """返回按概率降序的 general 标签（不含 character/rating）。"""
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5  # mean/std = 0.5
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        logits = self.model(x).float().cpu()[0]
        probs = torch.sigmoid(logits)

        gen = [(i, float(probs[i])) for i, t in enumerate(self.tags)
               if t["category"] == 0 and probs[i] >= general_thr]
        gen.sort(key=lambda kv: -kv[1])
        return [self.tags[i]["name"] for i, _ in gen[:MAX_GENERAL_TAGS]]

    def caption(self, role: str, image_path: str) -> str | None:
        """生成描述性 caption；失败返回 None（调用方回退模板）。"""
        try:
            tags = self.predict_tags(image_path)
            if not tags:
                return None
            from enrich_captions import build_caption  # 复用组装/过滤逻辑
            return build_caption(role, tags)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] 打标失败 {Path(image_path).name}: {e}")
            return None
