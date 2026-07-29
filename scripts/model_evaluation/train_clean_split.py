#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按无交叠切分【干净重训】EfficientNet-B3 —— 拿到真实泛化基线（修复数据泄漏）。

与 train_efficientnet_b3.py 的关系：
  - 复用其架构 / 训练配方（AutoAugment + Mixup + label smoothing + 余弦重启 + 加权采样）
  - 关键区别：只吃 data/splits/seed42/train.json 训练，val.json 早停选模型，
    test.json 全程不参与；训练产物写到 models/efficientnet_b3_v2/，**不覆盖现模型**。

用法：
  python scripts/model_evaluation/train_clean_split.py            # 从头训
  python scripts/model_evaluation/train_clean_split.py --resume  # 从 v2 断点续训
训练结束后自动调用 eval_on_manifest.py 在 test.json 上评测，输出真实 Top-1/MacroF1。
"""
import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, WeightedRandomSampler

# 复用原训练脚本的成熟组件
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training"))
from train_efficientnet_b3 import (
    create_efficientnet_b3,
    get_transforms,
    FilteredImageDataset,
    create_weighted_sampler_from_samples,
    train_one_epoch,
    save_checkpoint,
    save_training_results,
    MixupCriterion,
    get_best_device,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("train_clean_split")


class FocalLoss(nn.Module):
    """Focal Loss：降低易分样本权重、聚焦难分（弱类/混淆）样本。

    alpha 为每类权重张量（反频率归一化）时，兼具类别均衡作用。
    """

    def __init__(self, gamma: float = 2.0, alpha=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()

ROOT = Path(__file__).resolve().parent.parent.parent
SPLIT_DIR = ROOT / "data" / "splits" / "seed42"
MODEL_DIR = ROOT / "models" / "efficientnet_b3"          # 只读：架构/class_to_idx
OUT_DIR = ROOT / "models" / "efficientnet_b3_v2"          # 写：新模型，保护现模型
DATA_DIR = ROOT / "data" / "final_dataset"

CONFIG = {
    "image_size": 256,
    "batch_size": 24,
    "num_epochs": 45,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "patience": 8,
    "seed": 42,
    "use_auto_augment": True,
    "use_weighted_sampler": True,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,
}


def build_clean_model(num_classes: int, device: torch.device):
    """构建训练模型：优先 ImageNet 冷启动（无泄漏），失败则域适应热启动。

    热启动策略：backbone 从生产模型加载（域适应特征），分类头重建为随机初始化，
    避免分类头直接“记住” test 标签；仍需在 train-only 上重训、在 test 上评测。
    """
    init_mode = "imagenet"
    try:
        logger.info("尝试加载 ImageNet 预训练 EfficientNet-B3（冷启动）...")
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        logger.info("ImageNet 预训练权重加载成功")
    except Exception as e:
        logger.warning(f"ImageNet 权重不可用({e})，回退到生产模型热启动（域适应初始化）")
        init_mode = "warmstart"
        model = models.efficientnet_b3(weights=None)
        # 先按本项目架构重建分类头，避免默认 1000 类头部 shape 不匹配导致 load_state_dict 失败
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 768),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(768),
            nn.Dropout(p=0.15),
            nn.Linear(768, num_classes),
        )
        ckpt = torch.load(MODEL_DIR / "model_best.pth", map_location="cpu", weights_only=False)
        w = ckpt.get("model_state_dict") or ckpt
        model.load_state_dict(w, strict=False)
    # 统一重建分类头（num_classes 一致也重建，保证结构确定）
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 768),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(768),
        nn.Dropout(p=0.15),
        nn.Linear(768, num_classes),
    )
    logger.info(f"初始化方式: {init_mode}")
    return model.to(device), init_mode


def load_manifest(name):
    return json.load(open(SPLIT_DIR / name, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None, choices=["mps", "cpu", "cuda"])
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--no-auto-augment", action="store_true")
    ap.add_argument("--focal", action="store_true", help="使用 Focal Loss + 类别均衡 alpha（聚焦弱类/混淆样本）")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 models/efficientnet_b3_v2）")
    args = ap.parse_args()

    config = CONFIG.copy()
    if args.epochs:
        config["num_epochs"] = args.epochs
    config["image_size"] = args.image_size
    if args.no_auto_augment:
        config["use_auto_augment"] = False
    nw = args.num_workers

    torch.manual_seed(config["seed"])
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config["seed"])

    device = torch.device(args.device) if args.device else get_best_device()
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    c2i = json.load(open(MODEL_DIR / "class_to_idx.json", encoding="utf-8"))
    i2c = {v: k for k, v in c2i.items()}
    num_classes = len(c2i)

    # 把 manifest 的相对路径还原为绝对路径
    def to_abs(samples):
        return [(str(DATA_DIR / s["path"]), s["label"]) for s in samples]

    train_samples = to_abs(load_manifest("train.json"))
    val_samples = to_abs(load_manifest("val.json"))
    test_samples = to_abs(load_manifest("test.json"))
    logger.info(f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} (test 不参与训练)")

    train_tf, val_tf = get_transforms(config["image_size"], config["use_auto_augment"])
    train_ds = FilteredImageDataset(train_samples, train_tf, i2c)
    val_ds = FilteredImageDataset(val_samples, val_tf, i2c)

    nw = args.num_workers
    pin = device.type == "cuda"
    if config["use_weighted_sampler"]:
        sampler = create_weighted_sampler_from_samples(train_samples)
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler,
                                  num_workers=nw, pin_memory=pin)
    else:
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                                  num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                            num_workers=nw, pin_memory=pin)

    model, init_mode = build_clean_model(num_classes, device)
    config["init_mode"] = init_mode

    if args.focal:
        # 反频率归一化 alpha：弱类获得更高权重
        from collections import Counter
        cnt = Counter(lbl for _, lbl in train_samples)
        freq = torch.tensor([cnt.get(i, 1) for i in range(num_classes)], dtype=torch.float32)
        inv = 1.0 / (freq + 1e-6)
        alpha = inv / inv.sum() * num_classes
        alpha = alpha.to(device)
        base = FocalLoss(gamma=2.0, alpha=alpha, label_smoothing=0.05)
        logger.info("使用 Focal Loss (gamma=2, 类别均衡 alpha) —— 聚焦弱类/混淆样本")
    else:
        base = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    mixup = MixupCriterion(base, alpha=config["mixup_alpha"])
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    start_epoch = 0
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    patience_ctr = 0
    metrics = []

    ckpt_path = out_dir / "model_best.pth"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        best_wts = copy.deepcopy(model.state_dict())
        for _ in range(start_epoch):
            scheduler.step()
        logger.info(f"Resumed v2 at epoch {start_epoch}, best_val={best_acc:.4%}")
    else:
        logger.info("Fresh clean training (v2)")

    t0 = time.time()
    for epoch in range(start_epoch, start_epoch + config["num_epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, mixup, optimizer, device, "train")
        vl_loss, vl_acc = train_one_epoch(model, val_loader, base, optimizer, device, "val")
        scheduler.step()
        metrics.append({"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": vl_loss, "val_acc": vl_acc, "lr": optimizer.param_groups[0]["lr"]})
        logger.info(f"Epoch {epoch+1}: train {tr_acc:.4%} | val {vl_acc:.4%} | lr {optimizer.param_groups[0]['lr']:.6f}")
        if vl_acc > best_acc:
            best_acc = vl_acc
            best_wts = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            save_checkpoint(model, optimizer, epoch, best_acc, c2i, out_dir, "model_best.pth")
            logger.info(f"  *** best val {best_acc:.4%} ***")
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
                logger.info(f"Early stop @ epoch {epoch+1}")
                break

    model.load_state_dict(best_wts)
    dur = time.time() - t0
    logger.info(f"Clean training done in {dur/60:.1f}min, best_val={best_acc:.4%}")
    save_training_results(out_dir, c2i, config, metrics, best_acc, dur)

    # 复制 class_to_idx 到 v3（eval 需要）
    with open(out_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(c2i, f, ensure_ascii=False, indent=2)

    # 自动在 test.json 上评测（真实泛化值）
    logger.info("Evaluating on untouched test split...")
    r = subprocess.run([
        sys.executable, "-u", str(Path(__file__).resolve().parent / "eval_on_manifest.py"),
        "--manifest", str(SPLIT_DIR / "test.json"),
        "--model-dir", str(out_dir),
    ], check=False)
    logger.info(f"eval exit={r.returncode}")


if __name__ == "__main__":
    main()
