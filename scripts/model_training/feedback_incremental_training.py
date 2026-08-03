#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反馈驱动的增量训练消费脚本（Phase 2 缓存收尾）。

职责：
  1. 从 logs/feedback/feedback_*.jsonl 采集用户纠错反馈，去重 + 过滤 + 已消费排除；
  2. 从 data/final_dataset 抽 replay 子集做防遗忘；
  3. 在 v1（models/efficientnet_b3）权重上微调（默认只训分类头，冻结主干）；
  4. 评测闸门：v4 的 Macro-F1 不得比 v1 低 > 1.0 个点，否则判回归（退出码 3，不消费）；
  5. 产物落独立目录 models/efficientnet_b3_v4_<ts>/，绝不碰生产 models/efficientnet_b3；
  6. 仅当训练成功 + 产物已落盘 + 评测闸门通过时，原子更新 data/feedback_images/.consumed_manifest.json。

关键约束（详见任务说明）：
  - 顶层只 import stdlib + json + os + PIL（--dry-run 路径不得触发 torch import）。
  - 模型架构必须用 train_efficientnet_b3.create_efficientnet_b3（自定义 6 层头），
    且 strict=True 加载 v1 权重（pre-flight 断言）。
  - image_size 固定 224，与服务侧 classifiers.py 的 Resize((224,224)) 一致。
  - 类别数恒为 51，禁止引入 final_dataset 中其余 ~106 个无关类。
"""

import os
import sys
import json
import random
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── 项目路径 ──
# 本文件位于 <repo>/scripts/model_training/，repo 根需向上 3 级
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger

    logger = get_logger("feedback_incremental_training")
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("feedback_incremental_training")

# ── 路径常量 ──
DATASET_DIR = os.path.join(project_root, "data", "final_dataset")
V1_MODEL_DIR = os.path.join(project_root, "models", "efficientnet_b3")
FEEDBACK_LOG_DIR = os.path.join(project_root, "logs", "feedback")
DEFAULT_CONSUMED_MANIFEST = os.path.join(
    project_root, "data", "feedback_images", ".consumed_manifest.json"
)
TEST_MANIFEST = os.path.join(project_root, "data", "splits", "seed42", "test.json")

NUM_CLASSES = 51
PARENT_MODEL = "models/efficientnet_b3"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 退出码
EXIT_OK = 0
EXIT_INSUFFICIENT = 2
EXIT_REGRESS = 3
EXIT_ERROR = 1


# ======================================================================
# 反馈样本加载（纯逻辑，无 torch 依赖）
# ======================================================================
def load_feedback_samples(jsonl_dir, class_to_idx, consumed_manifest, min_samples=10):
    """读取 feedback_*.jsonl，按 recognition_id 去重，过滤后返回可用样本。

    返回：list[dict]，每个元素：
        {
          "recognition_id": str,
          "img_path": str,        # project_root 下绝对路径
          "label_idx": int,
          "corrected_label": str,
          "image_ref": str,       # 相对 project_root 的路径（用于台账）
          "source_jsonl": str,    # 来源 jsonl 文件名
        }

    去重规则：同一 recognition_id 取最后出现的一行（端点以 rid 命名图片，重提会覆盖图）。
    过滤规则：
        ① image_ref 指向文件真实存在；
        ② corrected_label ∈ 51 类；
        ③ recognition_id 不在 consumed_manifest（避免重复消费）。
    """
    jsonl_dir = jsonl_dir or FEEDBACK_LOG_DIR
    latest = {}  # recognition_id -> record（保留最后一行）
    file_order = sorted(
        f for f in os.listdir(jsonl_dir) if f.startswith("feedback_") and f.endswith(".jsonl")
    ) if os.path.isdir(jsonl_dir) else []

    for fname in file_order:
        fpath = os.path.join(jsonl_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过损坏行 {fname}: {e}")
                        continue
                    rid = payload.get("recognition_id")
                    if not rid:
                        continue
                    latest[rid] = (payload, fname)
        except OSError as e:
            logger.warning(f"无法读取 {fpath}: {e}")

    samples = []
    consumed_set = set(consumed_manifest.keys()) if consumed_manifest else set()
    for rid, (payload, fname) in latest.items():
        if rid in consumed_set:
            continue
        image_ref = payload.get("image_ref")
        if not image_ref:
            continue
        abs_path = os.path.join(project_root, image_ref)
        if not os.path.isfile(abs_path):
            continue
        corrected_label = payload.get("corrected_label")
        if corrected_label not in class_to_idx:
            continue
        samples.append(
            {
                "recognition_id": rid,
                "img_path": abs_path,
                "label_idx": class_to_idx[corrected_label],
                "corrected_label": corrected_label,
                "image_ref": image_ref,
                "source_jsonl": fname,
            }
        )

    logger.info(f"反馈样本加载完成: 去重后 {len(latest)} 条, 通过过滤 {len(samples)} 条 (min_samples={min_samples})")
    return samples


# ======================================================================
# Replay 数据集构建（纯逻辑，无 torch 依赖）
# ======================================================================
def build_replay_dataset(final_dataset_dir, class_to_idx, replay_per_class=30, seed=42):
    """遍历 51 类目录，每类随机抽 <= replay_per_class 张，返回 [(img_path, label_idx), ...]。

    只遍历 class_to_idx 中的 51 类；final_dataset 中其余无关类绝不引入（避免输出层维度错乱）。
    """
    rng = random.Random(seed)
    samples = []
    for class_name, idx in class_to_idx.items():
        class_dir = os.path.join(final_dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"replay 跳过缺失类目录: {class_name}")
            continue
        imgs = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(IMG_EXTS)
        ]
        if not imgs:
            continue
        if len(imgs) <= replay_per_class:
            chosen = imgs
        else:
            chosen = rng.sample(imgs, replay_per_class)
        for p in chosen:
            samples.append((p, idx))
    logger.info(f"replay 数据集构建完成: {len(samples)} 张 (覆盖 {len(class_to_idx)} 类)")
    return samples


def load_consumed_manifest(manifest_path):
    """读取已消费台账。返回 dict: recognition_id -> info。"""
    if not manifest_path or not os.path.isfile(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"读取 consumed_manifest 失败，按空处理: {e}")
    return {}


def mark_consumed(manifest_path, used_records, model_version):
    """原子更新已消费台账：read → modify → 写临时文件 → os.rename。

    used_records: list[dict]，含 recognition_id, source_jsonl, corrected_label, image_ref。
    manifest 是缓存清理的唯一事实源；仅在“训练成功且产物已落盘”后调用。
    """
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    existing = load_consumed_manifest(manifest_path)
    now_iso = datetime.now().isoformat()
    for rec in used_records:
        rid = rec.get("recognition_id")
        if not rid:
            continue
        existing[rid] = {
            "consumed_at": now_iso,
            "consumed_by": model_version,
            "corrected_label": rec.get("corrected_label"),
            "source_jsonl": rec.get("source_jsonl"),
            "image_ref": rec.get("image_ref"),
        }
    tmp_path = manifest_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, manifest_path)
    logger.info(f"已原子更新 consumed_manifest: {len(used_records)} 条 -> {manifest_path}")


# ======================================================================
# 懒加载 torch 相关原语（保证 --dry-run 不触发 torch import）
# ======================================================================
_PRIMITIVES = None


def _get_primitives():
    """懒加载 torch / numpy 及训练原语（EMA / WarmupCosine / mixup / cutmix 等）。

    仅在真正训练 / 评测时调用，返回缓存的 dict。
    """
    global _PRIMITIVES
    if _PRIMITIVES is not None:
        return _PRIMITIVES

    import torch
    import numpy as np
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms, models

    # ── EMA 模型 ──
    class EMA:
        def __init__(self, model, decay=0.999):
            self.model = model
            self.decay = decay
            self.shadow = {}
            self.backup = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()

        def update(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

        def apply_shadow(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    self.backup[name] = param.data.clone()
                    param.data = self.shadow[name]

        def restore(self):
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.backup
                    param.data = self.backup[name]
            self.backup = {}

    # ── 学习率预热 + 余弦退火 ──
    class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, T_max, warmup_epochs=5, eta_min=1e-7, last_epoch=-1):
            self.T_max = T_max
            self.warmup_epochs = warmup_epochs
            self.eta_min = eta_min
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch < self.warmup_epochs:
                return [
                    base_lr * (self.last_epoch + 1) / self.warmup_epochs
                    for base_lr in self.base_lrs
                ]
            cos_decay = 0.5 * (
                1
                + np.cos(
                    np.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.T_max - self.warmup_epochs)
                )
            )
            return [
                self.eta_min + (base_lr - self.eta_min) * cos_decay
                for base_lr in self.base_lrs
            ]

    # ── MixUp ──
    def mixup_data(x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # ── CutMix ──
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def cutmix_data(x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

    # ── 检查点保存 / 加载 ──
    def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, train_history, filepath, ema=None):
        checkpoint = {
            "epoch": epoch,
            "best_acc": best_acc,
            "train_history": train_history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "ema_shadow": ema.shadow if ema else None,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存: {filepath}")

    def load_checkpoint(filepath, model, optimizer, scheduler=None, ema=None):
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if ema and checkpoint.get("ema_shadow"):
            ema.shadow = checkpoint["ema_shadow"]
        return checkpoint["epoch"], checkpoint["best_acc"], checkpoint["train_history"]

    # ── 单 epoch 训练 ──
    def train_one_epoch(
        model, dataloader, criterion, optimizer, device, epoch, total_epochs,
        use_mixup=True, use_cutmix=True, gradient_clip_norm=0.0,
    ):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            augment_choice = np.random.random()
            if use_mixup and augment_choice < 0.25:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.8)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            elif use_cutmix and augment_choice < 0.5:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        epoch_loss = running_loss / max(1, len(dataloader))
        epoch_acc = 100.0 * correct / max(1, total)
        return epoch_loss, epoch_acc

    def validate(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        epoch_loss = running_loss / max(1, len(dataloader))
        epoch_acc = 100.0 * correct / max(1, total)
        return epoch_loss, epoch_acc

    _PRIMITIVES = dict(
        torch=torch, np=np, nn=nn, optim=optim,
        DataLoader=DataLoader, Dataset=Dataset, transforms=transforms, models=models,
        EMA=EMA, WarmupCosineAnnealingLR=WarmupCosineAnnealingLR,
        mixup_data=mixup_data, mixup_criterion=mixup_criterion,
        cutmix_data=cutmix_data, rand_bbox=rand_bbox,
        save_checkpoint=save_checkpoint, load_checkpoint=load_checkpoint,
        train_one_epoch=train_one_epoch, validate=validate,
    )
    return _PRIMITIVES


# ======================================================================
# 模型构建 + 冻结（torch 依赖，懒加载）
# ======================================================================
def build_model(num_classes):
    """create_efficientnet_b3(num_classes) + 加载 v1 权重（strict=True 预检）。"""
    assert num_classes == NUM_CLASSES, f"类别数必须 == {NUM_CLASSES}, 收到 {num_classes}"
    import torch
    from train_efficientnet_b3 import create_efficientnet_b3

    model = create_efficientnet_b3(num_classes, weights=None)
    ckpt_path = os.path.join(V1_MODEL_DIR, "model_best.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"v1 权重不存在: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # pre-flight 断言：strict=True 必须通过；键不匹配时 loud fail 并打印缺失/多余键
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.error(f"strict 加载 v1 权重失败! missing={missing} unexpected={unexpected}")
        raise
    logger.info(f"v1 权重 strict 加载成功 (num_classes={num_classes})")
    return model


def apply_freeze(model, freeze_backbone):
    """默认冻结主干，仅 classifier 可训练。"""
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"主干冻结，仅 classifier 可训练 (trainable params={trainable})")
    else:
        for p in model.parameters():
            p.requires_grad = True
        logger.info("全模型可训练")
    return model


# ======================================================================
# 数据装配（torch 依赖，懒加载）
# ======================================================================
def assemble_training_samples(feedback_samples, replay_samples, merge_ratio, seed=42):
    """按 merge_ratio 控制反馈在 batch 中的目标占比；反馈极少时对反馈上采样。

    返回 (feedback_list, replay_list)：
        feedback_list: list[(img_path, label_idx)]，已按目标占比上采样
        replay_list:   list[(img_path, label_idx)]
    各自使用不同增强（反馈轻量 / replay heavy），由调用方分别建 Dataset 后 Concat。
    """
    rng = random.Random(seed)
    F = len(feedback_samples)
    R = len(replay_samples)
    fb = [(s["img_path"], s["label_idx"]) for s in feedback_samples]
    if R == 0:
        logger.warning("replay 为空，仅使用反馈样本训练（无防遗忘保护）")
        return fb, []
    if merge_ratio >= 1.0:
        merge_ratio = 0.99
    target_feedback = int(round(R * merge_ratio / (1.0 - merge_ratio)))
    if F < target_feedback and F > 0:
        fb = fb + rng.choices(fb, k=target_feedback - F)
        logger.info(f"反馈上采样: {F} -> {len(fb)} (目标占比 {merge_ratio:.2f})")
    elif F == 0:
        logger.warning("无反馈样本可上采样")
    logger.info(
        f"训练样本装配完成: replay={R}, feedback={len(fb)} (反馈占比≈{len(fb) / max(1, len(fb) + R):.3f})"
    )
    return fb, replay_samples


def make_dataset(samples, transform):
    """从 (img_path, label_idx) 列表构建 Dataset（依赖 PIL，已在顶层 import）。"""
    from PIL import Image
    import torch.utils.data as tud

    class _DS(tud.Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            try:
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception as e:
                logger.error(f"加载失败 {img_path}: {e}")
                image = Image.new("RGB", (224, 224), (128, 128, 128))
                if self.transform:
                    image = self.transform(image)
                return image, label

    return _DS(samples, transform)


# ======================================================================
# 评测闸门（torch 依赖，懒加载）
# ======================================================================
def _compute_macro_f1(model_dir, test_manifest, device, image_size=224):
    """在 test.json 上评测模型的 Macro-F1。复用 eval_on_manifest 的口径。"""
    import torch
    from PIL import Image
    from train_efficientnet_b3 import create_efficientnet_b3

    model_dir = Path(model_dir)
    c2i = json.load(open(model_dir / "class_to_idx.json", encoding="utf-8"))
    i2c = {v: k for k, v in c2i.items()}
    num_classes = len(c2i)
    manifest = json.load(open(test_manifest, encoding="utf-8"))

    model = create_efficientnet_b3(num_classes, weights=None)
    ckpt = torch.load(model_dir / "model_best.pth", map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total = 0
    skipped = 0
    DATA_DIR = Path(DATASET_DIR)
    with torch.no_grad():
        for item in manifest:
            path = DATA_DIR / item["path"]
            label = item["label"]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                skipped += 1
                continue
            x = transform(img).unsqueeze(0).to(device)
            out = model(x)
            pred = int(torch.argmax(out, 1).item())
            true_c = i2c[label]
            pred_c = i2c[pred]
            total += 1
            if pred_c == true_c:
                tp[true_c] += 1
            else:
                fp[pred_c] += 1
                fn[true_c] += 1

    def safe_div(a, b):
        return a / b if b else 0.0

    f1s = []
    for c in c2i:
        p = safe_div(tp[c], tp[c] + fp[c])
        r = safe_div(tp[c], tp[c] + fn[c])
        f1 = safe_div(2 * p * r, p + r) if (p + r) else 0.0
        f1s.append(f1)
    macro_f1 = safe_div(sum(f1s), len(f1s))
    return {"macro_f1": macro_f1, "total": total, "skipped": skipped}


def evaluate_vs_baseline(v4_dir, v1_dir, test_manifest, image_size=224, device=None):
    """评测闸门：v4 的 Macro-F1 不得比 v1 低超过 1.0 个点。

    返回 dict，含 macro_f1_v4 / macro_f1_v1 / delta / passed / gate_margin。
    """
    import torch

    if device is None:
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    r_v1 = _compute_macro_f1(v1_dir, test_manifest, device, image_size)
    r_v4 = _compute_macro_f1(v4_dir, test_manifest, device, image_size)
    delta = r_v4["macro_f1"] - r_v1["macro_f1"]
    # passed = v4 不比 v1 低超过 1.0 个点
    passed = delta >= -1.0
    result = {
        "v1_dir": str(v1_dir),
        "v4_dir": str(v4_dir),
        "macro_f1_v1": round(r_v1["macro_f1"], 4),
        "macro_f1_v4": round(r_v4["macro_f1"], 4),
        "delta": round(delta, 4),
        "gate_margin": 1.0,
        "passed": passed,
    }
    logger.info(
        f"评测闸门: v1 MacroF1={result['macro_f1_v1']:.4f} "
        f"v4 MacroF1={result['macro_f1_v4']:.4f} delta={delta:.4f} passed={passed}"
    )
    return result


# ======================================================================
# 训练主流程（torch 依赖，懒加载）
# ======================================================================
def run_training(args, class_to_idx):
    env = _get_primitives()
    torch = env["torch"]
    np = env["np"]
    nn = env["nn"]
    optim = env["optim"]
    DataLoader = env["DataLoader"]
    EMA = env["EMA"]
    WarmupCosineAnnealingLR = env["WarmupCosineAnnealingLR"]
    save_checkpoint = env["save_checkpoint"]
    load_checkpoint = env["load_checkpoint"]
    train_one_epoch = env["train_one_epoch"]
    validate = env["validate"]

    device = _resolve_device(args.device)
    logger.info(f"使用设备: {device}")

    # 1) 加载已消费台账 + 反馈样本
    consumed = load_consumed_manifest(args.consumed_manifest)
    feedback_samples = load_feedback_samples(
        FEEDBACK_LOG_DIR, class_to_idx, consumed, args.min_samples
    )
    if len(feedback_samples) < args.min_samples:
        logger.error(
            f"可用反馈样本 {len(feedback_samples)} < min_samples {args.min_samples}，退出码 {EXIT_INSUFFICIENT}"
        )
        return EXIT_INSUFFICIENT

    # 2) replay 子集
    replay_samples = build_replay_dataset(
        DATASET_DIR, class_to_idx, replay_per_class=args.replay_per_class, seed=42
    )

    # 3) 数据装配 + 增强
    from train_efficientnet_b3 import get_transforms

    heavy_train, val_transform = get_transforms(args.image_size, use_auto_augment=True)
    light_train = env["transforms"].Compose([
        env["transforms"].RandomHorizontalFlip(p=0.5),
        env["transforms"].ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        env["transforms"].Resize((args.image_size, args.image_size)),
        env["transforms"].ToTensor(),
        env["transforms"].Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    fb_list, replay_list = assemble_training_samples(
        feedback_samples, replay_samples, args.merge_ratio, seed=42
    )
    # 反馈轻量增强；replay 用 get_transforms 的 heavy 增强；分别建 Dataset 后拼接
    feedback_ds = make_dataset(fb_list, light_train)
    replay_ds = make_dataset(replay_list, heavy_train)
    train_ds = torch.utils.data.ConcatDataset([feedback_ds, replay_ds])

    # 拆分 train/val：取 10% replay 作为验证（反馈样本全用于训练）
    n_val = max(1, int(0.1 * len(replay_list)))
    val_samples = replay_list[:n_val]
    val_ds = make_dataset(val_samples, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    # 4) 模型 + 冻结
    model = build_model(NUM_CLASSES)
    apply_freeze(model, args.freeze_backbone)
    model.to(device)

    # 5) 优化器 / 调度 / 损失
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = WarmupCosineAnnealingLR(
        optimizer, T_max=args.epochs, warmup_epochs=5, eta_min=1e-7
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    ema = EMA(model, decay=0.999) if args.ema else None

    best_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 8
    delta_thr = 0.001
    train_history = []
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        start_epoch, best_acc, train_history = load_checkpoint(
            args.resume, model, optimizer, scheduler, ema=ema
        )
        scheduler.last_epoch = start_epoch
        start_epoch += 1
        logger.info(f"从 epoch {start_epoch} 恢复训练")

    gradient_clip = 1.0
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs,
            use_mixup=args.mixup, use_cutmix=args.cutmix, gradient_clip_norm=gradient_clip,
        )
        if ema:
            ema.update()
        if ema:
            ema.apply_shadow()
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            ema.restore()
        else:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f}% val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )
        train_history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        if val_acc > best_acc + delta_thr:
            best_acc = val_acc
            if ema:
                ema.apply_shadow()
                best_model_state = model.state_dict().copy()
                ema.restore()
            else:
                best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停：验证准确率连续 {patience} 轮未提升")
                break

    if best_model_state is None:
        best_model_state = model.state_dict().copy()
    model.load_state_dict(best_model_state)
    logger.info(f"训练完成，最佳验证准确率: {best_acc:.2f}%")
    return {
        "model": model,
        "best_acc": best_acc,
        "train_history": train_history,
        "feedback_samples": feedback_samples,
        "device": device,
    }


# ======================================================================
# 产物落盘 + 消费标记
# ======================================================================
def save_artifacts(model, args, class_to_idx, feedback_samples, eval_result, train_meta):
    """写 model_best.pth (dict) / class_to_idx.json / training_results.json / consumed_feedback.json。"""
    import torch

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # dict checkpoint（约 48MB，不存 143MB 的 model_full）
    ckpt = {
        "epoch": train_meta.get("epoch", len(train_meta.get("train_history", []))),
        "model_state_dict": model.state_dict(),
        "best_acc": train_meta.get("best_acc", 0.0),
        "class_to_idx": class_to_idx,
    }
    torch.save(ckpt, os.path.join(out_dir, "model_best.pth"))

    # 复制 v1 的 51 类映射
    shutil.copy(
        os.path.join(V1_MODEL_DIR, "class_to_idx.json"),
        os.path.join(out_dir, "class_to_idx.json"),
    )

    training_results = {
        "model_name": "efficientnet_b3_v4",
        "num_classes": NUM_CLASSES,
        "image_size": args.image_size,
        "parent_model": PARENT_MODEL,
        "freeze_backbone": args.freeze_backbone,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "merge_ratio": args.merge_ratio,
        "replay_per_class": args.replay_per_class,
        "use_ema": args.ema,
        "use_mixup": args.mixup,
        "use_cutmix": args.cutmix,
        "consumed_feedback_count": len(feedback_samples),
        "best_val_acc": train_meta.get("best_acc", 0.0),
        "train_history": train_meta.get("train_history", []),
        "eval_vs_baseline": eval_result,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(out_dir, "training_results.json"), "w", encoding="utf-8") as f:
        json.dump(training_results, f, ensure_ascii=False, indent=2)

    consumed_feedback = [
        {
            "recognition_id": s["recognition_id"],
            "source_jsonl": s["source_jsonl"],
            "corrected_label": s["corrected_label"],
            "image_ref": s["image_ref"],
        }
        for s in feedback_samples
    ]
    with open(os.path.join(out_dir, "consumed_feedback.json"), "w", encoding="utf-8") as f:
        json.dump(consumed_feedback, f, ensure_ascii=False, indent=2)

    logger.info(f"产物已落盘: {out_dir}")
    return out_dir


# ======================================================================
# 设备解析
# ======================================================================
def _resolve_device(device_arg):
    import torch

    if device_arg:
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.backends.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ======================================================================
# main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="反馈驱动的增量训练消费脚本（v1 -> v4）"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="只统计 + 打印计划，不训练（不触发 torch import）")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="可用反馈样本下限，不足则退出码 2")
    parser.add_argument("--merge-ratio", type=float, default=0.1,
                        help="反馈在 batch 中的目标占比")
    parser.add_argument("--replay-per-class", type=int, default=30,
                        help="replay 每类抽样数")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率（head-only）")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--device", type=str, default=None,
                        help="设备: mps/cpu/cuda，默认 mps->cpu 回退")
    parser.add_argument("--image-size", type=int, default=224, help="输入尺寸（须 224）")
    parser.add_argument("--no-ema", dest="ema", action="store_false", default=True,
                        help="关闭 EMA")
    parser.add_argument("--no-mixup", dest="mixup", action="store_false", default=True,
                        help="关闭 MixUp")
    parser.add_argument("--no-cutmix", dest="cutmix", action="store_false", default=True,
                        help="关闭 CutMix")
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true",
                        default=True, help="冻结主干（默认开）")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                        help="不冻结主干")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="产物目录，默认 models/efficientnet_b3_v4_<ts>")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复")
    parser.add_argument("--consumed-manifest", type=str, default=DEFAULT_CONSUMED_MANIFEST,
                        help="已消费台账路径")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅对 --output-dir 指向的 v4 跑评测闸门（不训练）")
    parser.add_argument("--v1-dir", type=str, default=V1_MODEL_DIR, help="v1 模型目录")
    args = parser.parse_args()

    # 类别映射（51 类）
    c2i_path = os.path.join(V1_MODEL_DIR, "class_to_idx.json")
    if not os.path.isfile(c2i_path):
        logger.error(f"缺少 51 类映射: {c2i_path}")
        return EXIT_ERROR
    class_to_idx = json.load(open(c2i_path, encoding="utf-8"))
    assert len(class_to_idx) == NUM_CLASSES, f"class_to_idx 必须 {NUM_CLASSES} 类"

    # ── dry-run：纯计数 + 计划，不触发 torch ──
    if args.dry_run:
        consumed = load_consumed_manifest(args.consumed_manifest)
        feedback_samples = load_feedback_samples(
            FEEDBACK_LOG_DIR, class_to_idx, consumed, args.min_samples
        )
        replay_samples = build_replay_dataset(
            DATASET_DIR, class_to_idx, replay_per_class=args.replay_per_class, seed=42
        )
        print("=" * 60)
        print("[DRY-RUN] 反馈驱动增量训练计划")
        print("=" * 60)
        print(f"可用反馈样本 (通过过滤): {len(feedback_samples)}")
        print(f"min_samples 阈值        : {args.min_samples}")
        print(f"replay 样本总数         : {len(replay_samples)} (覆盖 {len(class_to_idx)} 类)")
        print(f"merge_ratio             : {args.merge_ratio}")
        print(f"image_size              : {args.image_size}")
        print(f"freeze_backbone         : {args.freeze_backbone}")
        print(f"epochs/lr/batch         : {args.epochs}/{args.lr}/{args.batch_size}")
        if len(feedback_samples) < args.min_samples:
            print(f"!! 样本不足，将退出码 {EXIT_INSUFFICIENT}")
            return EXIT_INSUFFICIENT
        print("样本充足，计划可执行训练（未实际训练）。")
        return EXIT_OK

    # ── eval-only：仅评测闸门 ──
    if args.eval_only:
        if not args.output_dir or not os.path.isdir(args.output_dir):
            logger.error("--eval-only 需要 --output-dir 指向已产出的 v4 目录")
            return EXIT_ERROR
        result = evaluate_vs_baseline(
            args.output_dir, args.v1_dir, TEST_MANIFEST, image_size=args.image_size
        )
        if not result["passed"]:
            logger.error(f"评测闸门未过 (delta={result['delta']:.4f})，退出码 {EXIT_REGRESS}")
            return EXIT_REGRESS
        return EXIT_OK

    # ── 完整训练流程 ──
    try:
        args.output_dir = args.output_dir or os.path.join(
            project_root, "models",
            f"efficientnet_b3_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        train_meta = run_training(args, class_to_idx)
        if isinstance(train_meta, int):
            return train_meta  # 样本不足等退出码
        model = train_meta["model"]
        feedback_samples = train_meta["feedback_samples"]

        # 评测闸门（在产物落盘前计算，但需先有模型权重；这里先落盘再评测，失败不消费）
        out_dir = save_artifacts(
            model, args, class_to_idx, feedback_samples, None, train_meta
        )
        eval_result = evaluate_vs_baseline(
            out_dir, args.v1_dir, TEST_MANIFEST, image_size=args.image_size
        )
        # 写回 eval 结果到 training_results.json
        tr_path = os.path.join(out_dir, "training_results.json")
        tr = json.load(open(tr_path, encoding="utf-8"))
        tr["eval_vs_baseline"] = eval_result
        json.dump(tr, open(tr_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        if not eval_result["passed"]:
            logger.error(
                f"评测闸门未过 (v4={eval_result['macro_f1_v4']:.4f} vs v1={eval_result['macro_f1_v1']:.4f}, delta={eval_result['delta']:.4f})，不消费反馈，退出码 {EXIT_REGRESS}"
            )
            return EXIT_REGRESS

        # 训练成功 + 产物落盘 + 闸门通过 → 原子更新消费台账
        mark_consumed(args.consumed_manifest, feedback_samples, os.path.basename(out_dir))
        logger.info(f"全部完成，产物: {out_dir}")
        return EXIT_OK
    except Exception as e:
        logger.error(f"训练流程异常: {e}", exc_info=True)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
