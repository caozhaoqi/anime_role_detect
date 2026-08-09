#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientNet-B3 角色分类模型训练脚本

基于项目 final_dataset 数据训练 EfficientNet-B3 分类模型，
产出格式与 model_loader.py 完全兼容：
  - models/efficientnet_b3/model_best.pth (state_dict checkpoint)
  - models/efficientnet_b3/class_to_idx.json
  - models/efficientnet_b3/training_results.json

模型架构与 create_model_from_name("efficientnet_b3", num_classes) 一致：
  backbone: EfficientNet-B3 (pretrained)
  classifier: Dropout(0.3) → Linear(1536→768) → ReLU → BN(768) → Dropout(0.15) → Linear(768→num_classes)
"""

import os
import sys
import json
import time
import copy
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy

# 防泄漏：以角色目录为 group 做分组切分（同角色图整体进 train 或 val，不拆散）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.data.split_utils import grouped_split, post_group_key  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─── 项目路径 ───
# 修正：原先写的是 parent.parent，指向 scripts/ 而非项目根，
# 导致 DATA_DIR=scripts/data/final_dataset（不存在）。__file__ 在
# scripts/model_training/ 下，必须回溯三级。
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "final_dataset"
# 默认输出目录：不要覆盖 v3 生产模型(models/efficientnet_b3) 或 v6。
MODEL_DIR = PROJECT_ROOT / "models" / "efficientnet_b3_v7"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / "seed42"

# ─── 默认训练配置 ───
DEFAULT_CONFIG = {
    # 256 = src/common/preprocess.IMAGE_SIZE 唯一真源。
    # 原默认 300 与统一口径冲突（Phase0 实测 300 相对 256 仅 +0.0019 MacroF1，
    # 属噪声级，却多约 37% 算力），已按实测结论改为 256。
    "image_size": 256,
    "batch_size": 16,
    "num_epochs": 40,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "patience": 8,           # 早停耐心
    "min_images_per_class": 30,
    "train_ratio": 0.85,
    "seed": 42,
    "use_auto_augment": True,
    "use_weighted_sampler": True,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,
}


def get_best_device():
    """自动检测最佳设备"""
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.backends.cuda.is_available():
        logger.info("Using CUDA GPU")
        return torch.device("cuda")
    else:
        logger.info("Using CPU only")
        return torch.device("cpu")


def create_efficientnet_b3(
    num_classes: int,
    weights=models.EfficientNet_B3_Weights.DEFAULT,
    pretrained_file: "str | None" = None,
) -> nn.Module:
    """创建与 model_loader.py 完全一致的 EfficientNet-B3 架构

    weights: 默认使用 ImageNet 预训练权重（v1 训练 / train_clean_split 需要）；
    增量微调与评测场景应传入 weights=None，跳过 47MB 权重下载——
    因为随即 strict 加载 checkpoint 会把整个主干覆盖掉，下载的权重永不使用。

    pretrained_file: 从**本地文件**加载 ImageNet 主干权重，绕开 torch.hub 下载。
        存在的理由：本机实测 torch.hub 下载 efficientnet_b3 会 hash 校验失败
        （期望 cf984f9c，实得 b3899882…），torchvision 直接抛 RuntimeError
        导致训练无法启动。取回的文件本身结构完好、经统计判定确为真实 ImageNet
        训练产物（BN running_var 离散、num_batches_tracked=1,971,260），
        只是容器字节与官方记录不一致。用本地文件显式加载，比关闭 hash 校验更可控。
    """
    if pretrained_file:
        model = models.efficientnet_b3(weights=None)
        state = torch.load(pretrained_file, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=True), None
        logger.info(f"已从本地文件加载 ImageNet 主干权重: {pretrained_file}")
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 768),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(768),
            nn.Dropout(p=0.15),
            nn.Linear(768, num_classes),
        )
        return model

    model = models.efficientnet_b3(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 768),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(768),
        nn.Dropout(p=0.15),
        nn.Linear(768, num_classes),
    )
    return model


class MixupCriterion:
    """Mixup 数据增强的损失函数"""
    def __init__(self, criterion, alpha=0.2):
        self.criterion = criterion
        self.alpha = alpha

    def __call__(self, model, inputs, targets):
        if self.alpha > 0 and torch.rand(1).item() > 0.5:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
            batch_size = inputs.size(0)
            index = torch.randperm(batch_size, device=inputs.device)
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
            outputs = model(mixed_inputs)
            loss = lam * self.criterion(outputs, targets) + \
                   (1 - lam) * self.criterion(outputs, targets[index])
            return loss, outputs, targets  # 返回原始 targets 用于计算准确率
        else:
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            return loss, outputs, targets


def get_transforms(image_size: int, use_auto_augment: bool = True):
    """创建训练和验证的数据变换 —— 委托给 src/common/preprocess 唯一真源。

    这里不再自己拼 transform，原因是历史上出现过两类不一致：
      1. val 用 Resize((256,256)) 直接压缩，而 train 是 Resize(288)->RandomCrop(256)，
         同一物体在 val 里比 train 里小 1.125 倍（尺度偏移，早停据此判断不可靠）。
      2. 服务端另写了一份 Resize((224,224))，与训练尺寸完全脱节。
    现在 train / val / 线上服务全部由同一个模块产出，改一处即全局生效。
    """
    from src.common.preprocess import build_eval_transform, build_train_transform

    train_transform = build_train_transform(
        image_size=image_size, use_auto_augment=use_auto_augment
    )
    # val 与 train 尺度对齐：Resize(image_size+32) -> CenterCrop(image_size)
    val_transform = build_eval_transform(image_size=image_size)
    return train_transform, val_transform


def load_frozen_split(split_dir: Path, data_dir: Path):
    """消费 make_split.py 产出的**冻结切分**，而不是在训练脚本里临时重切。

    为什么必须这样做
    ----------------
    此前 main() 是自己扫目录 + 内存里 grouped_split(post_id) 重切一遍，带来三个问题：
      1. 绕开了 #2 的内容哈希(sha256+union-find)去重 —— 同内容不同 post_id 的图
         会重新泄漏进 train/val。
      2. 切分不可复现、无法追溯：产物里没有 split_hash，事后无法判断某个模型
         有没有被污染（v6 就是这么变成一笔糊涂账的）。
      3. min_images_per_class 过滤会得到 151 类，与冻结切分的 171 类对不上。

    安全护栏：拒绝 test.json（评测集绝不进训练）、拒绝备份目录、
    校验文件确实存在、校验 label 空间连续无空洞。

    Returns:
        (train_samples, val_samples, class_to_idx, split_meta)
    """
    split_dir = Path(split_dir)
    name = split_dir.name.lower()
    if "backup" in name or "bak" in name:
        raise ValueError(
            f"拒绝使用备份切分目录 {split_dir} —— 备份仅供事后比对，不可用于训练"
        )

    train_path = split_dir / "train.json"
    val_path = split_dir / "val.json"
    summary_path = split_dir / "summary.json"
    for p in (train_path, val_path, summary_path):
        if not p.exists():
            raise FileNotFoundError(f"切分文件缺失: {p}")

    with open(train_path, encoding="utf-8") as f:
        train_rows = json.load(f)
    with open(val_path, encoding="utf-8") as f:
        val_rows = json.load(f)
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    # 标签空间以 train ∪ val 的并集为准，排序后编号，保证连续无空洞
    labels = sorted({r["label"] for r in train_rows} | {r["label"] for r in val_rows})
    class_to_idx = {c: i for i, c in enumerate(labels)}

    def to_samples(rows, tag):
        out = []
        missing = []
        for r in rows:
            p = Path(data_dir) / r["path"]
            if not p.exists():
                missing.append(r["path"])
                continue
            out.append((str(p), class_to_idx[r["label"]]))
        if missing:
            raise FileNotFoundError(
                f"{tag} 中有 {len(missing)} 个文件在 {data_dir} 下不存在，"
                f"切分与数据集不同步。示例: {missing[:3]}"
            )
        return out

    train_samples = to_samples(train_rows, "train.json")
    val_samples = to_samples(val_rows, "val.json")

    # train/val 路径不得相交（冻结切分理应已保证，这里再兜一道）
    overlap = {p for p, _ in train_samples} & {p for p, _ in val_samples}
    if overlap:
        raise AssertionError(
            f"FATAL: train 与 val 路径相交 {len(overlap)} 条，切分已损坏。"
            f" 示例: {list(overlap)[:3]}"
        )

    split_meta = {
        "split_dir": str(split_dir),
        "split_hash": summary.get("split_hash"),
        "split_schema_version": summary.get("schema_version"),
        "group_by": summary.get("group_by"),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "excluded_from_eval_characters": summary.get("excluded_from_eval_characters"),
    }
    return train_samples, val_samples, class_to_idx, split_meta


def filter_dataset_by_min_images(data_dir: str, min_images: int):
    """过滤出图片数量 >= min_images 的类别"""
    valid_classes = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        count = len([f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))])
        if count >= min_images:
            valid_classes.append(class_name)
    return valid_classes


class FilteredImageDataset(torch.utils.data.Dataset):
    """自定义 Dataset，直接从 (path, label) 样本列表构建。

    解码策略统一由 src/common/preprocess 提供（LOAD_TRUNCATED_IMAGES=True、
    放宽 MAX_IMAGE_PIXELS），本类不再自行设置——保持唯一真源。

    坏数据可见性
    ------------
    这里**曾经**是 `except Exception: continue` 静默吞异常：坏图被悄悄跳过，
    既不报错也不记日志。后果是数据集里 16 张截断 JPEG 计入切分总数、
    却从不贡献梯度，而且没有任何人能从日志里发现这件事。
    现在改为 warning 级日志（文件路径 + 异常类型 + 消息），每个文件每进程只报
    一次以免刷屏，并累计计数供训练结束后复盘。静默跳过一律不再保留。
    """

    def __init__(self, samples, transform, idx_to_class, image_size=256):
        self.samples = samples
        self.transform = transform
        self.idx_to_class = idx_to_class
        self._img_size = image_size
        # 坏样本可追溯：路径 -> "异常类型: 消息"；DataLoader 多 worker 时各进程独立
        self.decode_failures = {}

    def __len__(self):
        return len(self.samples)

    def _record_failure(self, path, exc):
        """记录一次解码失败——每个文件每进程只 warning 一次，避免逐 epoch 刷屏。"""
        key = str(path)
        if key in self.decode_failures:
            return
        detail = f"{type(exc).__name__}: {exc}"
        self.decode_failures[key] = detail
        logger.warning(
            "[decode-fail] 跳过无法解码的样本 path=%s reason=%s "
            "(该文件仍在切分内计数，但不贡献梯度)",
            key,
            detail,
        )

    def __getitem__(self, idx):
        # 注意：不在这里设 MAX_IMAGE_PIXELS / LOAD_TRUNCATED_IMAGES，
        # 二者已由 src.common.preprocess 在 import 时全局生效。
        from src.common.preprocess import load_image

        n = len(self.samples)
        for attempt in range(n):  # 最多遍历整个数据集寻找可解码样本
            path, label = self.samples[(idx + attempt) % n]
            try:
                img = load_image(path)
                if self.transform:
                    img = self.transform(img)
                return img, label
            except Exception as exc:  # noqa: BLE001 - 结构性损坏图跳过，但必须留痕
                self._record_failure(path, exc)
                continue
        # 极端兜底：整个数据集都无法解码时返回占位张量，保证训练循环不中断
        logger.error(
            "[decode-fail] 从 idx=%d 起遍历全部 %d 个样本均无法解码，返回占位张量！",
            idx,
            n,
        )
        return torch.zeros(3, self._img_size, self._img_size), self.samples[idx][1]


def create_weighted_sampler_from_samples(samples):
    """创建加权随机采样器，直接从 (path, label) 样本列表计算权重，不打开图片"""
    class_counts = {}
    for _, label in samples:
        class_counts[label] = class_counts.get(label, 0) + 1

    sample_weights = [1.0 / class_counts[label] for _, label in samples]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, phase="train", max_steps=None):
    """训练/验证一个 epoch"""
    if phase == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if max_steps is not None and batch_idx >= max_steps:
            break  # 仅用于计时/冒烟，正式训练不传该参数
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if phase == "train":
            optimizer.zero_grad(set_to_none=True)

            # Mixup
            loss, outputs, orig_labels = criterion(model, inputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss(label_smoothing=0.0)(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == orig_labels if phase == "train" else preds == labels).sum().item()
        total_samples += inputs.size(0)

        if batch_idx % 20 == 0 and phase == "train":
            logger.info(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = running_corrects / max(total_samples, 1)

    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, best_acc, class_to_idx, model_dir, filename="model_best.pth"):
    """保存 checkpoint（与 model_loader.py 兼容格式）"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "class_to_idx": class_to_idx,
    }
    path = model_dir / filename
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
    return path


def save_training_results(
    model_dir, class_to_idx, train_config, metrics, best_acc, training_time,
    split_meta=None,
):
    """保存训练结果（与 model_loader.py 兼容格式）

    split_meta 必须写入
    -------------------
    v6 的教训：产物里没有任何切分指纹，事后完全无法判断它是否被污染
    （只能靠 "test Top-1 0.7419 反超自报 val 0.6114" 这种间接反常来推断）。
    从 v7 起，split_hash / group_by / 预处理规格一律固化进 training_results.json，
    后续任何评测都能一眼确认模型与评测集是否同源。
    """
    from src.common.preprocess import describe as describe_preprocess

    results = {
        "model_name": "efficientnet_b3",
        "architecture": "EfficientNet-B3",
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "class_names": sorted(class_to_idx.keys()),
        "image_size": train_config.get("image_size", 256),
        "best_accuracy": float(best_acc),
        "training_time_seconds": training_time,
        "training_config": train_config,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        # ── 可追溯性字段（v6 缺失，v7 起强制）──
        "split": split_meta or {},
        "split_hash": (split_meta or {}).get("split_hash"),
        "preprocess": describe_preprocess(train_config.get("image_size", 256)),
    }

    results_path = model_dir / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved training results to {results_path}")

    # 同时保存 class_to_idx.json
    class_map_path = model_dir / "class_to_idx.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved class mapping to {class_map_path}")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B3 anime character classifier")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--min-images", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--no-mixup", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--no-auto-augment", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--device", type=str, default=None, choices=["mps", "cpu", "cuda"])
    parser.add_argument(
        "--split-dir", type=str, default=None,
        help=f"冻结切分目录（含 train/val/summary.json），默认 {DEFAULT_SPLIT_DIR}",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help=f"输出目录，默认 {MODEL_DIR}（勿指向 v3/v6 以免覆盖）",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="仅用于计时/冒烟：每个 epoch 最多跑 N 个 batch 后停止",
    )
    parser.add_argument(
        "--pretrained-weights", type=str, default=None,
        help="本地 ImageNet 主干权重文件；用于绕开 torch.hub 下载 hash 校验失败",
    )
    args = parser.parse_args()

    # 合并配置
    config = DEFAULT_CONFIG.copy()
    if args.epochs: config["num_epochs"] = args.epochs
    if args.batch_size: config["batch_size"] = args.batch_size
    if args.lr: config["learning_rate"] = args.lr
    if args.image_size: config["image_size"] = args.image_size
    if args.min_images: config["min_images_per_class"] = args.min_images
    if args.patience: config["patience"] = args.patience
    if args.no_mixup: config["mixup_alpha"] = 0.0
    if args.no_weighted_sampler: config["use_weighted_sampler"] = False
    if args.no_auto_augment: config["use_auto_augment"] = False

    # 设置随机种子
    torch.manual_seed(config["seed"])
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config["seed"])

    device = torch.device(args.device) if args.device else get_best_device()
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # ─── 创建模型目录 ───
    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {model_dir}")

    # ─── 消费冻结切分（不再在此临时重切）───
    # 切分由 scripts/model_evaluation/make_split.py 产出并冻结，含 #2 的
    # sha256+union-find 内容去重。训练脚本只读不切，保证可复现、可追溯。
    train_samples, val_samples, new_class_to_idx, split_meta = load_frozen_split(
        Path(args.split_dir) if args.split_dir else DEFAULT_SPLIT_DIR,
        DATA_DIR,
    )
    num_classes = len(new_class_to_idx)
    new_idx_to_class = {idx: cls for cls, idx in new_class_to_idx.items()}

    logger.info(
        "切分: %s | split_hash=%s | group_by=%s",
        split_meta["split_dir"], split_meta["split_hash"], split_meta["group_by"],
    )
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Train: {len(train_samples)} images, Val: {len(val_samples)} images")

    # ─── 数据变换 ───
    train_transform, val_transform = get_transforms(config["image_size"], config["use_auto_augment"])
    sys.stdout.flush()

    print("[DEBUG] Creating datasets...", flush=True)
    train_dataset = FilteredImageDataset(
        train_samples, train_transform, new_idx_to_class, image_size=config["image_size"]
    )
    val_dataset = FilteredImageDataset(
        val_samples, val_transform, new_idx_to_class, image_size=config["image_size"]
    )
    print("[DEBUG] Datasets created", flush=True)

    # macOS 上统一用 num_workers=0 避免多进程 pickle 问题
    _num_workers = 0
    _pin_memory = device.type == "cuda"

    print("[DEBUG] Creating DataLoader...", flush=True)
    if config["use_weighted_sampler"]:
        sampler = create_weighted_sampler_from_samples(train_samples)
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"],
            sampler=sampler, num_workers=_num_workers, pin_memory=_pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"],
            shuffle=True, num_workers=_num_workers, pin_memory=_pin_memory,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=_num_workers, pin_memory=_pin_memory,
    )
    print("[DEBUG] DataLoader created", flush=True)

    # ─── 创建模型 ───
    print("[DEBUG] Creating EfficientNet-B3 model...", flush=True)
    model = create_efficientnet_b3(
        num_classes, pretrained_file=args.pretrained_weights
    )
    print("[DEBUG] Model created, moving to device...", flush=True)
    model = model.to(device)
    print("[DEBUG] Model on device", flush=True)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: EfficientNet-B3, Total params: {total_params}, Trainable: {trainable_params}")
    logger.info(f"Classes: {num_classes} ({new_class_to_idx})")

    # ─── 损失函数和优化器 ───
    base_criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    mixup_criterion = MixupCriterion(base_criterion, alpha=config["mixup_alpha"])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # 学习率调度器：Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    # ─── 断点续训 ───
    start_epoch = 0
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    metrics_history = []

    resume_path = model_dir / "model_best.pth"
    if args.resume and resume_path.exists():
        logger.info(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        best_model_wts = copy.deepcopy(model.state_dict())
        # 推进 scheduler 到正确位置
        for _ in range(start_epoch):
            scheduler.step()
        logger.info(f"Resumed at epoch {start_epoch}, best_acc={best_acc:.4%}")
        # 加载已有 metrics
        results_path = model_dir / "training_results.json"
        if results_path.exists():
            with open(results_path) as f:
                old_results = json.load(f)
            metrics_history = old_results.get("metrics", [])
    else:
        logger.info("Starting fresh training (no checkpoint to resume from)")

    logger.info("=" * 60)
    logger.info(f"Training: EfficientNet-B3, {num_classes} classes, {device}, start_epoch={start_epoch}")
    logger.info("=" * 60)

    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + config["num_epochs"]):
        logger.info(f"\nEpoch {epoch+1}/{start_epoch + config['num_epochs']}")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, mixup_criterion, optimizer, device, phase="train",
            max_steps=args.max_steps,
        )

        # 验证
        val_loss, val_acc = train_one_epoch(
            model, val_loader, base_criterion, optimizer, device, phase="val",
            max_steps=args.max_steps,
        )

        # 更新学习率
        scheduler.step()

        # 记录指标
        current_lr = optimizer.param_groups[0]["lr"]
        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        logger.info(
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4%} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4%} | LR: {current_lr:.6f}"
        )

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"  *** Best model saved! Val Acc: {best_acc:.4%} ***")

            # 保存 checkpoint
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                new_class_to_idx, model_dir, "model_best.pth"
            )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                logger.info(f"  Early stopping: no improvement for {config['patience']} epochs")
                break

    # ─── 加载最佳模型并保存最终结果 ───
    model.load_state_dict(best_model_wts)
    training_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"Training completed in {training_time:.0f}s ({training_time/60:.1f}min)")
    logger.info(f"Best validation accuracy: {best_acc:.4%}")
    logger.info("=" * 60)

    # 保存训练结果
    save_training_results(
        model_dir, new_class_to_idx, config, metrics_history, best_acc, training_time,
        split_meta=split_meta,
    )

    # 同时保存 model_full.pth（完整模型权重，model_loader 优先加载）
    checkpoint_final = {
        "epoch": len(metrics_history),
        "model_state_dict": model.state_dict(),
        "best_acc": best_acc,
        "class_to_idx": new_class_to_idx,
    }
    torch.save(checkpoint_final, model_dir / "model_full.pth")
    logger.info(f"Saved model_full.pth to {model_dir}")

    logger.info(f"\nAll training artifacts saved to {model_dir}/")
    logger.info(f"  - model_best.pth")
    logger.info(f"  - model_full.pth")
    logger.info(f"  - class_to_idx.json")
    logger.info(f"  - training_results.json")

    return best_acc


if __name__ == "__main__":
    best_acc = main()
    logger.info(f"Final best accuracy: {best_acc:.4%}")
