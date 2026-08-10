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

import sys
import json
import math
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
# v8：最佳模型按 Macro-F1 选型（头条指标），需要 sklearn.metrics.f1_score。
# scikit-learn==1.6.1 已在 requirements.txt / requirements-base.txt 中声明。
from sklearn.metrics import f1_score

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
    # v8：45 与调度器 CosineAnnealingLR(T_max=num_epochs) 对齐，单周期退火到底。
    "num_epochs": 45,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    # v8：patience 8→15。v7 周期 2 从 restart 恢复用了 6ep，而周期 3 只拿到 7ep
    # 预算就被砍（ep37，退火仅完成 15%）；且 val n=1585，单点二项 SE=1.22 点，
    # patience=8 极易被噪声误杀。详见 outputs/v7_audit/v8_training_recommendations.md
    "patience": 15,          # 早停耐心
    # v8：早停/最佳模型判据的最小改善量，避免浮点平局被当成"无改善"计入 patience。
    "min_delta": 1e-4,
    # 注意：这里**不再有** min_images_per_class。它曾是死配置——训练消费的是
    # load_frozen_split 的冻结切分，该阈值从未生效，却被完整写进
    # training_results.json，让任何读产物的人都以为"每类至少 30 张"（实际
    # train 里 59 个类 <30 张，最小 1 张）。连同 --min-images CLI 一起移除。
    "train_ratio": 0.85,
    "seed": 42,
    # 真正生效的减类开关（与已移除的死配置 min_images_per_class 不同）：
    # 它作用在 load_frozen_split 之后、num_classes 之前，直接裁剪样本列表并
    # 连续重编号，因此 num_classes / class_names / dataset 全部自动跟随。
    "min_samples_per_class": 0,  # 0=不减类；>0=丢弃 train+val 合计样本<该值的类
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
      3. min_images_per_class 过滤会得到 151 类，与冻结切分的类数对不上
         （该死配置已随本次清理一并移除，见 DEFAULT_CONFIG 注释）。

    安全护栏：拒绝 test.json（评测集绝不进训练）、拒绝备份目录、
    校验文件确实存在、校验 label 空间连续无空洞。

    Returns:
        (train_samples, val_samples, class_to_idx, idx_to_name, split_meta)

    ``class_to_idx`` 的 key 是切分 JSON 里的**整数 label**（不是类名），因此单靠
    它无法还原真实角色名 —— 这正是 training_results.json 的 class_names 一度写成
    ``[0,1,...,170]`` 的原因。故这里额外返回 ``idx_to_name``（连续索引 -> 角色名，
    取自样本 path 的父目录），供产物写盘时输出真实类名。
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

    # 连续索引 -> 真实角色名（path 的父目录即类名，与切分文件自洽，不依赖外部映射）
    idx_to_name = {}
    for r in train_rows + val_rows:
        idx_to_name[class_to_idx[r["label"]]] = r["path"].split("/", 1)[0]

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
    return train_samples, val_samples, class_to_idx, idx_to_name, split_meta


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

    # v8：平方根采样取代全逆频。全逆频(1/count)让 k=1 的单张图每 epoch 被重复
    # 采样约 43 次，纯粹制造过拟合；平方根采样保留部分重平衡又不极端。
    # 依据：v7 test balanced_acc 0.5681 vs top1 0.5989 仅差 3.08 点，说明头部偏置
    # 已被基本抵消，无需继续用最激进的逆频权重。
    # 详见 outputs/v7_audit/v8_training_recommendations.md 建议表 #9
    sample_weights = [1.0 / math.sqrt(class_counts[label]) for _, label in samples]

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


def evaluate_val_metrics(model, val_loader, device, num_classes, criterion, max_steps=None):
    """在 val 集上完整评估，返回 {loss, acc(Top-1), macro_f1, n_present_classes, n_samples}。

    为什么不复用 train_one_epoch(phase="val")
    -----------------------------------------
    v7 用 Top-1(`val_acc`) 选 best checkpoint，但对外头条指标是 **Macro-F1**
    （171 类长尾，Top-1 被大类掩盖）—— **选型目标与考核目标不一致**，
    v7 的 best 只保证 Top-1 最优。train_one_epoch 只做流式累加，拿不到
    全量 preds/labels，算不出 Macro-F1，所以这里单独收集全量预测再统一计算。

    macro_f1 口径 = **present-class（val 中 support>=1 的类）**
    ------------------------------------------------------
    对应 v7 复盘中的 `macro_f1_present_only` 167 类口径。若按全 171 类算，
    jean / rina / seth / yae_miko 这 4 个 TRAIN_ONLY 单图类（全库各仅 1 张图
    且全在 train，`summary.json` 已声明 eval_status=TRAIN_ONLY）会以 F1=0
    计入分母，等价于施加一个固定 167/171 = 2.34% 的会计折扣，没有任何信息量。
    详见 outputs/v7_audit/v8_training_recommendations.md §2。

    注意：`criterion` 由调用方显式传入。v7 的 val loss 实际用的是
    `nn.CrossEntropyLoss(label_smoothing=0.0)`（train_one_epoch L382 硬编码），
    并非传入的 base_criterion(label_smoothing=0.1)。为保持 val_loss 与 v7 可比，
    调用方应传入 label_smoothing=0.0 的 criterion。
    """
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            if max_steps is not None and batch_idx >= max_steps:
                break  # 仅用于计时/冒烟，正式训练不传该参数
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total_samples += inputs.size(0)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if total_samples == 0:
        logger.warning("[val] 验证集为空（或 max_steps=0），返回全零指标")
        return {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0,
                "n_present_classes": 0, "n_samples": 0}

    y_true = torch.cat(all_labels).tolist()
    y_pred = torch.cat(all_preds).tolist()
    # present_classes：本次 val 中 support>=1 的类；显式传给 f1_score 的 labels=，
    # 保证分母只含可评类，不受 num_classes 里的 TRAIN_ONLY 类影响。
    present_classes = sorted(set(y_true))
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro",
                 labels=present_classes, zero_division=0)
    )

    return {
        "loss": running_loss / total_samples,
        "acc": running_corrects / total_samples,
        "macro_f1": macro_f1,
        # 保留 present/total 对照，便于事后判断口径（v7 = 166/171 on val）
        "n_present_classes": len(present_classes),
        "n_total_classes": num_classes,
        "n_samples": total_samples,
    }


def save_checkpoint(model, optimizer, epoch, best_acc, class_to_idx, model_dir,
                    filename="model_best.pth", best_macro_f1=None):
    """保存 checkpoint（与 model_loader.py 兼容格式）

    `best_macro_f1` 为 v8 新增的**尾部关键字参数**：v8 起最佳模型按 Macro-F1 选型，
    resume 时必须能恢复该值，否则续训会用 0.0 起跳、第一个 epoch 必然"改善"并覆盖
    best。放在参数表末尾且带默认值，是为了不破坏既有 7 位置参调用
    （scripts/model_evaluation/train_clean_split.py:287）。
    `best_acc` 字段保持不变且仍为 float —— outputs/v7_audit/eval_v7_audit.py:129
    直接对它做 `:.6f` 格式化，改类型会炸。
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "class_to_idx": class_to_idx,
    }
    if best_macro_f1 is not None:
        checkpoint["best_macro_f1"] = float(best_macro_f1)
    path = model_dir / filename
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
    return path


def compute_val_coverage_gap(split_dir, class_to_idx, val_samples, data_dir):
    """计算验证集覆盖缺口 —— **仅写入元数据，绝不把 test 喂给训练**。

    时序问题（不是偏好问题）：某些角色在 val.json 中完全没有样本，于是它们
    永远无法参与早停 / 最佳模型选择 / val 指标。其中部分在 test.json 中仍有
    样本（至少还能评测），部分是 train/test 双盲（完全不可评）。

    本函数显式列出这些"无 val 类"及其覆盖状态，固化进 training_results.json，
    避免事后无法判断"为什么某些类的指标永远缺失"。

    val_samples / test.json 只读标签空间，不进训练循环；与 load_frozen_split
    一样遵守护栏：test.json 仅作元数据消费，绝不泄漏进训练。

    注意：class_to_idx 的 key 是切分 JSON 里的 int label（不是类名），
    真实类名取自 path 的父目录，需重新解析切片文件建立 idx -> 类名映射。
    """
    split_dir = Path(split_dir)
    idx_to_label = {i: lbl for lbl, i in class_to_idx.items()}

    # 建立 idx -> 类名（path 父目录）。train/val/test 都扫，因为个别类
    # （如 theresa_apocalypse）只在 test 出现，需要从 test 补名字。
    idx_to_name = {}
    for fname in ("train.json", "val.json", "test.json"):
        fp = split_dir / fname
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as f:
            rows = json.load(f)
        for r in rows:
            # 减类（min_samples_per_class>0）后 class_to_idx 只含保留类，
            # 被丢弃的 label 在这里必须跳过，否则 KeyError 直接崩在训练前。
            if r["label"] not in class_to_idx:
                continue
            idx_to_name[class_to_idx[r["label"]]] = Path(r["path"]).parent.name

    val_labels = {idx for _, idx in val_samples}

    test_path = split_dir / "test.json"
    test_idx = set()
    if test_path.exists():
        with open(test_path, encoding="utf-8") as f:
            test_rows = json.load(f)
        test_idx = {
            class_to_idx[r["label"]] for r in test_rows if r["label"] in class_to_idx
        }

    missing_val = []
    for idx in sorted(class_to_idx.values()):
        has_val = idx in val_labels
        has_test = idx in test_idx
        if not has_val:
            missing_val.append({
                "class_index": idx,
                "class_name": idx_to_name.get(idx, f"<label {idx_to_label.get(idx)}>"),
                "has_val": False,
                "has_test": has_test,
                "note": (
                    "无 val 也无 test，完全不可评"
                    if not has_test
                    else "有 test 但无 val，可评测但无法参与早停/最佳模型选择"
                ),
            })

    return {
        "val_class_coverage": f"{len(val_labels)}/{len(class_to_idx)}",
        "missing_val_classes": missing_val,
    }


def save_training_results(
    model_dir, class_to_idx, train_config, metrics, best_acc, training_time,
    split_meta=None, val_coverage_gap=None, best_macro_f1=None, class_names=None,
):
    """保存训练结果（与 model_loader.py 兼容格式）

    best_macro_f1（v8 新增，尾部关键字参数）
    ---------------------------------------
    v8 起最佳模型按 val present-class Macro-F1 选型，而不是 Top-1。
    schema 兼容策略：**`best_accuracy` 字段保留且语义仍是 float Top-1**
    （= 最佳 Macro-F1 那个 epoch 的 val Top-1），因为下列消费方直接读它并做
    数值格式化 / 比较：
      - scripts/model_training/predict.py:15         `:.4f`
      - scripts/model_evaluation/cross_validate.py:200/210
      - scripts/model_evaluation/quick_benchmark.py:141
      - src/core/version/model_version_manager.py:243
    新指标以**新增字段** `best_macro_f1` 写入，不改动任何既有字段，
    因此 training_results.json 对旧 eval 脚本保持完全向后兼容。

    split_meta 必须写入
    -------------------
    v6 的教训：产物里没有任何切分指纹，事后完全无法判断它是否被污染
    （只能靠 "test Top-1 0.7419 反超自报 val 0.6114" 这种间接反常来推断）。
    从 v7 起，split_hash / group_by / 预处理规格一律固化进 training_results.json，
    后续任何评测都能一眼确认模型与评测集是否同源。

    val_coverage_gap 同理：某些类无 val 样本是无法事后推断的，必须显式记录。

    class_names（真实角色名）
    ------------------------
    class_to_idx 的 key 是切分 JSON 里的整数 label，所以 `sorted(class_to_idx)`
    只会得到 [0,1,2,...] 这种毫无信息量的"类名"（v7 产物即如此）。调用方应从
    load_frozen_split 返回的 idx_to_name 传入真实角色名列表（按索引升序）；
    未传时退回旧行为，仅为兼容不掌握名字的调用方。
    """
    from src.common.preprocess import describe as describe_preprocess

    results = {
        "model_name": "efficientnet_b3",
        "architecture": "EfficientNet-B3",
        "num_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "class_names": (
            list(class_names) if class_names is not None
            else sorted(class_to_idx.keys())
        ),
        "image_size": train_config.get("image_size", 256),
        # 兼容字段：仍是 Top-1 float，语义 = 最佳 Macro-F1 那个 epoch 的 val Top-1
        "best_accuracy": float(best_acc),
        # v8 新增：头条指标。present-class 口径（val 中 support>=1 的类）
        "best_macro_f1": float(best_macro_f1) if best_macro_f1 is not None else None,
        "model_selection_metric": "val_macro_f1_present_only",
        "training_time_seconds": training_time,
        "training_config": train_config,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        # ── 可追溯性字段（v6 缺失，v7 起强制）──
        "split": split_meta or {},
        "split_hash": (split_meta or {}).get("split_hash"),
        "preprocess": describe_preprocess(train_config.get("image_size", 256)),
        # ── val 覆盖缺口（时序问题：部分类无 val，事后无法判断为何指标缺失）──
        "val_coverage_gap": val_coverage_gap or {},
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
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--min-delta", type=float, default=None,
        help="最佳模型/早停判据的最小改善量（作用于 val Macro-F1），默认 "
             f"{DEFAULT_CONFIG['min_delta']}",
    )
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
        "--min-samples-per-class", type=int, default=None,
        help="按最小样本数减类：丢弃 train+val 合计样本数 < 该值的类（0=不减）。用于压低长尾、提高精度",
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
    if args.patience: config["patience"] = args.patience
    # 用 is not None 而非真值判断：--min-delta 0 是合法输入（退回严格 > 比较）
    if args.min_delta is not None: config["min_delta"] = args.min_delta
    # 同理用 is not None：--min-samples-per-class 0 是合法输入（显式声明不减类）
    if args.min_samples_per_class is not None:
        config["min_samples_per_class"] = args.min_samples_per_class
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
    train_samples, val_samples, new_class_to_idx, idx_to_name, split_meta = load_frozen_split(
        Path(args.split_dir) if args.split_dir else DEFAULT_SPLIT_DIR,
        DATA_DIR,
    )

    # ─── 可选：按最小样本数减类（提高精度、压低长尾噪声）───
    # min_samples_per_class>0 时，丢弃 train+val 合计样本数 < 该阈值的类，
    # 其余类连续重编号为 0..M-1。下游 num_classes / class_names / dataset 全部自动跟随。
    min_sp = int(config.get("min_samples_per_class", 0) or 0)
    if min_sp > 0:
        from collections import Counter
        _cnt = Counter(lbl for _, lbl in train_samples + val_samples)
        _keep = sorted(i for i, n in _cnt.items() if n >= min_sp)
        if not _keep:
            raise ValueError(
                f"min_samples_per_class={min_sp} 过滤后剩余 0 类，请降低阈值"
            )
        _remap = {old: new for new, old in enumerate(_keep)}   # 旧 idx(0..166) -> 新 idx(0..M-1)
        train_samples = [(p, _remap[l]) for p, l in train_samples if l in _remap]
        val_samples = [(p, _remap[l]) for p, l in val_samples if l in _remap]
        new_class_to_idx = {
            old_label: _remap[old_idx]
            for old_label, old_idx in new_class_to_idx.items()
            if old_idx in _remap
        }
        idx_to_name = {_remap[o]: idx_to_name[o] for o in _keep}
        logger.info(
            f"[减类] min_samples_per_class={min_sp}: 保留 {len(_keep)} 类, "
            f"丢弃 {len(_cnt) - len(_keep)} 类"
        )
        split_meta = dict(split_meta)
        split_meta["min_samples_per_class"] = min_sp
        split_meta["n_classes_after_filter"] = len(_keep)
        # n_train/n_val 会原样写进 training_results.json 的 "split"；减类后必须
        # 同步为实际训练消费的样本数，否则产物又变成"配置说一套、训练做一套"。
        split_meta["n_train"] = len(train_samples)
        split_meta["n_val"] = len(val_samples)

    num_classes = len(new_class_to_idx)
    new_idx_to_class = {idx: cls for cls, idx in new_class_to_idx.items()}
    # 真实角色名，按连续索引升序 —— 写进 training_results.json 的 class_names
    class_names = [idx_to_name[i] for i in range(num_classes)]

    logger.info(
        "切分: %s | split_hash=%s | group_by=%s",
        split_meta["split_dir"], split_meta["split_hash"], split_meta["group_by"],
    )
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Train: {len(train_samples)} images, Val: {len(val_samples)} images")

    # ─── val 覆盖缺口（仅元数据，绝不进训练）───
    # 部分类在 val.json 中无样本，无法参与早停/最佳模型选择；其中个别在
    # test.json 中仍有样本。显式列出并固化进 training_results.json。
    val_coverage_gap = compute_val_coverage_gap(
        Path(args.split_dir) if args.split_dir else DEFAULT_SPLIT_DIR,
        new_class_to_idx, val_samples, DATA_DIR,
    )
    logger.info(
        "val 覆盖: %s | 无 val 类 %d 个: %s",
        val_coverage_gap["val_class_coverage"],
        len(val_coverage_gap["missing_val_classes"]),
        ", ".join(c["class_name"] for c in val_coverage_gap["missing_val_classes"]),
    )

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

    # 学习率调度器：单周期余弦退火到 eta_min（T_max=总 epoch 数）
    # 旧 CosineAnnealingWarmRestarts(T_0=10,T_mult=2) 与 patience=8+epochs=40 三重冲突，
    # 周期3 在退火仅 15% 时被早停砍掉，白烧约 1/3。详见 outputs/v7_audit/v8_training_recommendations.md
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-6,
    )

    # val loss 专用 criterion：label_smoothing=0.0。
    # v7 的 train_one_epoch(phase="val") 内部硬编码用 LS=0.0 计算 val loss
    # （忽略了传入的 base_criterion），此处显式化以保持 val_loss 与 v7 可比。
    val_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    # ─── 断点续训 ───
    start_epoch = 0
    # v8：最佳模型按 present-class Macro-F1 选型（头条指标）。
    # best_acc 仍保留 —— 它是"最佳 Macro-F1 那个 epoch 的 val Top-1"，
    # 既用于 training_results.json 的 best_accuracy 兼容字段，
    # 也是泄漏健康检查信号（test Top-1 − val Top-1 ≈ −2.0 点为健康）。
    best_macro_f1 = 0.0
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
        # v8：恢复 Macro-F1 选型基准。从 **v7 及更早的 checkpoint** 续训时该键不存在，
        # 回退 0.0 —— 此时第一个 epoch 必然被判为"改善"并覆盖 best，这是刻意的：
        # 旧 ckpt 的 best 是按 Top-1 选出来的，本就不能作为 Macro-F1 的基准。
        best_macro_f1 = ckpt.get("best_macro_f1", 0.0)
        best_model_wts = copy.deepcopy(model.state_dict())
        # 推进 scheduler 到正确位置（CosineAnnealingLR 同样是每 epoch step 一次，
        # 逐次 step 到 start_epoch 后相位正确，此处无需改动）
        for _ in range(start_epoch):
            scheduler.step()
        logger.info(
            f"Resumed at epoch {start_epoch}, "
            f"best_macro_f1={best_macro_f1:.4f}, best_acc={best_acc:.4%}"
        )
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

        # 验证：v8 起走 evaluate_val_metrics，额外产出 present-class Macro-F1
        val_metrics = evaluate_val_metrics(
            model, val_loader, device, num_classes, val_criterion,
            max_steps=args.max_steps,
        )
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]
        val_macro_f1 = val_metrics["macro_f1"]

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
            "val_macro_f1": val_macro_f1,
            "val_present_classes": val_metrics["n_present_classes"],
            "lr": current_lr,
        })

        logger.info(
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4%} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4%}, "
            f"Val MacroF1: {val_macro_f1:.4f} "
            f"({val_metrics['n_present_classes']}/{num_classes} present) | "
            f"LR: {current_lr:.6f}"
        )

        # 保存最佳模型 —— v8 判据换成 Macro-F1（头条指标），并引入 min_delta
        # 避免浮点平局/噪声级回落驱动 patience 计数器。
        if val_macro_f1 - best_macro_f1 > config["min_delta"]:
            best_macro_f1 = val_macro_f1
            # Top-1 同步记录：作为 best_accuracy 兼容字段 + 泄漏健康检查信号
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(
                f"  *** Best model saved! Val Macro-F1: {best_macro_f1:.4f} "
                f"(Top-1 {val_acc:.4%}) ***"
            )

            # 保存 checkpoint
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                new_class_to_idx, model_dir, "model_best.pth",
                best_macro_f1=best_macro_f1,
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
    logger.info(f"Best validation Macro-F1 (present-class): {best_macro_f1:.4f}  <-- 选型指标")
    logger.info(f"Val Top-1 at that epoch: {best_acc:.4%}")
    logger.info("=" * 60)

    # 保存训练结果
    save_training_results(
        model_dir, new_class_to_idx, config, metrics_history, best_acc, training_time,
        split_meta=split_meta, val_coverage_gap=val_coverage_gap,
        best_macro_f1=best_macro_f1, class_names=class_names,
    )

    # 同时保存 model_full.pth（完整模型权重，model_loader 优先加载）
    checkpoint_final = {
        "epoch": len(metrics_history),
        "model_state_dict": model.state_dict(),
        "best_acc": best_acc,
        "best_macro_f1": float(best_macro_f1),
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
