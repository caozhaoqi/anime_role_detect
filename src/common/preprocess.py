#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一图像预处理 —— 训练与服务的唯一真源 (single source of truth)。

背景（本模块存在的理由）
------------------------
修复前，同一个 EfficientNet-B3 分类器在两端用了不同的预处理：

    训练端 scripts/model_training/train_efficientnet_b3.py
        train: Resize((288,288)) -> RandomCrop(256) -> augs -> ToTensor -> Normalize
        val  : Resize((256,256)) -> ToTensor -> Normalize
    服务端 src/core/detection/multi_target_detector.py
        Resize((224,224)) -> ToTensor -> Normalize        <-- 224!

于是存在两个独立缺陷：

1. **训练 256 vs 生产 224 的二元鸿沟**（不是历史上误传的 288/256/224 三分裂：
   288 只是 RandomCrop 的增广余量，并非某个独立的推理尺寸）。生产端把图缩到
   224 送进按 256 训练出来的权重，物体尺度整体缩小 12.5%。
2. **val 与 train 之间还有一层尺度偏移**：train 先放大到 288 再裁 256，等效视野
   是 256/288 = 88.9%；而 val 直接压到 256，视野 100%。同一个物体在 val 里比在
   train 里小 1.125 倍，早停/选模型因此建立在偏移过的分布上。

统一口径
--------
本模块把三条路径全部收敛到 **256**：

    train : Resize((288,288)) -> RandomCrop(256) -> augs -> ToTensor -> Normalize
    eval  : Resize((288,288)) -> CenterCrop(256)         -> ToTensor -> Normalize
            ^ val / test / 线上服务共用同一个 eval 变换

选 256 而不是 300 的理由：256 是既有权重的训练尺寸，改成 256 属于"消除鸿沟"，
零重训成本；300 只作为 Phase0 的分辨率探索项测量，不落地。

CenterCrop 而不是直接 Resize((256,256))：只有 Resize(288)->CenterCrop(256) 才能
让 eval 的等效视野与 train 的 RandomCrop 一致，从而消除上面第 2 条尺度偏移。

注意：不要把本模块的尺寸套用到 ONNX / CoreML 导出图上——那些图的输入形状是
静态固化的，必须先重新导出再改调用侧，否则会形状不匹配。
"""

from __future__ import annotations

import contextlib
from typing import List, Sequence, Union

import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# --- 解码策略：同样是唯一真源，导入本模块即自动继承 -------------------------
#
# 数据集实测有 16 张"仅缺尾字节"的截断 JPEG（train 11 / val 2 / test 3），
# 清单见 data/quarantine_corrupt_20260809/manifest.txt。它们是 2026-08-09 被
# 有意从隔离区恢复进 final_dataset 的，理由是"LOAD_TRUNCATED_IMAGES=True 下可解码"。
#
# 但该理由当时并不成立，构成又一处训练/服务不一致：
#   * manifest 声称 "all scripts/model_training/*.py already set" 此标志，
#     而真正在用的 train_efficientnet_b3.py **没有设**；它靠
#     FilteredImageDataset.__getitem__ 里的 `except Exception: continue`
#     静默跳到下一张 —— 这 16 张计入切分总数却从不贡献梯度。
#   * 生产侧（routes / detector）完全没有兜底，遇到同样的图会直接 OSError 崩。
#
# 收口方式：把标志提到这里，任何 `import src.common.preprocess` 的调用点
# （训练、评测、服务、诊断脚本）自动继承同一套解码语义，不在各站点重复设置。
# 语义：仅缺尾字节的图按已解码部分补齐返回，结构性损坏仍会正常抛错。
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 放宽 Pillow 的解压炸弹上限：数据集内有 >1.7 亿像素的合法大图
# （如 verina/5153039.jpg 17.9MB）。Resize 随后会立刻降采样并释放内存。
Image.MAX_IMAGE_PIXELS = 200_000_000


@contextlib.contextmanager
def allow_unlimited_pixels():
    """显式请求"临时关闭解压炸弹上限"的例外通道。

    仅数据清洗等需逐张扫描原始超大图的场景才允许使用，正常训练/推理一律继承
    模块级的 200_000_000。要点：

      * 调用方**声明**"我要用宽松模式"，策略仍归 preprocess 管、例外也归它管——
        不允许在任意脚本里裸写 `Image.MAX_IMAGE_PIXELS = None` 悄悄关闭全局防护
        （那会污染后续所有解码且无法追溯，正是我们要收口的坏味道）。
      * 进入上下文时暂存当前上限并设为 None（不触发 DecompressionBombError），
        退出时**自动恢复**模块级上限，无残留副作用。

    用法：
        from src.common.preprocess import allow_unlimited_pixels
        with allow_unlimited_pixels():
            Image.open(maybe_huge_path)  # 此刻不触发 DecompressionBombError
        # 离开 with 后自动恢复 200_000_000
    """
    saved = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = saved


__all__ = [
    "IMAGE_SIZE",
    "RESIZE_MARGIN",
    "RESIZE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INTERPOLATION",
    "ensure_rgb",
    "load_image",
    "build_train_transform",
    "build_eval_transform",
    "preprocess_image",
    "preprocess_batch",
    "describe",
    "allow_unlimited_pixels",
]

# --- 固化的预处理规格 -------------------------------------------------------
IMAGE_SIZE: int = 256          # 模型实际输入边长（train crop 后 / eval crop 后）
RESIZE_MARGIN: int = 32        # RandomCrop / CenterCrop 前预留的增广余量
RESIZE_SIZE: int = IMAGE_SIZE + RESIZE_MARGIN  # 288
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
INTERPOLATION = InterpolationMode.BILINEAR

ImageSource = Union[str, "Image.Image"]


def ensure_rgb(image: "Image.Image") -> "Image.Image":
    """统一到 3 通道 RGB。

    实测（scripts/diagnostics/repro_convert_bug.py）：RGBA/CMYK 会让 Normalize 报
    "tensor a (4) must match tensor b (3)"；P/L 会报 "output with shape [1,H,W]
    doesn't match the broadcast shape [3,H,W]"。因为 Normalize 内部是 in-place
    sub_/div_，不允许扩张广播 —— 所以 4 种非 RGB 模式全部**崩溃**，没有一种是
    静默通过的。数据集实测非 RGB 占 19.8%。
    """
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def load_image(src: ImageSource) -> "Image.Image":
    """从路径或 PIL 对象载入图像，并保证是 RGB。"""
    if isinstance(src, Image.Image):
        return ensure_rgb(src)
    with Image.open(src) as im:
        return ensure_rgb(im.copy())


def build_train_transform(
    image_size: int = IMAGE_SIZE,
    use_auto_augment: bool = True,
) -> "transforms.Compose":
    """训练变换：Resize(+margin) -> RandomCrop -> 增广 -> ToTensor -> Normalize。"""
    resize_to = image_size + RESIZE_MARGIN
    aug: List[object] = [
        transforms.Resize((resize_to, resize_to), interpolation=INTERPOLATION),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ]
    if use_auto_augment:
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy

        aug.append(AutoAugment(policy=AutoAugmentPolicy.IMAGENET))
    aug += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(aug)


def build_eval_transform(image_size: int = IMAGE_SIZE) -> "transforms.Compose":
    """评测/服务变换：Resize(+margin) -> CenterCrop -> ToTensor -> Normalize。

    val / test / 线上推理必须共用这一个函数——这是 train/serve 一致性的落点。
    等效视野与训练期 RandomCrop 相同 (image_size / (image_size+margin))。
    """
    resize_to = image_size + RESIZE_MARGIN
    return transforms.Compose(
        [
            transforms.Resize((resize_to, resize_to), interpolation=INTERPOLATION),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_image(
    src: ImageSource, image_size: int = IMAGE_SIZE, batch: bool = False
) -> "torch.Tensor":
    """一步到位：任意输入 -> 归一化后的张量。

    Args:
        src: 图片路径或 PIL Image（任意 mode）。
        image_size: 模型输入边长，默认 256。
        batch: True 时返回 (1,3,H,W)，否则 (3,H,W)。
    """
    tensor = build_eval_transform(image_size)(load_image(src))
    return tensor.unsqueeze(0) if batch else tensor


def preprocess_batch(
    sources: Sequence[ImageSource], image_size: int = IMAGE_SIZE
) -> "torch.Tensor":
    """批量预处理 -> (N,3,H,W)。"""
    if not sources:
        raise ValueError("sources is empty")
    tf = build_eval_transform(image_size)
    return torch.stack([tf(load_image(s)) for s in sources])


def describe(image_size: int = IMAGE_SIZE) -> dict:
    """返回当前预处理规格，便于写入实验记录 / 模型元数据。"""
    return {
        "image_size": image_size,
        "resize_size": image_size + RESIZE_MARGIN,
        "resize_margin": RESIZE_MARGIN,
        "interpolation": str(INTERPOLATION),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "eval_pipeline": (
            f"ensure_rgb -> Resize(({image_size + RESIZE_MARGIN},"
            f"{image_size + RESIZE_MARGIN})) -> CenterCrop({image_size}) "
            "-> ToTensor -> Normalize"
        ),
    }
