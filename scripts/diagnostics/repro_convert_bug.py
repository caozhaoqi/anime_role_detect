#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小复现：multi_target_detector 缺失 .convert("RGB") 的后果。

构造 RGBA / P / L / CMYK / RGB 五种模式的小图，分别走：
  BEFORE = 修复前管线（无 convert，且手动 resize 一次 + transform 再 resize 一次）
  AFTER  = 修复后管线（ensure_rgb + 单点 resize）

不需要加载 B3 权重，只验证进入模型前的 tensor 是否成立/是否正确。
EfficientNet-B3 首层要求 3 通道，任何非 3 通道张量即为故障。

运行：
  .venv/bin/python scripts/diagnostics/repro_convert_bug.py
"""

import os
import sys

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.detection.multi_target_detector import (  # noqa: E402
    CLASSIFIER_INPUT_SIZE,
    ensure_rgb,
)

# 与 MultiTargetDetector.__init__ 中完全一致的 transform
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def make_images():
    """构造各模式的小图（64x64），模拟真实数据集中的模式分布。"""
    base = Image.new("RGB", (64, 64), (200, 120, 40))
    return {
        "RGB": base,
        "RGBA": base.convert("RGBA"),
        "P": base.convert("P"),
        "L": base.convert("L"),
        "CMYK": base.convert("CMYK"),
    }


def run_before(img):
    """修复前：直接 crop -> 手动 resize -> transform，全程无 convert。"""
    cropped = img.crop((0, 0, 64, 64))  # crop 保留原 mode
    crop_resized = cropped.resize((224, 224), Image.BILINEAR)
    return TRANSFORM(crop_resized).unsqueeze(0)


def run_after(img):
    """修复后：ensure_rgb 之后再走单点 resize 的 transform。"""
    img = ensure_rgb(img)
    cropped = img.crop((0, 0, 64, 64))
    cropped = ensure_rgb(cropped)
    return TRANSFORM(cropped).unsqueeze(0)


def describe(fn, img):
    try:
        t = fn(img)
    except Exception as e:
        return "CRASH", f"{type(e).__name__}: {str(e)[:90]}"
    ch = t.shape[1]
    if ch != 3:
        return "BAD_SHAPE", f"shape={tuple(t.shape)} 通道数={ch} (模型要求 3)"
    return "OK", f"shape={tuple(t.shape)} mean={t.mean().item():+.4f}"


def main():
    imgs = make_images()
    ref = run_after(imgs["RGB"])

    print("=" * 78)
    print("最小复现：缺失 convert('RGB') 的实际后果")
    print(f"torch={torch.__version__}  CLASSIFIER_INPUT_SIZE={CLASSIFIER_INPUT_SIZE}")
    print("=" * 78)
    header = f"{'mode':<6} | {'BEFORE(修复前)':<46} | {'AFTER(修复后)':<34}"
    print(header)
    print("-" * 78)

    n_crash = n_silent = 0
    for mode, img in imgs.items():
        s_before, d_before = describe(run_before, img)
        s_after, d_after = describe(run_after, img)
        if s_before == "CRASH":
            n_crash += 1
        elif s_before == "BAD_SHAPE":
            n_silent += 1
        print(f"{mode:<6} | [{s_before:<9}] {d_before:<34} | [{s_after:<9}] {d_after}")

    print("-" * 78)
    print(f"修复前：崩溃 {n_crash} 种模式，静默错误 {n_silent} 种模式")

    # 语义校验：非 RGB 模式修复后应与 RGB 基准数值接近（P/L 有损，仅校验形状）
    print("\n[修复后一致性校验] 所有模式输出形状必须为 (1,3,H,W):")
    all_ok = True
    for mode, img in imgs.items():
        t = run_after(img)
        ok = tuple(t.shape) == tuple(ref.shape)
        all_ok &= ok
        print(f"  {mode:<6} shape={tuple(t.shape)}  {'✅' if ok else '❌'}")

    # RGBA/CMYK 是无损可逆的，数值应与 RGB 基准完全一致
    print("\n[数值等价校验] RGBA/CMYK 转回 RGB 后应与 RGB 基准接近:")
    for mode in ("RGBA", "CMYK"):
        t = run_after(imgs[mode])
        diff = (t - ref).abs().max().item()
        print(f"  {mode:<6} max|diff| vs RGB = {diff:.6f}")

    print("\n结论:", "✅ 修复后全部模式均产出合法 3 通道张量" if all_ok else "❌ 仍有模式异常")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
