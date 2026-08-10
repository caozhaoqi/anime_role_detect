#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练路径 vs 服务路径 预处理一致性测试。

核心断言：同一张图分别走「训练端 val 变换」和「服务端推理变换」，
输出 tensor 必须**逐字节相同**。这同时验证了：
  #1 convert("RGB") 缺失修复（非 RGB 模式不再崩溃）
  #3 训练 256 / 生产 224 鸿沟已消除

运行：
  .venv/bin/python -m pytest tests/test_preprocess_parity.py -v
  或直接：.venv/bin/python tests/test_preprocess_parity.py
"""

import os
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.preprocess import (  # noqa: E402
    IMAGE_SIZE,
    RESIZE_SIZE,
    build_eval_transform,
    build_train_transform,
    ensure_rgb,
    load_image,
    preprocess_batch,
    preprocess_image,
)

MODES = ["RGB", "RGBA", "P", "L", "CMYK"]


def _img(mode="RGB", size=(137, 91)):
    """构造一张有内容的小图（非纯色，确保 resize/crop 差异能被检出）。"""
    base = Image.new("RGB", size)
    px = base.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = ((x * 7) % 256, (y * 13) % 256, ((x + y) * 3) % 256)
    return base if mode == "RGB" else base.convert(mode)


# ---------------------------------------------------------------------------
# 核心：训练路径 vs 服务路径逐字节一致
# ---------------------------------------------------------------------------
def test_train_val_and_serving_transform_are_byte_identical():
    """训练端 val 变换 与 服务端推理变换 必须产出完全相同的 tensor。"""
    from scripts.model_training.train_efficientnet_b3 import get_transforms

    img = _img("RGB")
    _, val_transform = get_transforms(image_size=IMAGE_SIZE)
    train_side = val_transform(ensure_rgb(img))
    serving_side = build_eval_transform()(ensure_rgb(img))

    assert train_side.shape == serving_side.shape
    assert torch.equal(train_side, serving_side), (
        "训练端 val 变换与服务端变换输出不一致，最大差异="
        f"{(train_side - serving_side).abs().max().item()}"
    )


def test_detector_transform_matches_common_source():
    """MultiTargetDetector 的 transform 必须来自同一真源（不再是 224）。"""
    from src.core.detection.multi_target_detector import CLASSIFIER_INPUT_SIZE

    assert CLASSIFIER_INPUT_SIZE == IMAGE_SIZE == 256, (
        f"服务端输入尺寸 {CLASSIFIER_INPUT_SIZE} 与训练尺寸 {IMAGE_SIZE} 不一致"
    )


def test_preprocess_image_matches_eval_transform():
    """preprocess_image 便捷函数与 build_eval_transform 结果一致。"""
    img = _img("RGB")
    assert torch.equal(preprocess_image(img), build_eval_transform()(ensure_rgb(img)))


# ---------------------------------------------------------------------------
# #1 回归：非 RGB 模式不得崩溃，且必须是 3 通道
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_all_pil_modes_yield_three_channels(mode):
    """RGBA/P/L/CMYK 修复前会在 Normalize 处 RuntimeError，修复后必须正常。"""
    tensor = preprocess_image(_img(mode))
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE), (
        f"mode={mode} 输出形状 {tuple(tensor.shape)}，期望 (3,{IMAGE_SIZE},{IMAGE_SIZE})"
    )
    assert torch.isfinite(tensor).all(), f"mode={mode} 输出含 NaN/Inf"


@pytest.mark.parametrize("mode", ["RGBA", "CMYK"])
def test_lossless_modes_equal_rgb_baseline(mode):
    """RGBA/CMYK 是无损可逆的，转 RGB 后应与 RGB 基准逐字节相同。"""
    ref = preprocess_image(_img("RGB"))
    got = preprocess_image(_img(mode))
    assert torch.equal(ref, got), f"mode={mode} 与 RGB 基准不一致"


def test_ensure_rgb_is_idempotent():
    img = _img("RGBA")
    once = ensure_rgb(img)
    assert once.mode == "RGB"
    assert ensure_rgb(once) is once  # 已是 RGB 时不应再复制


# ---------------------------------------------------------------------------
# 规格固化：防止有人偷偷改回 224 或改掉 CenterCrop
# ---------------------------------------------------------------------------
def test_spec_is_pinned_to_256():
    assert IMAGE_SIZE == 256
    assert RESIZE_SIZE == 288


def test_eval_pipeline_has_resize_then_centercrop():
    """eval 变换必须是 Resize(288) -> CenterCrop(256)，而不是直接 Resize(256)。

    直接 Resize(256) 会让 val/线上 的等效视野是 100%，而训练期 RandomCrop 的
    视野是 256/288=88.9%，两者存在 1.125x 尺度偏移。
    """
    from torchvision import transforms as T

    ops = build_eval_transform().transforms
    assert isinstance(ops[0], T.Resize)
    assert isinstance(ops[1], T.CenterCrop)
    assert tuple(ops[0].size) == (RESIZE_SIZE, RESIZE_SIZE)
    assert tuple(ops[1].size) == (IMAGE_SIZE, IMAGE_SIZE)


def test_train_transform_crops_to_image_size():
    out = build_train_transform(use_auto_augment=False)(_img("RGB"))
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_batch_preprocess_shape():
    batch = preprocess_batch([_img("RGB"), _img("RGBA"), _img("L")])
    assert batch.shape == (3, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_load_image_from_path(tmp_path):
    p = tmp_path / "sample.png"
    _img("RGBA").save(p)
    assert load_image(str(p)).mode == "RGB"


# ---------------------------------------------------------------------------
# 截断图：训练路径与服务路径都不得崩溃，且结果一致
# ---------------------------------------------------------------------------
def _truncated_jpeg(tmp_path, drop_bytes=42):
    """造一张"仅缺尾字节"的截断 JPEG，复刻数据集里那 16 张的形态。

    数据集实测 16 张截断 JPEG（train 11 / val 2 / test 3），缺失 0~114 字节。
    没有 LOAD_TRUNCATED_IMAGES 时它们会抛
    OSError: image file is truncated (N bytes not processed)。
    """
    good = tmp_path / "good.jpg"
    _img("RGB", size=(320, 240)).save(good, "JPEG", quality=95)
    raw = good.read_bytes()
    bad = tmp_path / "truncated.jpg"
    bad.write_bytes(raw[:-drop_bytes])  # 砍掉尾部字节
    return bad


def test_truncated_jpeg_decodes_via_single_source(tmp_path):
    """截断图必须能解码——策略由 src/common/preprocess 统一提供。

    回归目标：生产侧过去没有任何兜底，遇到这类图直接 OSError 崩；
    而训练侧靠 except-continue 静默跳过。两边现在共用同一套解码语义。
    """
    bad = _truncated_jpeg(tmp_path)
    img = load_image(str(bad))
    assert img.mode == "RGB"
    assert img.size == (320, 240)


def test_truncated_jpeg_train_and_serving_paths_agree(tmp_path):
    """同一张截断 JPEG 走训练端 val 变换与服务端变换，都不崩且逐字节一致。"""
    from scripts.model_training.train_efficientnet_b3 import get_transforms

    bad = _truncated_jpeg(tmp_path)
    img = load_image(str(bad))

    _, val_transform = get_transforms(image_size=IMAGE_SIZE)
    train_side = val_transform(img)
    serving_side = build_eval_transform()(img)

    assert train_side.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.equal(train_side, serving_side), (
        "截断图在训练路径与服务路径下结果不一致，最大差异="
        f"{(train_side - serving_side).abs().max().item()}"
    )
    assert torch.isfinite(train_side).all()


def test_truncated_jpeg_goes_through_dataset_without_silent_skip(tmp_path):
    """截断图必须被 FilteredImageDataset 正常读入，而不是被静默跳过。

    过去 __getitem__ 是 `except Exception: continue`，坏图会被悄悄换成
    下一张样本 —— 标签仍是原样本的，既污染数据又无声无息。
    现在截断图能真正解码，因此不应触发任何 decode_failures。
    """
    from scripts.model_training.train_efficientnet_b3 import FilteredImageDataset

    bad = _truncated_jpeg(tmp_path)
    ds = FilteredImageDataset(
        [(str(bad), 7)], build_eval_transform(), {7: "dummy"}, image_size=IMAGE_SIZE
    )
    tensor, label = ds[0]

    assert label == 7, "标签被换成了别的样本——说明发生了静默跳过"
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert ds.decode_failures == {}, f"不应有解码失败: {ds.decode_failures}"


def test_dataset_records_failure_instead_of_silent_skip(tmp_path, caplog):
    """真正无法解码的文件必须留下 warning 日志 + 可追溯记录，不得静默吞掉。"""
    import logging

    from scripts.model_training.train_efficientnet_b3 import FilteredImageDataset

    junk = tmp_path / "not_an_image.jpg"
    junk.write_bytes(b"this is definitely not a jpeg")
    good = tmp_path / "ok.jpg"
    _img("RGB").save(good, "JPEG")

    ds = FilteredImageDataset(
        [(str(junk), 3), (str(good), 5)],
        build_eval_transform(),
        {3: "a", 5: "b"},
        image_size=IMAGE_SIZE,
    )
    with caplog.at_level(logging.WARNING):
        tensor, label = ds[0]  # 坏图 -> 跳到下一张，但必须留痕

    assert label == 5, "应回退到下一个可解码样本"
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert str(junk) in ds.decode_failures, "坏样本未被记录，仍是静默跳过"
    assert any("decode-fail" in r.message or "decode-fail" in r.getMessage()
               for r in caplog.records), "未产生 warning 级日志"


def test_truncation_policy_owned_by_preprocess_only():
    """解码策略只能由 preprocess 设置，任何调用点不得重复设置（全仓唯一真源）。

    全仓扫描：src/common/preprocess.py 是唯一合法设置位置；本测试文件自身含
    正则字符串，已排除。其余任何 .py 重复设置都会失败——形成回归护栏，能拦住
    未来在任何文件里重新写死解码策略的行为。

    已知遗留站点（known_legacy）：现已完全清空——全仓除 preprocess.py 自身外
    再无任何 `LOAD_TRUNCATED_IMAGES=` / `MAX_IMAGE_PIXELS=` 赋值点。3 个训练
    辅助脚本已收口到唯一真源；2 个 data_cleaning 脚本改用 preprocess 暴露的
    allow_unlimited_pixels() 显式 API（而非裸写 MAX_IMAGE_PIXELS=None）。
    护栏放行该 API 调用（不匹配赋值正则），但继续禁止任何裸赋值——达到"全仓零例外"。
    """
    import re
    from pathlib import Path

    from PIL import Image as PILImage, ImageFile

    # 导入 preprocess 即应生效
    assert ImageFile.LOAD_TRUNCATED_IMAGES is True
    assert PILImage.MAX_IMAGE_PIXELS == 200_000_000

    root = Path(__file__).resolve().parent.parent
    # 唯一合法的设置位置
    owner = (root / "src" / "common" / "preprocess.py").resolve()
    self_file = Path(__file__).resolve()
    # 跳过的目录（虚拟环境 / 第三方 / 构建产物）
    skip_dirs = {".venv", "node_modules", "__pycache__", ".git", "build", "dist"}
    # 已知遗留站点：现已全部收口，本列表清空——护栏达到"全仓零例外"状态。
    #   * 3 个训练辅助脚本（three_experiments/compare_train_test/train_three_models）
    #     已收口到唯一真源；
    #   * 2 个 data_cleaning 脚本改用 preprocess.allow_unlimited_pixels() 显式
    #     API（不再裸写 MAX_IMAGE_PIXELS=None）。全仓扫描确认除 preprocess.py
    #     自身外再无任何赋值点，故不再需要白名单。
    known_legacy = set()
    known_legacy = {p.resolve() for p in known_legacy}
    offenders = []
    for py in root.rglob("*.py"):
        rp = py.resolve()
        if rp == owner or rp == self_file or rp in known_legacy:
            continue
        if skip_dirs & set(rp.relative_to(root).parts):
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # 全仓扫描：任何调用点重复设置解码策略都算破坏唯一真源
        for ln, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.match(r"^(ImageFile\.)?LOAD_TRUNCATED_IMAGES\s*=", stripped) or \
               re.match(r"^(Image\.)?MAX_IMAGE_PIXELS\s*=", stripped):
                offenders.append(f"{rp.relative_to(root)}:{ln}: {stripped}")
    assert not offenders, "调用点重复设置解码策略，破坏唯一真源:\n" + "\n".join(offenders)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header"]))
