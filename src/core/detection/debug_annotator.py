#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug 标注图生成：把候选 YOLO 框按决策上色，返回 base64 JPEG data-URI。

颜色编码（与 debug 面板一致）：
- 绿 (kept && is_known)：保留且为已知角色
- 黄 (kept && !is_known)：保留但被判为未知（开集兜底）
- 红 (!kept)：被阈值/未知过滤丢弃

护栏：编码前把最长边下采样到 <=1280px，避免 debug=true 时响应体过大。
"""

import base64
from io import BytesIO
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

_MAX_LONG_EDGE = 1280

_COLOR_KEPT_KNOWN = (0, 200, 0)     # 绿
_COLOR_KEPT_UNKNOWN = (255, 200, 0)  # 黄
_COLOR_DISCARDED = (220, 20, 20)     # 红

# 置信度标签字体：优先系统 TrueType，缺失则回退默认位图字体
def _load_font(size: int = 18) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

_FONT = _load_font(size=18)


def _to_bbox_list(bbox: Any) -> List[float]:
    """归一化 bbox 为 [x1, y1, x2, y2] 浮点列表（兼容 list / dict 两种形态）。"""
    if isinstance(bbox, dict):
        return [
            float(bbox.get("x1", 0)),
            float(bbox.get("y1", 0)),
            float(bbox.get("x2", 0)),
            float(bbox.get("y2", 0)),
        ]
    return [float(v) for v in bbox]


def annotate(image: Image.Image, debug_boxes: List[Dict[str, Any]]) -> str:
    """在 image 副本上绘制 debug_boxes，返回 `data:image/jpeg;base64,...`。

    Args:
        image: PIL Image（原始待检测图）。
        debug_boxes: detect_and_classify / detect_roles 收集的调试框列表，
            每个元素含 bbox、kept、is_known_character 等键。
    """
    img = image.convert("RGB")

    # 编码前下采样：最长边 <=1280px，并记录缩放比用于同步框坐标
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > _MAX_LONG_EDGE:
        scale = _MAX_LONG_EDGE / float(long_edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        scale_factor = scale
    else:
        scale_factor = 1.0

    draw = ImageDraw.Draw(img)
    thickness = 3

    for box in debug_boxes:
        bbox = _to_bbox_list(box.get("bbox", [0, 0, 0, 0]))
        x1, y1, x2, y2 = bbox
        # 按缩放比映射到已下采样的图像坐标
        x1, y1, x2, y2 = (
            x1 * scale_factor,
            y1 * scale_factor,
            x2 * scale_factor,
            y2 * scale_factor,
        )

        kept = box.get("kept", False)
        is_known = box.get("is_known_character", False)
        if not kept:
            color = _COLOR_DISCARDED
        elif is_known:
            color = _COLOR_KEPT_KNOWN
        else:
            color = _COLOR_KEPT_UNKNOWN

        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        # 置信度文本标签：在框左上角叠加（不改动矩形颜色编码）
        conf = float(box.get("raw_confidence", 0.0))
        text = f"{conf * 100:.1f}%"
        tx = x1
        ty = max(0, y1 - 22)
        # 先画与文本尺寸匹配的实色背景小矩形，提升可读性
        try:
            tb = draw.textbbox((tx, ty), text, font=_FONT)
            draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=(0, 0, 0))
        except Exception:
            pass
        draw.text((tx, ty), text, fill=(255, 255, 255), font=_FONT)

    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    encoded = base64.b64encode(buffered.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded
