#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频渲染器
在视频帧上绘制识别结果（识别框 + 角色名 + 置信度），输出带标注的结果视频
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger


# 绘图风格常量
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
BOX_PADDING = 4
COLORS = [
    (0, 255, 0),    # 绿色
    (255, 0, 0),    # 蓝色
    (0, 0, 255),    # 红色
    (255, 255, 0),  # 青色
    (255, 0, 255),  # 品红
    (0, 255, 255),  # 黄色
    (128, 255, 0),  # 黄绿
    (255, 128, 0),  # 橙色
    (0, 128, 255),  # 天蓝
    (255, 0, 128),  # 粉紫
]


def _get_color(index: int) -> Tuple[int, int, int]:
    """根据索引获取颜色"""
    return COLORS[index % len(COLORS)]


def _draw_text_with_background(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = FONT_SCALE,
    thickness: int = FONT_THICKNESS,
) -> None:
    """
    在帧上绘制带背景框的文字（更清晰易读）
    """
    # 获取文字尺寸
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)

    x, y = pos
    # 背景框
    bg_x1 = x - BOX_PADDING
    bg_y1 = y - text_h - BOX_PADDING
    bg_x2 = x + text_w + BOX_PADDING
    bg_y2 = y + baseline + BOX_PADDING

    # 绘制半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 绘制白色文字
    cv2.putText(frame, text, (x, y), FONT, font_scale, (255, 255, 255), thickness)


def draw_recognition_on_frame(
    frame: np.ndarray,
    roles: List[Dict],
    timestamp: float,
    frame_number: int,
) -> np.ndarray:
    """
    在单帧上绘制识别结果

    Args:
        frame: OpenCV BGR 帧
        roles: 识别结果列表，每项包含 role, similarity, 可选 box (x,y,w,h)
        timestamp: 时间戳（秒）
        frame_number: 帧序号

    Returns:
        标注后的帧
    """
    h, w = frame.shape[:2]

    # 在左上角显示帧信息和时间戳
    info_text = f"Frame: {frame_number} | Time: {timestamp:.2f}s"
    _draw_text_with_background(frame, info_text, (12, 30), (100, 100, 100), 0.5, 1)

    # 为每个识别的角色绘制信息
    for i, role_info in enumerate(roles):
        role_name = role_info.get("role", "unknown")
        similarity = role_info.get("similarity", 0.0)
        box = role_info.get("box")  # 可选: {x, y, w, h}

        color = _get_color(i)
        label = f"{role_name} ({similarity * 100:.0f}%)"

        if box:
            # 有边界框：绘制框 + 标签
            x, y, bw, bh = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
            # 确保坐标在帧范围内
            x, y = max(0, x), max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            # 在框上方绘制标签
            label_pos = (x, max(y - 8, 20))
            _draw_text_with_background(frame, label, label_pos, color)
        else:
            # 无边界框：在右上角列表显示
            label_x = w - 250
            label_y = 60 + i * 35
            _draw_text_with_background(frame, label, (label_x, label_y), color)

    return frame


def render_result_video(
    video_path: str,
    results: List[Dict],
    output_path: str,
    frame_interval: float = 1.0,
) -> Optional[str]:
    """
    根据识别结果生成带标注的结果视频

    Args:
        video_path: 原始视频路径
        results: 识别结果（与 /video/recognize 返回格式一致）
        output_path: 输出视频路径
        frame_interval: 抽帧间隔（秒），需与识别时使用的值一致

    Returns:
        输出视频路径，失败返回 None
    """
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return None

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"视频属性: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

    # 索引结果 by 帧号
    result_by_frame = {}
    for r in results:
        fn = r.get("frame_number")
        if fn is not None:
            result_by_frame[fn] = r

    # 准备视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        logger.error(f"无法创建输出视频: {output_path}")
        cap.release()
        return None

    frame_count = 0
    success, frame = cap.read()
    processed = 0

    while success:
        if frame_count in result_by_frame:
            # 该帧有识别结果，绘制标注
            result = result_by_frame[frame_count]
            frame = draw_recognition_on_frame(
                frame,
                result.get("roles", []),
                result.get("timestamp", frame_count / fps),
                frame_count,
            )
            processed += 1

        # 写入帧
        out.write(frame)
        frame_count += 1
        success, frame = cap.read()

        # 进度日志（每10%输出一次）
        if total_frames > 0 and frame_count % max(1, total_frames // 10) == 0:
            pct = frame_count * 100 // total_frames
            logger.info(f"视频渲染进度: {pct}% ({frame_count}/{total_frames})")

    cap.release()
    out.release()

    logger.info(
        f"视频渲染完成: {processed}/{len(results)} 帧已标注, 输出: {output_path}"
    )
    return output_path