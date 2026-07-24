#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键点检测进程池

使用 concurrent.futures.ProcessPoolExecutor 维护常驻 worker 进程，
替代每次请求 subprocess.run fork 新进程的开销。

注意：子进程内不导入 torch，仅 lazily import mediapipe。
"""

import os
import sys
import base64
import json
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

from PIL import Image
from io import BytesIO

from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("keypoint_worker")

# 确保 PYTHONPATH 正确设置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class KeypointWorkerPool:
    """常驻 worker 进程池，替代每次 subprocess.run fork"""

    def __init__(self, num_workers: int = 2):
        """初始化进程池配置

        Args:
            num_workers: worker 进程数量，默认 2
        """
        self._pool: Optional[ProcessPoolExecutor] = None
        self._num_workers: int = num_workers

    def start(self) -> None:
        """初始化进程池"""
        if self._pool is not None:
            logger.warning("进程池已启动，跳过重复初始化")
            return
        self._pool = ProcessPoolExecutor(
            max_workers=self._num_workers,
            mp_context=__import__("multiprocessing").get_context("spawn"),
        )
        logger.info(f"关键点检测进程池已创建 (workers={self._num_workers})")

    async def detect_keypoints(self, image: Image.Image) -> List:
        """提交关键点检测任务到 worker 进程

        Args:
            image: PIL.Image 对象

        Returns:
            关键点列表，检测失败时返回空列表
        """
        if self._pool is None:
            logger.error("进程池未启动，无法执行检测")
            return []

        try:
            # 将图像编码为 base64 传递给子进程
            buf = BytesIO()
            image.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # 在事件循环中运行子进程任务
            loop = asyncio.get_running_loop()
            result_json = await loop.run_in_executor(
                self._pool, KeypointWorkerPool._worker_entry, img_b64
            )

            if result_json:
                keypoints = json.loads(result_json)
                logger.info(f"关键点检测完成: {len(keypoints)} 个关键点")
                return keypoints
            else:
                logger.warning("关键点检测返回空结果")
                return []
        except Exception as e:
            logger.warning(f"关键点检测失败: {e}")
            return []

    def shutdown(self) -> None:
        """关闭进程池"""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
            logger.info("关键点检测进程池已关闭")

    @staticmethod
    def _worker_entry(img_b64: str) -> str:
        """子进程入口：lazy import mediapipe，执行检测

        注意：此函数在子进程中执行，不导入 torch。
        传入 base64 编码的 JPEG 图像，返回 JSON 编码的关键点列表。

        Args:
            img_b64: base64 编码的 JPEG 图像字符串

        Returns:
            JSON 编码的关键点列表字符串，失败时返回空字符串
        """
        import base64 as _b64
        import json as _json
        from io import BytesIO as _BytesIO

        try:
            from PIL import Image as _Image
            img = _Image.open(_BytesIO(_b64.b64decode(img_b64))).convert("RGB")

            # 在子进程内 lazily import mediapipe（不导入 torch）
            from src.core.keypoint.mediapipe_keypoint_detector import detect_keypoints
            kps = detect_keypoints(img)
            return _json.dumps(kps)
        except Exception:
            return ""
