#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR 检测器

使用 EasyOCR 进行文字识别
支持预加载（preload）和就绪状态检查（is_ready）
"""

import os
import time
import threading
import easyocr
from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("ocr_detector")

# EasyOCR 模型缓存目录
_EASYOCR_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".EasyOCR", "model")


def _detect_available_languages() -> list:
    """检测本地已缓存的 EasyOCR 语言模型

    Returns:
        list: 可用的语言列表（如 ["ch_sim"] 或 ["ch_sim", "en"]）
    """
    available_langs = []

    # 检测器模型（所有语言共用）
    has_detector = os.path.exists(os.path.join(_EASYOCR_MODEL_DIR, "craft_mlt_25k.pth"))
    if not has_detector:
        logger.warning("EasyOCR 检测器模型 (craft_mlt_25k.pth) 未缓存")
        return available_langs

    # 检查各语言模型
    lang_model_map = {
        "ch_sim": "zh_sim_g2.pth",
        "en": "en_g2.pth",
    }

    for lang, model_file in lang_model_map.items():
        if os.path.exists(os.path.join(_EASYOCR_MODEL_DIR, model_file)):
            available_langs.append(lang)

    if not available_langs:
        logger.warning("EasyOCR 未检测到任何已缓存的语言模型")

    return available_langs


class EasyOCRDetector:
    """EasyOCR 文字检测器

    支持预加载和就绪状态检查，避免首次请求时长时间阻塞。
    支持懒加载 + TTL 自动卸载，空闲时释放 ~200MB 内存。
    """

    TTL_SECONDS = 180  # 3 分钟空闲后自动卸载

    def __init__(self):
        """初始化 OCR 检测器（不自动加载模型，需调用 preload() 或 detect_text() 触发懒加载）"""
        self.reader = None
        self._ready = False
        self._loading = False
        self._lock = threading.Lock()
        self._last_used_time = 0.0
        self.logger = logger

    def preload(self) -> None:
        """预加载 EasyOCR Reader（同步加载）

        根据本地已缓存的语言模型初始化 Reader。
        如果加载失败，设置 _ready=False 并记录降级日志。
        重复调用是安全的（通过 _loading 标志防止重复加载）。
        """
        with self._lock:
            if self._ready or self._loading:
                return

            self._loading = True

        try:
            available_langs = _detect_available_languages()
            if not available_langs:
                logger.warning("[DEGRADE] EasyOCR 无可用语言模型，OCR 功能不可用")
                self._ready = False
                return

            logger.info(f"EasyOCR 预加载开始，语言: {available_langs}")
            self.reader = easyocr.Reader(available_langs, gpu=False)
            self._ready = True
            logger.info("EasyOCR 预加载完成，OCR 就绪")
        except Exception as e:
            logger.error(f"[DEGRADE] EasyOCR 预加载失败: {e}")
            self._ready = False
            self.reader = None
        finally:
            with self._lock:
                self._loading = False

    def is_ready(self) -> bool:
        """检查 OCR 检测器是否就绪

        Returns:
            bool: 是否已加载完成且可用
        """
        return self._ready and self.reader is not None

    def detect_text(self, image_source):
        """
        检测图像中的文字

        如果检测器未就绪，尝试懒加载。如果加载失败，返回空列表并记录降级日志。

        Args:
            image_source: 图像路径(str)、PIL Image 或 numpy 数组

        Returns:
            list: 文字检测结果，每个元素包含文字、置信度和边界框
        """
        # 懒加载保障：未就绪时自动尝试加载
        if not self.is_ready():
            self.logger.info("[LazyLoad] EasyOCR 首次使用，触发懒加载...")
            self.preload()
        if not self.is_ready():
            logger.warning("[DEGRADE] OCR 检测器未就绪，跳过文字检测")
            return []
        self._last_used_time = time.time()

        try:
            # EasyOCR 接受文件路径、bytes、numpy 数组，不接受 PIL Image
            import numpy as np
            _source = image_source
            if hasattr(image_source, "__array__") or hasattr(image_source, "convert"):
                # PIL Image → numpy array (RGB)
                _source = np.array(image_source.convert("RGB") if hasattr(image_source, "convert") else image_source)

            logger.info(f"开始 OCR 文字检测")
            results = self.reader.readtext(_source)

            # 处理结果
            text_detections = []
            for bbox, text, confidence in results:
                # 过滤掉只有特殊字符的文本
                if text and not (len(text) == 1 and text in ["@", "#", "$", "%", "^", "&", "*"]):
                    # 转换边界框格式
                    bbox_flat = []
                    for point in bbox:
                        bbox_flat.extend(point)

                    text_detections.append(
                        {"text": text, "confidence": float(confidence),
                         "bbox": [float(v) for v in bbox_flat]}
                    )

            logger.info(f"OCR 文字检测完成，检测到 {len(text_detections)} 个文本区域")
            return text_detections
        except Exception as e:
            logger.error(f"OCR 文字检测失败: {e}")
            return []


    def unload_if_idle(self) -> bool:
        """空闲超过 TTL 时自动卸载 Reader，释放 ~200MB 内存

        Returns:
            bool: 是否执行了卸载
        """
        if self.reader is None:
            return False
        if self._last_used_time == 0:
            return False
        idle_seconds = time.time() - self._last_used_time
        if idle_seconds < self.TTL_SECONDS:
            return False
        self.logger.info(
            f"[TTL] EasyOCR 空闲 {idle_seconds:.0f}s 超过 {self.TTL_SECONDS}s，卸载释放内存"
        )
        self.reader = None
        self._ready = False
        import gc
        gc.collect()
        self.logger.info("[TTL] EasyOCR Reader 已卸载")
        return True


# 全局 OCR 检测器实例
_ocr_detector = None


def get_ocr_detector():
    """
    获取 OCR 检测器实例

    Returns:
        EasyOCRDetector: OCR 检测器实例
    """
    global _ocr_detector
    if _ocr_detector is None:
        _ocr_detector = EasyOCRDetector()
    return _ocr_detector


def detect_text(image_path):
    """
    检测图像中的文字

    Args:
        image_path: 图像路径

    Returns:
        list: 文字检测结果
    """
    detector = get_ocr_detector()
    return detector.detect_text(image_path)
