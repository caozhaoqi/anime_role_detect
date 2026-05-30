#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR 检测器

使用 EasyOCR 进行文字识别
"""

import easyocr
from src.core.logging.global_logger import get_logger

logger = get_logger("ocr_detector")


class EasyOCRDetector:
    """EasyOCR 文字检测器"""

    def __init__(self):
        """初始化 OCR 检测器"""
        self.reader = None
        self._initialize_reader()

    def _initialize_reader(self):
        """初始化 EasyOCR 读取器"""
        try:
            # 只加载中英文模型
            self.reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
            logger.info("EasyOCR 初始化成功")
        except Exception as e:
            logger.error(f"EasyOCR 初始化失败: {e}")
            self.reader = None

    def detect_text(self, image_source):
        """
        检测图像中的文字

        Args:
            image_source: 图像路径或文件对象

        Returns:
            list: 文字检测结果，每个元素包含文字、置信度和边界框
        """
        if self.reader is None:
            logger.warning("OCR 检测器未初始化，跳过文字检测")
            return []

        try:
            logger.info(f"开始 OCR 文字检测")
            results = self.reader.readtext(image_source)

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
                        {"text": text, "confidence": float(confidence), "bbox": bbox_flat}
                    )

            logger.info(f"OCR 文字检测完成，检测到 {len(text_detections)} 个文本区域")
            return text_detections
        except Exception as e:
            logger.error(f"OCR 文字检测失败: {e}")
            return []


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
