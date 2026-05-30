#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理任务
用于异步处理图像相关任务
"""

from src.core.celery_config import app
from src.core.onnx_engine import get_engine
from src.core.prometheus_metrics import MetricsCollector
from PIL import Image
import io
import os
import time


@app.task(bind=True, queue="image_queue")
def process_image(self, image_path: str, model_name: str = "yolov8n"):
    """
    处理单张图片

    Args:
        image_path: 图片路径
        model_name: 模型名称
    """
    start_time = time.time()

    try:
        # 加载图片
        image = Image.open(image_path).convert("RGB")

        # 加载模型并推理
        engine = get_engine(model_name)
        result = engine.predict(image)

        # 计算耗时
        duration = time.time() - start_time

        # 记录指标
        MetricsCollector.record_inference_time(model_name=model_name, duration=duration)

        return {
            "status": "success",
            "image_path": image_path,
            "model_name": model_name,
            "result_shape": result.shape,
            "inference_time_ms": duration * 1000,
        }

    except Exception as e:
        return {"status": "error", "image_path": image_path, "error": str(e)}


@app.task(bind=True, queue="image_queue")
def process_images_batch(self, image_paths: list, model_name: str = "yolov8n"):
    """
    批量处理图片

    Args:
        image_paths: 图片路径列表
        model_name: 模型名称
    """
    start_time = time.time()

    try:
        # 加载模型
        engine = get_engine(model_name)

        # 加载所有图片
        images = []
        valid_paths = []
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(image)
                valid_paths.append(path)
            except Exception as e:
                print(f"无法加载图片: {path}, 错误: {e}")

        # 批量推理
        results = engine.predict_batch(images)

        # 计算耗时
        duration = time.time() - start_time

        # 记录指标
        MetricsCollector.record_inference_time(model_name=model_name, duration=duration)

        return {
            "status": "success",
            "total_count": len(image_paths),
            "processed_count": len(valid_paths),
            "model_name": model_name,
            "result_shape": results.shape,
            "inference_time_ms": duration * 1000,
            "avg_time_per_image_ms": (duration / len(valid_paths)) * 1000 if valid_paths else 0,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.task(bind=True, queue="image_queue")
def sync_images_task(self):
    """
    同步图片任务
    将多个来源的图片同步到统一目录
    """
    import subprocess

    try:
        # 执行同步脚本
        result = subprocess.run(
            ["python3", "scripts/analysis/sync_images_to_merged.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        )

        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.task(bind=True, queue="image_queue")
def cleanup_low_quality_images(self, directory: str):
    """
    清理低质量图片

    Args:
        directory: 图片目录
    """
    import subprocess

    try:
        # 执行清理脚本
        result = subprocess.run(
            ["python3", "scripts/analysis/clean_low_quality.py", "--dir", directory],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        )

        return {
            "status": "success",
            "directory": directory,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
