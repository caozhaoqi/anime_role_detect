#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理任务
用于异步处理视频相关任务
"""

from src.core.celery_config import app
from src.core.onnx_engine import get_engine
from src.core.prometheus_metrics import MetricsCollector
import cv2
import os
import time


@app.task(bind=True, queue="video_queue")
def process_video(self, video_path: str, model_name: str = "yolov8n", frame_interval: int = 10):
    """
    处理视频文件

    Args:
        video_path: 视频路径
        model_name: 模型名称
        frame_interval: 帧间隔（每隔多少帧处理一帧）
    """
    start_time = time.time()

    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "error": "无法打开视频文件"}

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 加载模型
        engine = get_engine(model_name)

        # 处理帧
        frame_count = 0
        processed_frames = 0
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔 frame_interval 帧处理一帧
            if frame_count % frame_interval == 0:
                # 转换颜色空间并预处理
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image

                image = Image.fromarray(frame_rgb)

                # 推理
                result = engine.predict(image)
                results.append(
                    {
                        "frame": frame_count,
                        "result_shape": result.shape,
                        "prediction": (
                            int(result.argmax()) if len(result.shape) == 2 else "detection"
                        ),
                    }
                )
                processed_frames += 1

            frame_count += 1

            # 更新任务进度
            progress = (frame_count / total_frames) * 100
            self.update_state(state="PROGRESS", meta={"progress": progress})

        cap.release()

        # 计算耗时
        duration = time.time() - start_time

        # 记录指标
        MetricsCollector.record_inference_time(model_name=model_name, duration=duration)

        return {
            "status": "success",
            "video_path": video_path,
            "model_name": model_name,
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "width": width,
            "height": height,
            "inference_time_ms": duration * 1000,
            "avg_time_per_frame_ms": (
                (duration / processed_frames) * 1000 if processed_frames else 0
            ),
            "results": results,
        }

    except Exception as e:
        return {"status": "error", "video_path": video_path, "error": str(e)}


@app.task(bind=True, queue="video_queue")
def extract_frames(self, video_path: str, output_dir: str, frame_interval: int = 10):
    """
    从视频中提取帧

    Args:
        video_path: 视频路径
        output_dir: 输出目录
        frame_interval: 帧间隔
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "error": "无法打开视频文件"}

        frame_count = 0
        extracted_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_count += 1

            frame_count += 1

            # 更新进度
            progress = (frame_count / int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) * 100
            self.update_state(state="PROGRESS", meta={"progress": progress})

        cap.release()

        return {
            "status": "success",
            "video_path": video_path,
            "output_dir": output_dir,
            "total_frames": frame_count,
            "extracted_frames": extracted_count,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
