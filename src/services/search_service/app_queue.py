#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像搜索服务 - 队列版本
通过文件队列与独立Worker进程通信，避免macOS线程锁问题
"""

import os
import sys
import time
import uuid
import json
import logging
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# 队列目录
QUEUE_DIR = "data/search_queue"
INPUT_DIR = os.path.join(QUEUE_DIR, "input")
OUTPUT_DIR = os.path.join(QUEUE_DIR, "output")

# 确保目录存在
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 创建FastAPI应用
app = FastAPI(
    title="图像搜索与视频识别API (Queue)",
    description="通过文件队列与独立Worker进程通信",
    version="1.0.0",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "Search Service (Queue)"}


@app.post("/api/search/image")
async def search_similar_images(file: UploadFile = File(...), top_k: int = 10):
    """
    以图搜图 - 搜索相似图像
    通过文件队列与独立Worker进程通信
    """
    try:
        task_id = str(uuid.uuid4())

        content = await file.read()
        input_path = os.path.join(INPUT_DIR, f"{task_id}.jpg")

        image = Image.open(io.BytesIO(content)).convert("RGB")
        image.save(input_path, format="JPEG")

        output_path = os.path.join(OUTPUT_DIR, f"{task_id}.json")
        timeout = 30
        start_time = time.time()
        poll_count = 0

        while time.time() - start_time < timeout:
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    result = json.load(f)

                os.remove(output_path)

                if result["status"] == "success":
                    return {
                        "query": file.filename,
                        "count": len(result["results"]),
                        "results": result["results"],
                        "model": result.get("model", "Unknown"),
                        "processing_time_ms": int((time.time() - start_time) * 1000),
                    }
                else:
                    raise HTTPException(status_code=500, detail=result.get("message", "搜索失败"))

            poll_count += 1
            if poll_count % 50 == 0:
                logger.info(f"等待Worker处理... 已轮询 {poll_count} 次")
            time.sleep(0.1)

        if os.path.exists(input_path):
            os.remove(input_path)

        queue_status = await get_queue_status()
        detail = (
            f"搜索超时（{timeout}秒）。"
            f"待处理任务: {queue_status.get('pending_tasks', 0)}，"
            f"Worker推荐: {queue_status.get('worker_recommendation', '检查Worker状态')}"
        )
        raise HTTPException(status_code=504, detail=detail)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {e}")


@app.get("/api/search/stats")
async def get_search_stats():
    """获取搜索服务统计信息"""
    input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]
    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")]

    return {
        "index_count": 1000,
        "model_name": "CLIP (Worker Process)",
        "status": "running",
        "queue_input_count": len(input_files),
        "queue_output_count": len(output_files),
        "pending_tasks": len(input_files),
        "worker_status": "running" if input_files or output_files else "idle",
    }


@app.get("/api/search/queue-status")
async def get_queue_status():
    """获取队列详细状态，帮助诊断超时问题"""
    input_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")])
    output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")])

    oldest_task = None
    if input_files:
        oldest_task = os.path.getctime(os.path.join(INPUT_DIR, input_files[0]))
        oldest_task = time.time() - oldest_task

    return {
        "queue_dir": QUEUE_DIR,
        "input_dir_exists": os.path.exists(INPUT_DIR),
        "output_dir_exists": os.path.exists(OUTPUT_DIR),
        "pending_tasks": len(input_files),
        "completed_tasks": len(output_files),
        "oldest_task_age_seconds": round(oldest_task, 2) if oldest_task else None,
        "worker_recommendation": "请确保 search_worker.py 进程正在运行",
    }


@app.post("/api/search/build-index")
async def build_search_index(dataset_dir: str = "data/test"):
    """构建搜索索引（模拟）"""
    return {
        "status": "success",
        "message": f"索引构建完成，数据集目录: {dataset_dir}",
        "indexed_count": 100,
    }


@app.post("/api/video/recognize")
async def recognize_video(
    file: UploadFile = File(...), frame_interval: float = 1.0, confidence_threshold: float = 0.5
):
    """视频文件角色识别（模拟）"""
    return {
        "status": "success",
        "filename": file.filename,
        "frame_count": 10,
        "results": [
            {"frame": 1, "role": "Madoka", "confidence": 0.95},
            {"frame": 5, "role": "Homura", "confidence": 0.88},
            {"frame": 9, "role": "Sayaka", "confidence": 0.72},
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
