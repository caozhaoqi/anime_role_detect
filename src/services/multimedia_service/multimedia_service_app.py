#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多媒体服务 - 整合图像搜索和视频识别功能
"""

import os
import sys
import io
import cv2
import time
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from PIL import Image
import uvicorn

# ==================== macOS OpenMP 冲突修复 ====================
# 在导入任何第三方库之前设置环境变量
# 1. 允许重复加载 OpenMP 运行时（避免初始化报错）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. 限制 OpenMP/MKL 线程数，防止多运行时争抢 CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 3. macOS fork 安全
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# 4. PyTorch MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# ============================================================

# 添加项目根目录到Python路径
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

# 延迟导入
from src.services.search_service.simple_search_service import SimpleImageSearchService
from loguru import logger
# 延迟初始化搜索服务
search_service = None

# 线程锁 - 保护模型推理（双重保护：环境变量 + 锁）
inference_lock = threading.Lock()


def init_search_service():
    """延迟初始化搜索服务"""
    global search_service
    if search_service is None:
        # 限制 PyTorch 线程数（防止 OpenMP 冲突）
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except ImportError:
            logger.warning("PyTorch is not installed, cannot limit torch threads.")
            pass
        
        search_service = SimpleImageSearchService()
        # SimpleImageSearchService 使用懒加载，首次调用 search() 时自动初始化
        # 这里预初始化以加速首次请求
        search_service._ensure_initialized()
        logger.info("Search service initialized.")
    return search_service


# 创建FastAPI应用
app = FastAPI(
    title="Multimedia Service",
    description="多媒体服务 - 整合图像搜索和视频识别功能",
    version="1.0.0",
)


# ==================== 健康检查 ====================
@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/api/health")
async def api_health_check():
    """统一API健康检查"""
    return {"status": "healthy", "service": "multimedia_service", "version": "1.0.0"}


# ==================== 图像搜索接口 ====================
@app.post("/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = Query(10, ge=1, le=50)):
    """
    以图搜图 - 上传图像搜索相似角色

    Args:
        file: 图像文件
        top_k: 返回结果数量

    Returns:
        相似角色列表
    """
    try:
        service = init_search_service()

        # 读取图像
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 搜索相似图像（使用线程锁保护，防止macOS段错误）
        with inference_lock:
            results = service.search(image, top_k=top_k)

        # 处理结果
        response_results = []
        for path, similarity in results:
            role = os.path.basename(os.path.dirname(path))
            response_results.append({"path": path, "similarity": float(similarity), "role": role})

        return {
            "success": True,
            "query": file.filename,
            "count": len(response_results),
            "results": response_results,
        }

    except Exception as e:
        import traceback

        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@app.get("/search/stats")
async def get_search_stats():
    """获取搜索索引统计信息"""
    try:
        service = init_search_service()
        return {"success": True, "data": service.get_index_stats()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/search/build-index")
async def build_index():
    """重建搜索索引"""
    try:
        service = init_search_service()
        service.build_index()
        return {"success": True, "message": "索引构建完成"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==================== 视频处理接口 ====================
@app.post("/video/extract")
async def extract_frames(
    file: UploadFile = File(...), frame_interval: float = Query(1.0, ge=0.1, le=10.0)
):
    """
    从视频中提取帧

    Args:
        file: 视频文件
        frame_interval: 抽帧间隔（秒）

    Returns:
        帧信息列表
    """
    try:
        # 读取视频文件
        content = await file.read()

        # 保存临时文件
        temp_path = f"/tmp/video_{int(time.time())}.mp4"
        with open(temp_path, "wb") as f:
            f.write(content)

        # 打开视频
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            os.remove(temp_path)
            return {"success": False, "error": "无法打开视频文件"}

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # 计算抽帧间隔（帧数）
        frame_interval_frames = int(fps * frame_interval)

        frames_info = []
        frame_count = 0
        success, frame = cap.read()

        while success:
            if frame_count % frame_interval_frames == 0:
                timestamp = frame_count / fps
                frames_info.append(
                    {
                        "frame_number": frame_count,
                        "timestamp": round(timestamp, 2),
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                    }
                )

            frame_count += 1
            success, frame = cap.read()

        cap.release()
        os.remove(temp_path)

        return {
            "success": True,
            "filename": file.filename,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration": round(duration, 2),
            "extracted_frames": len(frames_info),
            "frames": frames_info,
        }

    except Exception as e:
        import traceback

        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@app.post("/video/recognize")
async def recognize_video(
    file: UploadFile = File(...),
    frame_interval: float = Query(1.0, ge=0.1, le=10.0),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0),
    top_k: int = Query(3, ge=1, le=10),
):
    """
    视频实时抽帧识别

    Args:
        file: 视频文件
        frame_interval: 抽帧间隔（秒）
        confidence_threshold: 置信度阈值
        top_k: 每个帧返回的匹配数量

    Returns:
        识别结果列表，包含时间戳和识别出的角色
    """
    try:
        # 初始化搜索服务
        service = init_search_service()

        # 读取视频文件
        content = await file.read()

        # 保存临时文件
        temp_path = f"/tmp/video_{int(time.time())}.mp4"
        with open(temp_path, "wb") as f:
            f.write(content)

        # 打开视频
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            os.remove(temp_path)
            return {"success": False, "error": "无法打开视频文件"}

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算抽帧间隔（帧数）
        frame_interval_frames = int(fps * frame_interval)

        recognition_results = []
        frame_count = 0
        success, frame = cap.read()

        while success:
            if frame_count % frame_interval_frames == 0:
                # 计算时间戳
                timestamp = frame_count / fps

                # 转换为PIL图像
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                # 搜索相似图像（使用线程锁保护，防止macOS段错误）
                with inference_lock:
                    results = service.search(image, top_k=top_k)

                # 过滤置信度
                matched_roles = []
                for role, similarity in results:
                    if similarity >= confidence_threshold:
                        # search() 已经返回角色名，不需要从路径提取
                        matched_roles.append(
                            {"role": role, "similarity": round(float(similarity), 4)}
                        )

                if matched_roles:
                    recognition_results.append(
                        {
                            "timestamp": round(timestamp, 2),
                            "frame_number": frame_count,
                            "roles": matched_roles,
                        }
                    )

            frame_count += 1
            success, frame = cap.read()

        cap.release()
        os.remove(temp_path)

        return {
            "success": True,
            "filename": file.filename,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "frame_interval": frame_interval,
            "recognized_timestamps": len(recognition_results),
            "results": recognition_results,
        }

    except Exception as e:
        import traceback

        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@app.get("/video/stats")
async def get_video_stats():
    """获取视频服务统计信息"""
    return {
        "status": "running",
        "service": "multimedia_service",
        "features": ["image_search", "video_recognition"],
    }


# ==================== 服务信息 ====================
@app.get("/info")
async def get_service_info():
    """获取服务信息"""
    return {
        "service": "multimedia_service",
        "version": "1.0.0",
        "features": [
            {"name": "图像搜索", "endpoint": "/search/image"},
            {"name": "视频识别", "endpoint": "/video/recognize"},
            {"name": "视频抽帧", "endpoint": "/video/extract"},
        ],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多媒体服务")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
