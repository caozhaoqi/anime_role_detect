"""
多媒体服务路由
"""
import os
import sys
import io
import cv2
import time
import threading
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.services.search_service.simple_search_service import SimpleImageSearchService
from src.services.multimedia_service.video_renderer import render_result_video
import requests
from loguru import logger

router = APIRouter()

# 延迟初始化搜索服务
search_service = None

# 结果视频保存目录
RESULT_VIDEO_DIR = os.path.join(project_root, "data", "video_results")
os.makedirs(RESULT_VIDEO_DIR, exist_ok=True)

# Model Service URL (用于模型推理模式)
MODEL_SERVICE_URL = "http://localhost:8001"

# 线程锁
inference_lock = threading.Lock()


def init_search_service():
    """延迟初始化搜索服务"""
    global search_service
    if search_service is None:
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except ImportError:
            logger.warning("PyTorch is not installed, cannot limit torch threads.")

        search_service = SimpleImageSearchService()
        search_service._ensure_initialized()
        logger.info("Search service initialized.")
    return search_service


def classify_with_model(image: Image.Image, model_name: str = "efficientnet_b3_loli_optimized_v2_20260529_133654") -> list:
    """使用模型推理模式分类图像"""
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        data = {'model_name': model_name, 'use_attributes': False, 'cache_bypass': False}

        response = requests.post(f"{MODEL_SERVICE_URL}/api/classify", files=files, data=data, timeout=30)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return [(result.get('role', 'unknown'), result.get('similarity', 0.0))]
        return []
    except Exception as e:
        logger.error(f"Model classification error: {e}")
        return []


@router.get("/health")
@router.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "multimedia_service", "version": "1.0.0"}


@router.post("/search/image")
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Query(10, ge=1, le=50),
    recognition_mode: str = Query("search", pattern="^(search|inference)$"),
    model_name: str = Query("efficientnet_b3_loli_optimized_v2_20260529_133654"),
):
    """以图搜图"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        with inference_lock:
            if recognition_mode == "inference":
                results = classify_with_model(image, model_name=model_name)
            else:
                service = init_search_service()
                results = service.search(image, top_k=top_k)
        response_results = [{"role": role, "similarity": float(sim)} for role, sim in results]
        return {"success": True, "query": file.filename, "recognition_mode": recognition_mode,
                "count": len(response_results), "results": response_results}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/search/stats")
async def get_search_stats():
    try:
        service = init_search_service()
        return {"success": True, "data": service.get_index_stats()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/search/build-index")
async def build_index():
    try:
        service = init_search_service()
        service.build_index()
        return {"success": True, "message": "索引构建完成"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/video/extract")
async def extract_frames(file: UploadFile = File(...), frame_interval: float = Query(1.0, ge=0.1, le=10.0)):
    """从视频中提取帧"""
    try:
        content = await file.read()
        temp_path = f"/tmp/video_{int(time.time())}.mp4"
        with open(temp_path, "wb") as f:
            f.write(content)
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            os.remove(temp_path)
            return {"success": False, "error": "无法打开视频文件"}
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval_frames = int(fps * frame_interval)
        frames_info = []
        frame_count = 0
        success, frame = cap.read()
        while success:
            if frame_count % frame_interval_frames == 0:
                frames_info.append({"frame_number": frame_count, "timestamp": round(frame_count / fps, 2),
                                    "width": frame.shape[1], "height": frame.shape[0]})
            frame_count += 1
            success, frame = cap.read()
        cap.release()
        os.remove(temp_path)
        return {"success": True, "filename": file.filename, "fps": round(fps, 2), "total_frames": total_frames,
                "duration": round(total_frames / fps, 2), "extracted_frames": len(frames_info), "frames": frames_info}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/video/recognize")
async def recognize_video(
    file: UploadFile = File(...),
    frame_interval: float = Query(1.0, ge=0.1, le=10.0),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0),
    top_k: int = Query(3, ge=1, le=10),
    recognition_mode: str = Query("search", pattern="^(search|inference)$"),
    model_name: str = Query("efficientnet_b3_loli_optimized_v2_20260529_133654"),
):
    """视频实时抽帧识别"""
    try:
        content = await file.read()
        temp_path = f"/tmp/video_{int(time.time())}.mp4"
        with open(temp_path, "wb") as f:
            f.write(content)
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            os.remove(temp_path)
            return {"success": False, "error": "无法打开视频文件"}
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval_frames = int(fps * frame_interval)
        recognition_results = []
        frame_count = 0
        success, frame = cap.read()
        while success:
            if frame_count % frame_interval_frames == 0:
                timestamp_s = frame_count / fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                with inference_lock:
                    if recognition_mode == "inference":
                        results = classify_with_model(image, model_name=model_name)
                    else:
                        service = init_search_service()
                        results = service.search(image, top_k=top_k)
                matched_roles = [{"role": role, "similarity": round(float(sim), 4)} for role, sim in results if sim >= confidence_threshold]
                if matched_roles:
                    recognition_results.append({"timestamp": round(timestamp_s, 2), "frame_number": frame_count, "roles": matched_roles})
            frame_count += 1
            success, frame = cap.read()
        cap.release()
        os.remove(temp_path)
        return {"success": True, "filename": file.filename, "fps": round(fps, 2), "total_frames": total_frames,
                "frame_interval": frame_interval, "recognition_mode": recognition_mode,
                "recognized_timestamps": len(recognition_results), "results": recognition_results}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/video/recognize-with-overlay")
async def recognize_video_with_overlay(
    file: UploadFile = File(...),
    frame_interval: float = Query(1.0, ge=0.1, le=10.0),
    confidence_threshold: float = Query(0.5, ge=0.0, le=1.0),
    top_k: int = Query(3, ge=1, le=10),
    recognition_mode: str = Query("search", pattern="^(search|inference)$"),
    model_name: str = Query("efficientnet_b3_loli_optimized_v2_20260529_133654"),
):
    """视频识别并生成带标注的结果视频"""
    try:
        timestamp = int(time.time())
        content = await file.read()
        temp_path = f"/tmp/video_{timestamp}.mp4"
        with open(temp_path, "wb") as f:
            f.write(content)
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            os.remove(temp_path)
            return {"success": False, "error": "无法打开视频文件"}
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval_frames = int(fps * frame_interval)
        recognition_results = []
        frame_count = 0
        success, frame = cap.read()
        while success:
            if frame_count % frame_interval_frames == 0:
                timestamp_s = frame_count / fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                with inference_lock:
                    if recognition_mode == "inference":
                        results = classify_with_model(image, model_name=model_name)
                    else:
                        service = init_search_service()
                        results = service.search(image, top_k=top_k)
                matched_roles = [{"role": role, "similarity": round(float(sim), 4)} for role, sim in results if sim >= confidence_threshold]
                if matched_roles:
                    recognition_results.append({"timestamp": round(timestamp_s, 2), "frame_number": frame_count, "roles": matched_roles})
            frame_count += 1
            success, frame = cap.read()
        cap.release()
        result_filename = f"result_{timestamp}.mp4"
        output_path = os.path.join(RESULT_VIDEO_DIR, result_filename)
        rendered = render_result_video(video_path=temp_path, results=recognition_results, output_path=output_path, frame_interval=frame_interval)
        os.remove(temp_path)
        _cleanup_old_results(keep=20)
        return {"success": True, "filename": file.filename, "fps": round(fps, 2), "total_frames": total_frames,
                "frame_interval": frame_interval, "recognition_mode": recognition_mode,
                "recognized_timestamps": len(recognition_results), "results": recognition_results,
                "result_video_url": f"/api/video/result/{result_filename}" if rendered else None}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/video/result/{filename}")
async def download_result_video(filename: str):
    filepath = os.path.join(RESULT_VIDEO_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="结果视频不存在或已过期")
    return FileResponse(path=filepath, media_type="video/mp4", filename=filename,
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.post("/video/result/cleanup")
async def cleanup_result_videos():
    removed = _cleanup_old_results(keep=20)
    return {"success": True, "removed": removed}


def _cleanup_old_results(keep: int = 20):
    try:
        files = sorted([os.path.join(RESULT_VIDEO_DIR, f) for f in os.listdir(RESULT_VIDEO_DIR) if f.endswith(".mp4")], key=os.path.getmtime)
        removed = 0
        while len(files) > keep:
            os.remove(files.pop(0))
            removed += 1
        if removed:
            logger.info(f"清理了 {removed} 个旧结果视频")
        return removed
    except Exception:
        return 0


@router.get("/video/stats")
async def get_video_stats():
    return {"status": "running", "service": "multimedia_service", "features": ["image_search", "video_recognition"]}


@router.get("/info")
async def get_service_info():
    return {"service": "multimedia_service", "version": "1.0.0",
            "features": [
                {"name": "图像搜索", "endpoint": "/search/image"},
                {"name": "视频识别", "endpoint": "/video/recognize"},
                {"name": "视频识别+标注", "endpoint": "/video/recognize-with-overlay"},
                {"name": "识别结果视频下载", "endpoint": "/video/result/{filename}"},
                {"name": "视频抽帧", "endpoint": "/video/extract"},
            ]}