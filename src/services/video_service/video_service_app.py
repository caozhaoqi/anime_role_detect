#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频服务FastAPI应用
"""

import os
import sys
import io
import cv2
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import uvicorn

# 设置环境变量
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# 延迟导入
from src.services.search_service.simple_search_service import SimpleImageSearchService

# 延迟初始化搜索服务
search_service = None

def init_search_service():
    """延迟初始化搜索服务"""
    global search_service
    if search_service is None:
        search_service = SimpleImageSearchService()
        search_service.load_index()
    return search_service

# 创建FastAPI应用
app = FastAPI(
    title="Video Recognition Service",
    description="视频实时抽帧识别服务",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

@app.get("/api/health")
async def api_health_check():
    """统一API健康检查"""
    return {"status": "healthy", "service": "video_service", "version": "1.0.0"}

@app.post("/video/extract")
async def extract_frames(
    file: UploadFile = File(...),
    frame_interval: float = 1.0
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
        with open(temp_path, 'wb') as f:
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
                # 计算时间戳
                timestamp = frame_count / fps
                frames_info.append({
                    "frame_number": frame_count,
                    "timestamp": round(timestamp, 2),
                    "width": frame.shape[1],
                    "height": frame.shape[0]
                })
            
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
            "frames": frames_info
        }
    
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

@app.post("/video/recognize")
async def recognize_video(
    file: UploadFile = File(...),
    frame_interval: float = 1.0,
    confidence_threshold: float = 0.5
):
    """
    视频实时抽帧识别
    
    Args:
        file: 视频文件
        frame_interval: 抽帧间隔（秒）
        confidence_threshold: 置信度阈值
    
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
        with open(temp_path, 'wb') as f:
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
                
                # 搜索相似图像
                results = service.search(image, top_k=3)
                
                # 过滤置信度
                matched_roles = []
                for path, similarity in results:
                    if similarity >= confidence_threshold:
                        role = os.path.basename(os.path.dirname(path))
                        matched_roles.append({
                            "role": role,
                            "similarity": round(float(similarity), 4)
                        })
                
                if matched_roles:
                    recognition_results.append({
                        "timestamp": round(timestamp, 2),
                        "frame_number": frame_count,
                        "roles": matched_roles
                    })
            
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
            "results": recognition_results
        }
    
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

@app.get("/video/stats")
async def get_stats():
    """获取服务统计信息"""
    return {
        "status": "running",
        "service": "video_recognition_service"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)