#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频识别API路由
提供视频文件识别和实时视频流识别功能
"""

import os
import sys
import tempfile
from fastapi import APIRouter, File, UploadFile, Form, Query
from typing import Optional
from fastapi.responses import JSONResponse

# 添加项目根目录到Python路径
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("video_routes")

router = APIRouter(prefix="/api/video", tags=["视频识别"])


@router.post("/recognize")
async def recognize_video(
    file: UploadFile = File(...),
    frame_interval: float = Query(1.0, description="抽帧间隔（秒），前端经 query 传入"),
    confidence_threshold: float = Query(0.5, description="识别置信度阈值，前端经 query 传入"),
):
    """
    视频识别端点 - 处理上传的视频文件并识别其中的角色

    Args:
        file: 上传的视频文件
        frame_interval: 抽帧间隔（秒），默认1秒
        confidence_threshold: 识别置信度阈值，默认0.5

    Returns:
        dict: 视频识别结果
    """
    try:
        logger.info(f"接收到视频识别请求: {file.filename}, 帧间隔: {frame_interval}, 置信度阈值: {confidence_threshold}")

        # 验证文件类型
        allowed_types = ["video/mp4", "video/mpeg", "video/avi", "video/webm", "video/mov"]
        if file.content_type not in allowed_types:
            logger.error(f"不支持的视频格式: {file.content_type}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": f"不支持的视频格式，请上传MP4、AVI、WebM或MOV格式的视频"}
            )

        # 保存上传的视频文件到临时目录
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        logger.info(f"视频文件已保存到临时位置: {temp_path}")

        # 导入视频识别服务
        from src.services.video_service.video_recognition_service import VideoRecognitionService
        
        # 创建视频识别服务实例
        service = VideoRecognitionService(
            frame_interval=frame_interval,
            confidence_threshold=confidence_threshold
        )

        # 处理视频文件
        results = service.process_video_file(temp_path)

        # 清理临时文件
        os.unlink(temp_path)

        # 统计识别结果
        roles_found = {}
        formatted_results = []
        
        for result in results:
            role = result["role"]
            roles_found[role] = roles_found.get(role, 0) + 1
            
            # 转换为前端期望的格式
            formatted_result = {
                "timestamp": result["timestamp"],
                "frame_index": result["frame_number"],
                "roles": [{
                    "role": result["role"],
                    "similarity": result["similarity"],
                    "box": result.get("boxes", [{}])[0] if result.get("boxes") else None
                }]
            }
            formatted_results.append(formatted_result)

        # 构建响应 - 同时返回两种格式以兼容前端
        response = {
            "success": True,
            "message": "视频识别完成",
            "data": {
                "total_frames_processed": service.frame_count,
                "total_detections": len(results),
                "roles_found": roles_found,
                "detections": results,
                "results": formatted_results,  # 前端期望的格式
                "frame_interval": frame_interval,
                "confidence_threshold": confidence_threshold,
            },
        }

        logger.info(f"视频识别完成：{len(results)} 个检测结果，{len(roles_found)} 个不同角色")
        
        return response

    except Exception as e:
        logger.error(f"视频识别失败: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"视频识别失败: {str(e)}"}
        )


@router.post("/recognize/realtime")
async def recognize_realtime(
    video_source: Optional[str] = Form("0"),
    frame_interval: float = Form(1.0),
    confidence_threshold: float = Form(0.5),
):
    """
    实时视频识别端点 - 处理实时视频流（摄像头或RTSP流）

    Args:
        video_source: 视频源（0为默认摄像头，或RTSP URL）
        frame_interval: 抽帧间隔（秒），默认1秒
        confidence_threshold: 识别置信度阈值，默认0.5

    Returns:
        dict: 服务启动状态
    """
    try:
        # 特性开关：实时端点默认关闭。VideoRecognitionService.start() 会阻塞事件循环，
        # 且当前无 WebSocket/轮询消费方，生产环境启用需谨慎（配 ARD_ENABLE_REALTIME_VIDEO=true）
        from src.config import config
        if not config.app_features.ENABLE_REALTIME_VIDEO:
            logger.warning("实时视频识别端点被调用，但该特性未启用（ARD_ENABLE_REALTIME_VIDEO=false）")
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "message": "实时视频识别功能当前未启用。如需开启，请设置环境变量 ARD_ENABLE_REALTIME_VIDEO=true。",
                    "data": {"status": "disabled", "feature": "realtime_video"},
                },
            )

        logger.info(f"接收到实时视频识别请求: {video_source}, 帧间隔: {frame_interval}, 置信度阈值: {confidence_threshold}")

        # 导入视频识别服务
        from src.services.video_service.video_recognition_service import VideoRecognitionService
        
        # 创建视频识别服务实例
        service = VideoRecognitionService(
            frame_interval=frame_interval,
            confidence_threshold=confidence_threshold
        )

        # 定义回调函数处理实时结果
        results = []
        
        def handle_result(result):
            results.append(result)
            logger.info(f"[{result['timestamp']:.2f}s] 检测到角色: {result['role']} (相似度: {result['similarity']:.4f})")
        
        service.set_result_callback(handle_result)

        # 尝试解析视频源
        try:
            source = int(video_source)
        except ValueError:
            source = video_source

        # 启动实时处理（这会阻塞，需要异步处理）
        # 对于实时模式，这里返回启动状态，实际结果需要通过WebSocket或轮询获取
        service.start()

        return {
            "success": True,
            "message": "实时视频识别服务已启动",
            "data": {
                "video_source": video_source,
                "frame_interval": frame_interval,
                "confidence_threshold": confidence_threshold,
                "status": "running",
            },
        }

    except Exception as e:
        logger.error(f"启动实时视频识别失败: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"启动实时视频识别失败: {str(e)}"}
        )


@router.get("/recognize/status")
async def get_recognition_status():
    """
    获取视频识别服务状态

    Returns:
        dict: 服务状态
    """
    try:
        from src.services.video_service.video_recognition_service import VideoRecognitionService
        
        # 这里应该获取全局服务实例的状态
        # 由于服务可能是动态创建的，这里返回通用状态
        return {
            "success": True,
            "message": "获取状态成功",
            "data": {
                "status": "available",
                "message": "视频识别服务就绪",
            },
        }

    except Exception as e:
        logger.error(f"获取视频识别状态失败: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"获取状态失败: {str(e)}"}
        )



