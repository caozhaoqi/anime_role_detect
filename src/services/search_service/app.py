#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像搜索与视频识别API服务
提供以图搜图和视频实时抽帧识别功能
"""

import os
import sys
import time
import tempfile
import threading
from typing import List, Dict, Optional

# 解决macOS上的Mutex锁失败问题
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# 设置环境变量，避免锁竞争问题和OpenMP冲突
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# 延迟导入核心模块
get_logger = None
ImageSearchService = None
VideoRecognitionService = None

def import_core_modules():
    """动态导入核心模块（仅导入非PyTorch依赖）"""
    global get_logger, ImageSearchService, VideoRecognitionService
    
    from src.core.logging.global_logger import get_logger
    from src.services.search_service.image_search_service import ImageSearchService
    from src.services.video_service.video_recognition_service import VideoRecognitionService

# 初始化日志（仅导入基础模块，不导入分类服务）
import_core_modules()
logger = get_logger("search_api_service")

# 添加线程锁
torch_import_lock = threading.Lock()
service_init_lock = threading.Lock()

# 全局服务实例（延迟初始化）
search_service = None
video_service = None

# 初始化FastAPI应用
app = FastAPI(
    title="图像搜索与视频识别API",
    description="提供以图搜图和视频实时抽帧识别功能",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_search_service():
    """延迟初始化搜索服务"""
    global search_service
    with service_init_lock:
        if search_service is None:
            logger.info("初始化图像搜索服务...")
            search_service = ImageSearchService()
            logger.info("图像搜索服务初始化完成")
    return search_service


def init_video_service():
    """延迟初始化视频识别服务"""
    global video_service
    with service_init_lock:
        if video_service is None:
            logger.info("初始化视频识别服务...")
            video_service = VideoRecognitionService(
                frame_interval=1.0,
                confidence_threshold=0.5
            )
            logger.info("视频识别服务初始化完成")
    return video_service


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    logger.info("启动搜索服务API")
    # 服务延迟初始化，不在启动时加载模型


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "Search Service"}


# ========== 以图搜图 API ==========

@app.post("/api/search/image")
async def search_similar_images(
    file: UploadFile = File(...),
    top_k: int = 10
):
    """
    以图搜图 - 搜索相似图像
    
    Args:
        file: 上传的查询图像
        top_k: 返回前k个相似图像
    
    Returns:
        相似图像列表，包含路径和相似度
    """
    try:
        # 延迟初始化服务
        service = init_search_service()
        
        # 读取图像
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        
        # 搜索相似图像
        results = service.search(image, top_k)
        
        # 构建响应
        response = {
            "query": file.filename,
            "count": len(results),
            "results": [
                {
                    "path": path,
                    "similarity": similarity,
                    "role": os.path.basename(os.path.dirname(path))  # 从路径提取角色名
                }
                for path, similarity in results
            ]
        }
        
        logger.info(f"以图搜图完成: {file.filename}, 找到 {len(results)} 个相似图像")
        return response
    
    except Exception as e:
        logger.error(f"以图搜图失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {e}")


@app.post("/api/search/build-index")
async def build_search_index(
    dataset_dir: str = "data/merged_english_dataset"
):
    """
    构建搜索索引
    
    Args:
        dataset_dir: 数据集目录
    
    Returns:
        索引构建结果
    """
    try:
        # 延迟初始化服务
        service = init_search_service()
        
        logger.info(f"开始构建索引，数据集目录: {dataset_dir}")
        
        # 检查目录是否存在
        full_path = os.path.join(project_root, dataset_dir)
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"数据集目录不存在: {full_path}")
        
        # 构建索引
        count = service.build_index_from_dataset(full_path)
        service.save_index()
        
        # 获取统计信息
        stats = service.get_index_stats()
        
        response = {
            "status": "success",
            "dataset_dir": full_path,
            "added_images": count,
            "index_stats": stats
        }
        
        logger.info(f"索引构建完成，共添加 {count} 张图像")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"构建索引失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"构建索引失败: {e}")


@app.get("/api/search/stats")
async def get_search_stats():
    """
    获取搜索服务统计信息
    
    Returns:
        索引统计信息
    """
    try:
        # 延迟初始化服务
        service = init_search_service()
        
        return service.get_index_stats()
    
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {e}")


# ========== 视频识别 API ==========

@app.post("/api/video/recognize")
async def recognize_video(
    file: UploadFile = File(...),
    frame_interval: float = 1.0,
    confidence_threshold: float = 0.5
):
    """
    视频文件角色识别
    
    Args:
        file: 上传的视频文件
        frame_interval: 抽帧间隔（秒）
        confidence_threshold: 置信度阈值
    
    Returns:
        识别结果列表
    """
    try:
        # 保存上传的视频文件
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_video_path = temp_file.name
        
        logger.info(f"开始处理视频: {file.filename}")
        
        # 创建视频识别服务实例
        service = VideoRecognitionService(
            frame_interval=frame_interval,
            confidence_threshold=confidence_threshold
        )
        
        # 处理视频
        results = service.process_video_file(temp_video_path)
        
        # 清理临时文件
        os.remove(temp_video_path)
        
        # 统计识别到的角色
        role_counts = {}
        for result in results:
            role = result['role']
            role_counts[role] = role_counts.get(role, 0) + 1
        
        # 构建响应
        response = {
            "video": file.filename,
            "frame_interval": frame_interval,
            "total_frames": service.frame_count,
            "detections": len(results),
            "roles": [
                {"role": role, "count": count}
                for role, count in sorted(role_counts.items(), key=lambda x: x[1], reverse=True)
            ],
            "timestamps": [
                {
                    "timestamp": result['timestamp'],
                    "role": result['role'],
                    "similarity": result['similarity'],
                    "attributes": [attr['tag'] for attr in result.get('attributes', [])[:5]]
                }
                for result in results
            ]
        }
        
        logger.info(f"视频识别完成: {file.filename}, 检测到 {len(role_counts)} 个角色")
        return response
    
    except Exception as e:
        logger.error(f"视频识别失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"视频识别失败: {e}")


@app.post("/api/video/realtime/start")
async def start_realtime_recognition(
    frame_interval: float = 1.0,
    confidence_threshold: float = 0.5,
    video_source: int = 0
):
    """
    启动实时视频识别（摄像头）
    
    Args:
        frame_interval: 抽帧间隔（秒）
        confidence_threshold: 置信度阈值
        video_source: 视频源（0为默认摄像头）
    
    Returns:
        启动状态
    """
    try:
        global video_service
        
        # 创建新的视频识别服务
        video_service = VideoRecognitionService(
            frame_interval=frame_interval,
            confidence_threshold=confidence_threshold
        )
        
        # 启动服务（后台运行）
        thread = threading.Thread(
            target=video_service.process_realtime,
            args=(video_source,),
            daemon=True
        )
        thread.start()
        
        return {
            "status": "started",
            "message": "实时视频识别已启动",
            "frame_interval": frame_interval,
            "confidence_threshold": confidence_threshold
        }
    
    except Exception as e:
        logger.error(f"启动实时识别失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"启动失败: {e}")


@app.get("/api/video/realtime/stop")
async def stop_realtime_recognition():
    """
    停止实时视频识别
    
    Returns:
        停止状态和统计信息
    """
    try:
        global video_service
        
        if video_service is None:
            raise HTTPException(status_code=400, detail="实时识别未启动")
        
        stats = video_service.get_stats()
        video_service.stop()
        
        return {
            "status": "stopped",
            "message": "实时视频识别已停止",
            "stats": stats
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止实时识别失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"停止失败: {e}")


@app.get("/api/video/realtime/stats")
async def get_realtime_stats():
    """
    获取实时识别统计信息
    
    Returns:
        统计信息
    """
    try:
        global video_service
        
        if video_service is None:
            raise HTTPException(status_code=400, detail="实时识别未启动")
        
        return video_service.get_stats()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取实时统计信息失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {e}")


# ========== 弹幕模式 API ==========

# 弹幕消息队列（用于实时推送）
danmaku_queue = []
danmaku_lock = threading.Lock()


def danmaku_callback(result: Dict):
    """弹幕回调函数"""
    with danmaku_lock:
        # 保持队列大小
        if len(danmaku_queue) > 100:
            danmaku_queue.pop(0)
        
        danmaku_queue.append({
            "timestamp": result['timestamp'],
            "role": result['role'],
            "similarity": result['similarity'],
            "time": time.strftime("%H:%M:%S", time.localtime())
        })


@app.get("/api/danmaku/latest")
async def get_latest_danmaku(count: int = 10):
    """
    获取最新的弹幕消息
    
    Args:
        count: 返回数量
    
    Returns:
        弹幕消息列表
    """
    with danmaku_lock:
        return {
            "count": len(danmaku_queue),
            "messages": danmaku_queue[-count:]
        }


@app.post("/api/video/danmaku/start")
async def start_danmaku_mode(
    video_source: int = 0,
    frame_interval: float = 1.0,
    confidence_threshold: float = 0.5
):
    """
    启动弹幕模式 - 实时识别并推送角色信息
    
    Args:
        video_source: 视频源
        frame_interval: 抽帧间隔
        confidence_threshold: 置信度阈值
    
    Returns:
        启动状态
    """
    try:
        global video_service
        
        # 创建视频识别服务并设置回调
        video_service = VideoRecognitionService(
            frame_interval=frame_interval,
            confidence_threshold=confidence_threshold
        )
        video_service.set_result_callback(danmaku_callback)
        
        # 清空弹幕队列
        with danmaku_lock:
            danmaku_queue.clear()
        
        # 启动服务
        thread = threading.Thread(
            target=video_service.process_realtime,
            args=(video_source,),
            daemon=True
        )
        thread.start()
        
        return {
            "status": "started",
            "message": "弹幕模式已启动",
            "tip": "访问 /api/danmaku/latest 获取实时弹幕消息"
        }
    
    except Exception as e:
        logger.error(f"启动弹幕模式失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"启动失败: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="图像搜索与视频识别API服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机")
    parser.add_argument("--port", type=int, default=8001, help="服务端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    
    args = parser.parse_args()
    
    # 启动服务
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        workers=args.workers
    )
