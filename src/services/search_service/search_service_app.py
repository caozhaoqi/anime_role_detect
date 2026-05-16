#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的搜索服务应用 - 使用简化版搜索服务，避免macOS上PyTorch锁问题
"""

import os
import sys
import io

# 设置环境变量
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# 添加项目根目录到Python路径
# 当前文件: src/services/search_service/search_service_app.py
# 需要向上走4级目录才能到达项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(f"项目根目录: {project_root}")
sys.path.insert(0, project_root)

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import uvicorn

# 使用简化版搜索服务（不依赖CLIP模型）
from src.services.search_service.simple_search_service import SimpleImageSearchService

# 延迟初始化搜索服务
search_service = None

def init_search_service():
    """延迟初始化搜索服务"""
    global search_service
    if search_service is None:
        search_service = SimpleImageSearchService()
        # 尝试加载已保存的索引
        search_service.load_index()
    return search_service

# 创建FastAPI应用
app = FastAPI(
    title="Search Service",
    description="独立的图像搜索服务 - 使用传统图像特征",
    version="1.0.0"
)

@app.post("/search/image")
async def search_similar_images(file: UploadFile = File(...), top_k: int = 10):
    """
    以图搜图 - 搜索相似图像
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
            "success": True,
            "query": file.filename,
            "count": len(results),
            "results": [
                {
                    "path": path,
                    "similarity": float(similarity),
                    "role": os.path.basename(os.path.dirname(path))
                }
                for path, similarity in results
            ]
        }
        
        return response
    
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

@app.post("/search/build-index")
async def build_search_index(dataset_dir: str = "data/merged_english_dataset"):
    """
    构建搜索索引
    """
    try:
        service = init_search_service()
        
        # 检查目录是否存在
        full_path = os.path.join(project_root, dataset_dir)
        if not os.path.exists(full_path):
            return {"success": False, "error": f"数据集目录不存在: {full_path}"}
        
        # 构建索引
        count = service.build_index_from_dataset(full_path)
        service.save_index()
        
        stats = service.get_index_stats()
        
        return {
            "success": True,
            "dataset_dir": full_path,
            "added_images": count,
            "index_stats": stats
        }
    
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

@app.get("/search/stats")
async def get_search_stats():
    """
    获取搜索服务统计信息
    """
    try:
        service = init_search_service()
        return {"success": True, "data": service.get_index_stats()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# @app.get("/health")
# async def health_check():
#     """健康检查"""
#     return {"status": "healthy"}

@app.get("/api/health")
async def api_health_check():
    """统一API健康检查"""
    return {"status": "healthy", "service": "search_service", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, workers=1)