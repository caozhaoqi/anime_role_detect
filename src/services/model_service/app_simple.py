#!/usr/bin/env python3
"""
简化版模型服务
只提供基本的健康检查和文档端点
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 初始化FastAPI应用
app = FastAPI(
    title="Model Service",
    description="Anime Role Detect Model Service",
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

# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {"message": "Model Service", "docs": "/docs"}

# 健康检查
@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "Model Service"}

# 模型预测（简化版）
@app.post("/api/model/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("mobilenet_v2"),
    use_attributes: bool = Form(True)
):
    """
    模型预测
    
    Args:
        file: 上传的图像文件
        model_name: 模型名称
        use_attributes: 是否使用属性
    
    Returns:
        预测结果
    """
    try:
        # 简化版：返回模拟结果
        return {
            "role": "unknown",
            "similarity": 0.0,
            "tags": ["anime", "digital art"],
            "nsfw": {
                "is_nsfw": False,
                "details": {}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 特征提取（简化版）
@app.post("/api/model/extract")
async def extract_features(
    file: UploadFile = File(...)
):
    """
    特征提取
    
    Args:
        file: 上传的图像文件
    
    Returns:
        特征向量
    """
    try:
        # 简化版：返回随机特征
        import numpy as np
        return {
            "feature": np.random.rand(512).tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 兼容前端的分类端点
@app.post("/api/classify")
async def classify_image(
    file: UploadFile = File(...),
    use_coreml: bool = Form(False),
    use_model: bool = Form(True),
    use_attributes: bool = Form(True),
    model_name: str = Form("mobilenet_v2"),
    cache_bypass: bool = Form(False)
):
    """
    分类图像（兼容前端调用）
    """
    return await predict_image(
        file=file,
        model_name=model_name,
        use_attributes=use_attributes
    )

# 主函数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    
    args = parser.parse_args()
    
    # 启动服务
    uvicorn.run(
        "app_simple:app",
        host=args.host,
        port=args.port,
        timeout_keep_alive=30
    )
