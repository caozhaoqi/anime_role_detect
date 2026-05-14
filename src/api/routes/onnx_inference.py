#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 推理 API 路由
提供高性能推理服务
"""

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import time
from typing import List, Optional

# 导入 ONNX 引擎
from src.core.onnx_engine import get_engine, list_available_models, ModelManager

# 延迟导入指标收集器，避免重复注册
def record_metrics(endpoint, method, status, model_name=None, duration=None):
    try:
        from src.core.prometheus_metrics import MetricsCollector
        MetricsCollector.record_request(endpoint=endpoint, method=method, status=status)
        if model_name and duration:
            MetricsCollector.record_inference_time(model_name=model_name, duration=duration)
            MetricsCollector.record_api_response_time(endpoint=endpoint, duration=duration)
    except Exception:
        pass  # 指标收集失败不影响业务

router = APIRouter(prefix="/api/v1/onnx", tags=["ONNX Inference"])

@router.get("/models", summary="获取可用模型列表")
async def get_models():
    """获取所有可用的 ONNX 模型列表"""
    models = list_available_models()
    return {"models": models}

@router.post("/predict/{model_name}", summary="单张图片推理")
async def predict(
    model_name: str,
    file: UploadFile = File(...),
    use_gpu: Optional[bool] = False
):
    """
    使用指定模型进行单张图片推理
    
    Args:
        model_name: 模型名称
        file: 上传的图片文件
        use_gpu: 是否使用 GPU 推理
    """
    start_time = time.time()
    
    try:
        # 加载模型
        engine = get_engine(model_name, use_gpu=use_gpu)
        
        # 读取图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 推理
        result = engine.predict(image)
        
        # 计算耗时
        duration = time.time() - start_time
        
        # 记录指标
        record_metrics(endpoint="/api/v1/onnx/predict", method="POST", status=200, model_name=model_name, duration=duration)
        
        return {
            "model_name": model_name,
            "input_size": engine.input_size,
            "result_shape": result.shape,
            "inference_time_ms": duration * 1000,
            "top_prediction": int(result.argmax()) if len(result.shape) == 2 else "detection"
        }
    
    except FileNotFoundError as e:
        record_metrics(endpoint="/api/v1/onnx/predict", method="POST", status=404)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        record_metrics(endpoint="/api/v1/onnx/predict", method="POST", status=500)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch/{model_name}", summary="批量图片推理")
async def predict_batch(
    model_name: str,
    files: List[UploadFile] = File(...),
    use_gpu: Optional[bool] = False
):
    """
    使用指定模型进行批量图片推理
    
    Args:
        model_name: 模型名称
        files: 上传的图片文件列表
        use_gpu: 是否使用 GPU 推理
    """
    start_time = time.time()
    
    try:
        # 加载模型
        engine = get_engine(model_name, use_gpu=use_gpu)
        
        # 读取所有图片
        images = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(image)
        
        # 批量推理
        results = engine.predict_batch(images)
        
        # 计算耗时
        duration = time.time() - start_time
        
        # 记录指标
        record_metrics(endpoint="/api/v1/onnx/predict/batch", method="POST", status=200, model_name=model_name, duration=duration)
        
        return {
            "model_name": model_name,
            "input_size": engine.input_size,
            "batch_size": len(images),
            "result_shape": results.shape,
            "inference_time_ms": duration * 1000,
            "avg_time_per_image_ms": (duration / len(images)) * 1000 if images else 0
        }
    
    except FileNotFoundError as e:
        record_metrics(endpoint="/api/v1/onnx/predict/batch", method="POST", status=404)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        record_metrics(endpoint="/api/v1/onnx/predict/batch", method="POST", status=500)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/info/{model_name}", summary="获取模型信息")
async def get_model_info(model_name: str):
    """
    获取指定模型的详细信息
    
    Args:
        model_name: 模型名称
    """
    try:
        engine = get_engine(model_name)
        return {
            "model_name": model_name,
            "input_shape": engine.input_shape,
            "input_size": engine.input_size,
            "input_name": engine.input_name,
            "output_name": engine.output_name
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/model/{model_name}", summary="卸载模型")
async def unload_model(model_name: str):
    """
    从内存中卸载指定模型
    
    Args:
        model_name: 模型名称
    """
    from src.core.onnx_engine import release_engine
    
    try:
        release_engine(model_name)
        return {"message": f"模型 {model_name} 已卸载"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
