#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多媒体服务主文件 - 只负责初始化和生命周期管理
"""
import os
import sys

# macOS OpenMP 冲突修复
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.services.multimedia_service.routes import router
from loguru import logger

app = FastAPI(title="Multimedia Service", description="多媒体服务 - 整合图像搜索和视频识别功能", version="1.0.0")

# 配置CORS：生产环境应通过环境变量限定
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
if _cors_origins_env:
    _allowed_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
else:
    _allowed_origins = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Accept",
        "Origin",
    ],
)
app.include_router(router)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="多媒体服务")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
