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
sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.services.multimedia_service.routes import router
from loguru import logger

app = FastAPI(title="Multimedia Service", description="多媒体服务 - 整合图像搜索和视频识别功能", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="多媒体服务")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)