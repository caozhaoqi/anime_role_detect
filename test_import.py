#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试导入顺序，定位Mutex锁问题
"""

import os
import sys

# 设置环境变量
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("Step 1: 基础导入")
import time
import threading
from typing import List, Dict, Optional

print("Step 2: FastAPI导入")
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io

print("Step 3: 项目模块导入")
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.logging.global_logger import get_logger
logger = get_logger("test_import")
print("日志模块导入成功")

print("Step 4: 延迟导入核心服务")
# 不立即导入，只检查路径
print("检查服务模块路径...")
service_path = os.path.join(project_root, "src", "services", "search_service", "image_search_service.py")
if os.path.exists(service_path):
    print(f"服务模块存在: {service_path}")
else:
    print(f"服务模块不存在: {service_path}")

print("\n测试完成！")
