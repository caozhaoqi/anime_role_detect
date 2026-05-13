#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试导入搜索服务模块
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

print("Step 1: 添加项目路径")
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Step 2: 导入搜索服务")
try:
    from src.services.search_service.image_search_service import ImageSearchService
    print("搜索服务导入成功")
except Exception as e:
    print(f"搜索服务导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成！")
