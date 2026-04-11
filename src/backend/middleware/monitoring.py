#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控中间件
负责监控API请求和响应
"""

import time
from fastapi.responses import JSONResponse
from src.core.logging.global_logger import get_logger

logger = get_logger("monitoring")

# 服务监控信息
service_monitor = {
    "start_time": time.time(),
    "request_count": 0,
    "error_count": 0,
    "last_request_time": None
}


async def monitoring_middleware(request, call_next):
    """
    服务监控中间件
    
    Args:
        request: 请求对象
        call_next: 下一个中间件或路由处理函数
    
    Returns:
        响应对象
    """
    global service_monitor
    
    # 更新请求计数
    service_monitor["request_count"] += 1
    service_monitor["last_request_time"] = time.time()
    
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # 更新错误计数
        service_monitor["error_count"] += 1
        logger.error(f"请求处理失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        
        # 构建错误响应
        error_response = {
            "error": True,
            "message": str(e),
            "detail": {
                "type": type(e).__name__,
                "traceback": traceback.format_exc().split('\n')
            }
        }
        
        # 返回500错误
        return JSONResponse(
            status_code=500,
            content=error_response
        )


def get_service_monitor():
    """
    获取服务监控信息
    
    Returns:
        dict: 服务监控信息
    """
    global service_monitor
    return service_monitor
