#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控中间件
负责监控API请求和响应
"""

import time
import numpy as np
from fastapi.responses import JSONResponse
from src.core.logging.global_logger import get_logger

logger = get_logger("monitoring")

def convert_numpy_types(obj):
    """
    转换numpy类型为Python原生类型
    
    Args:
        obj: 要转换的对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# 服务监控信息
service_monitor = {
    "start_time": time.time(),
    "request_count": 0,
    "error_count": 0,
    "last_request_time": None,
    "total_response_time": 0,
    "request_types": {},
    "endpoint_stats": {},
    "status_codes": {}
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
    
    # 记录请求开始时间
    start_time = time.time()
    endpoint = str(request.url.path)
    method = request.method
    
    # 更新请求计数
    service_monitor["request_count"] += 1
    service_monitor["last_request_time"] = time.time()
    
    # 更新请求类型统计
    if method not in service_monitor["request_types"]:
        service_monitor["request_types"][method] = 0
    service_monitor["request_types"][method] += 1
    
    # 更新端点统计
    if endpoint not in service_monitor["endpoint_stats"]:
        service_monitor["endpoint_stats"][endpoint] = {
            "count": 0,
            "total_time": 0,
            "errors": 0
        }
    service_monitor["endpoint_stats"][endpoint]["count"] += 1
    
    try:
        response = await call_next(request)
        
        # 计算响应时间
        response_time = time.time() - start_time
        service_monitor["total_response_time"] += response_time
        service_monitor["endpoint_stats"][endpoint]["total_time"] += response_time
        
        # 更新状态码统计
        status_code = response.status_code
        if status_code not in service_monitor["status_codes"]:
            service_monitor["status_codes"][status_code] = 0
        service_monitor["status_codes"][status_code] += 1
        
        return response
    except Exception as e:
        # 计算响应时间
        response_time = time.time() - start_time
        service_monitor["total_response_time"] += response_time
        service_monitor["endpoint_stats"][endpoint]["total_time"] += response_time
        
        # 更新错误计数
        service_monitor["error_count"] += 1
        service_monitor["endpoint_stats"][endpoint]["errors"] += 1
        
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
        
        # 转换numpy类型
        error_response = convert_numpy_types(error_response)
        
        # 更新状态码统计
        status_code = 500
        if status_code not in service_monitor["status_codes"]:
            service_monitor["status_codes"][status_code] = 0
        service_monitor["status_codes"][status_code] += 1
        
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
    
    # 计算平均响应时间
    avg_response_time = 0
    if service_monitor["request_count"] > 0:
        avg_response_time = service_monitor["total_response_time"] / service_monitor["request_count"]
    
    # 计算错误率
    error_rate = 0
    if service_monitor["request_count"] > 0:
        error_rate = (service_monitor["error_count"] / service_monitor["request_count"]) * 100
    
    # 计算服务运行时间
    uptime = time.time() - service_monitor["start_time"]
    
    # 构建完整的监控信息
    monitor_info = {
        **service_monitor,
        "average_response_time": avg_response_time,
        "error_rate": error_rate,
        "uptime": uptime,
        "uptime_hours": uptime / 3600
    }
    
    return monitor_info
